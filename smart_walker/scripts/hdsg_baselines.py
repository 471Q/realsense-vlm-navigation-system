"""Offline comparison conditions and the frozen scorer for their explanations.

Chapter 5 compares three matched conditions over the same recorded observation and intent:

    C0_VLM_ONLY          the image alone, no measured facts
    C1_GROUNDED_UNGATED  the image plus the measured facts, no release enforcement
    C2_FULL_HDSG         the complete architecture

C0 and C1 run offline against a recording and cannot reach the walker display, per Chapter 5
section 5.2.3. Nothing in this module touches the live release path; it exists to produce and
score comparison evidence.

**What is being measured.** The thesis contribution is grounded navigational transparency: a
walker that intervenes must state the measured reason rather than withhold it. Five of Chapter 5
Table 5-9's six measures therefore concern the explanation, and only one concerns the action.
The scorer below is built around that: `score_cause` is the primary function here, and the action
and distance comparisons support it.

**Unscoreable is a real outcome.** Section 5.6.1 requires that wording the frozen scorer cannot
resolve is recorded as unscoreable rather than given a favourable reading, and that the
unscoreable denominator is reported. `CauseOutcome.UNSCOREABLE` exists for that and must never be
collapsed into a pass or a fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
import re
from typing import Any, Mapping, Optional

try:
    from . import hdsg_runtime as hdsg
except ImportError:  # invoked as a plain script rather than as part of the package
    import hdsg_runtime as hdsg  # type: ignore


BASELINE_RESPONSE_SCHEMA = "hdsg.baseline_response.v1"

CONDITION_C0 = "C0_VLM_ONLY"
CONDITION_C1 = "C1_GROUNDED_UNGATED"
CONDITION_C2 = "C2_FULL_HDSG"
OFFLINE_CONDITIONS = (CONDITION_C0, CONDITION_C1)
ALL_CONDITIONS = (CONDITION_C0, CONDITION_C1, CONDITION_C2)

SECTOR_NAMES = ("left", "centre", "right")


class CauseOutcome(str, Enum):
    """How a baseline's stated reason relates to the measured facts.

    `BINDING` and `UNRELATED_TRUE` are deliberately separate. A reason can be perfectly true and
    still misleading: reporting a wall at 1.2 m when the system stopped for a chair at 0.5 m is
    entailed by the Fact Packet and still sends the user into the chair.
    `HDSG_EXPLANATION_BINDING_POLICY.md` section 7 requires the two to be scored apart.
    """

    BINDING = "BINDING"                  # names the fact that caused the decision
    UNRELATED_TRUE = "UNRELATED_TRUE"    # names a real fact that did not cause the decision
    UNSUPPORTED = "UNSUPPORTED"          # names something the Fact Packet does not contain
    SILENT = "SILENT"                    # offers no reason at all
    UNSCOREABLE = "UNSCOREABLE"          # wording the frozen scorer cannot resolve


@dataclass
class CauseScore:
    outcome: CauseOutcome
    detail: str
    named_fact_id: Optional[str] = None


# The baseline is asked for a structured reply so the explanation measures can be scored without
# a human coder, and for a free-text advisory so the condition still shows what an unconstrained
# model would actually say to a user. Both come from one call, so they describe the same moment.
_RESPONSE_INSTRUCTION = """Reply with one JSON object and nothing else, in this exact shape:

{"sector_distance_m": {"left": <number or null>, "centre": <number or null>, "right": <number or null>},
 "recommended_action": "PROCEED" | "SLOW" | "REDIRECT" | "STOP",
 "recommended_sector": "LEFT" | "CENTRE" | "RIGHT" | "NONE",
 "reasons": [{"role": "ACTION" | "SCENE" | "ALTERNATIVE",
              "kind": "SECTOR" | "OBJECT",
              "sector": "LEFT" | "CENTRE" | "RIGHT" | null,
              "object_label": <short lower-case label or null>}],
 "advisory_text": "<one sentence you would say to the walker user>"}

sector_distance_m is your estimate in metres of the nearest obstruction in each third of the view.

Give one reason with role ACTION: the single thing that most justifies your recommended action.
Add a reason with role SCENE only if something elsewhere in the view also needs mentioning to make
your recommendation make sense. Add a reason with role ALTERNATIVE only if you are recommending a
change of direction, naming where you are sending the user instead. Use an empty list if you have
no reason to give."""

_C0_PREAMBLE = """You are assisting a person walking with a wheeled walking frame. You can see
only the camera image. Estimate what you can and advise them."""

_C1_PREAMBLE = """You are assisting a person walking with a wheeled walking frame. You can see the
camera image, and you are also given measurements taken by a depth camera on the walker. Use them
as you see fit."""


def _format_measured_facts(fact_packet: Mapping[str, Any]) -> str:
    """Renders the measured facts supplied to C1 and withheld from C0."""
    lines = ["Measured facts from the depth camera:"]
    for name in SECTOR_NAMES:
        sector = fact_packet["sectors"][name]
        clearance = sector.get("clearance_m")
        value = "no reliable reading" if clearance is None else f"{clearance:.2f} m"
        lines.append(f"- {name} third: {sector.get('status')}, nearest obstruction {value}")
    objects = fact_packet.get("objects") or []
    if objects:
        lines.append("Detected objects:")
        for item in objects:
            label = item.get("canonical_label") or item.get("raw_label") or "object"
            distance = item.get("distance_m")
            value = "distance unmeasured" if distance is None else f"{distance:.2f} m"
            bearing = str(item.get("bearing") or "CENTRE").lower()
            lines.append(f"- {label} in the {bearing} third at {value}")
    else:
        lines.append("Detected objects: none")
    return "\n".join(lines)


def build_baseline_prompt(condition: str, fact_packet: Mapping[str, Any]) -> str:
    """Builds the offline prompt for a comparison condition.

    The only difference between C0 and C1 is whether the measured facts are supplied. Task,
    response shape and wording are otherwise identical, so a difference in their results is
    attributable to grounding rather than to prompt design.
    """
    if condition not in OFFLINE_CONDITIONS:
        raise ValueError(f"{condition} is not an offline comparison condition")
    intent = fact_packet["interaction"]["intent"]
    intent_line = (
        "The user has not expressed a direction."
        if intent == "NONE"
        else f"The user intends to move {intent.lower()}."
    )
    parts = [_C0_PREAMBLE if condition == CONDITION_C0 else _C1_PREAMBLE, intent_line]
    if condition == CONDITION_C1:
        parts.append(_format_measured_facts(fact_packet))
    parts.append(_RESPONSE_INSTRUCTION)
    return "\n\n".join(parts)


def parse_baseline_response(raw: str) -> tuple[Optional[dict], Optional[str]]:
    """Parses a baseline reply. Returns (response, failure reason).

    Unlike the release path's `parse_candidate`, this tolerates prose around the JSON object.
    The baselines are deliberately unconstrained, so refusing to read a reply that happens to be
    wrapped in commentary would report a formatting artefact as a content failure. The release
    path's strictness is a safety property; here it would be a measurement error.
    """
    if not isinstance(raw, str) or not raw.strip():
        return None, "empty response"
    try:
        return json.loads(raw), None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None, "no JSON object found"
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None, "JSON object did not parse"
    return (value, None) if isinstance(value, dict) else (None, "response was not an object")


def _binding_fact_ids(fact_packet: Mapping[str, Any]) -> list[str]:
    return list(fact_packet["deterministic"]["action_binding"]["accepted_fact_ids"])


def _sector_fact_id(sector: Any) -> Optional[str]:
    if not isinstance(sector, str):
        return None
    name = sector.strip().lower()
    return f"sector:{name}" if name in SECTOR_NAMES else None


def _match_object(fact_packet: Mapping[str, Any], label: Any, sector: Any) -> Optional[str]:
    """Finds a detected object matching a named label, preferring one in the named third."""
    if not isinstance(label, str) or not label.strip():
        return None
    wanted = label.strip().lower()
    bearing = str(sector).strip().upper() if isinstance(sector, str) else None
    candidates = []
    for item in fact_packet.get("objects") or []:
        name = str(item.get("canonical_label") or item.get("raw_label") or "").lower()
        if not name:
            continue
        if wanted == name or wanted in name or name in wanted:
            candidates.append(item)
    if not candidates:
        return None
    if bearing:
        for item in candidates:
            if str(item.get("bearing") or "").upper() == bearing:
                return str(item["fact_id"])
    return str(candidates[0]["fact_id"])


def _resolve_reason(reason: Mapping[str, Any],
                    fact_packet: Mapping[str, Any]) -> tuple[Optional[str], Optional[str]]:
    """Resolves one stated reason to a Fact Packet identifier.

    Returns (fact_id, failure). A fact_id of None with no failure cannot occur; exactly one of
    the two is set.
    """
    kind = str(reason.get("kind") or "").upper()
    if kind == "SECTOR":
        fact_id = _sector_fact_id(reason.get("sector"))
        if fact_id is None:
            return None, "sector not recognisable"
        return fact_id, None
    if kind == "OBJECT":
        fact_id = _match_object(fact_packet, reason.get("object_label"), reason.get("sector"))
        if fact_id is None:
            return None, "named an object the Fact Packet does not contain"
        return fact_id, None
    return None, f"unrecognised reason kind {kind!r}"


def _reasons_with_role(response: Mapping[str, Any], role: str) -> list[Mapping[str, Any]]:
    reasons = response.get("reasons")
    if not isinstance(reasons, list):
        return []
    return [
        item for item in reasons
        if isinstance(item, Mapping) and str(item.get("role") or "").upper() == role
    ]


def score_cause(response: Mapping[str, Any], fact_packet: Mapping[str, Any]) -> CauseScore:
    """Scores whether the stated action reason names the measured cause of the decision.

    This is the primary measure: it is the one that tests grounded navigational transparency,
    the half of the aim that Chapter 1 says can only be established by measurement.
    """
    reasons = response.get("reasons")
    if not isinstance(reasons, list):
        return CauseScore(CauseOutcome.UNSCOREABLE, "no reasons field")
    if not reasons:
        return CauseScore(CauseOutcome.SILENT, "no reason offered")

    action_reasons = _reasons_with_role(response, "ACTION")
    if not action_reasons:
        return CauseScore(CauseOutcome.SILENT, "no reason given for the recommended action")

    binding = _binding_fact_ids(fact_packet)
    fact_id, failure = _resolve_reason(action_reasons[0], fact_packet)
    if fact_id is None:
        kind = str(action_reasons[0].get("kind") or "").upper()
        # A named object the packet does not contain is a claim about the world that the
        # measurements do not support. An unreadable field is a scoring failure, not a claim,
        # and section 5.6.1 requires those to be kept apart.
        outcome = CauseOutcome.UNSUPPORTED if kind == "OBJECT" else CauseOutcome.UNSCOREABLE
        return CauseScore(outcome, failure or "unresolvable")
    if fact_id in binding:
        return CauseScore(CauseOutcome.BINDING, "named the binding fact", fact_id)
    return CauseScore(
        CauseOutcome.UNRELATED_TRUE, "named a measured fact that did not cause the decision",
        fact_id,
    )


def score_scene_reason(response: Mapping[str, Any],
                       fact_packet: Mapping[str, Any]) -> Optional[CauseScore]:
    """Scores the conditional scene reason, or returns None when none was required.

    `HDSG_EXPLANATION_BINDING_POLICY.md` section 4 requires the scene fact only when omitting it
    would make the advisory and the decision appear inconsistent. Events that do not require one
    are excluded from the denominator rather than counted as passes.
    """
    if not fact_packet["deterministic"].get("scene_fact_required"):
        return None
    scene_reasons = _reasons_with_role(response, "SCENE")
    if not scene_reasons:
        return CauseScore(CauseOutcome.SILENT, "a scene reason was required and none was given")
    expected = list(fact_packet["deterministic"]["scene_binding"]["accepted_fact_ids"])
    fact_id, failure = _resolve_reason(scene_reasons[0], fact_packet)
    if fact_id is None:
        kind = str(scene_reasons[0].get("kind") or "").upper()
        outcome = CauseOutcome.UNSUPPORTED if kind == "OBJECT" else CauseOutcome.UNSCOREABLE
        return CauseScore(outcome, failure or "unresolvable")
    if fact_id in expected:
        return CauseScore(CauseOutcome.BINDING, "named the scene binding fact", fact_id)
    return CauseScore(
        CauseOutcome.UNRELATED_TRUE, "named a measured fact that is not the scene cause", fact_id
    )


def score_redirect_completeness(response: Mapping[str, Any],
                                fact_packet: Mapping[str, Any]) -> Optional[bool]:
    """Scores whether a redirect names both the unavailable intended sector and the alternative.

    Returns None when the event is not a redirect, so non-redirects stay out of the denominator.
    A redirect that explains only half of itself tells the user to change direction without
    saying why the original direction is unavailable, or without saying where they are being
    sent; the binding policy requires both.
    """
    deterministic = fact_packet["deterministic"]
    if deterministic["motion_decision"] != "REDIRECT":
        return None
    required = set(deterministic["action_binding"]["accepted_fact_ids"])
    named: set[str] = set()
    for role in ("ACTION", "SCENE", "ALTERNATIVE"):
        for reason in _reasons_with_role(response, role):
            fact_id, _ = _resolve_reason(reason, fact_packet)
            if fact_id is not None:
                named.add(fact_id)
    return required.issubset(named)


def has_unsupported_reason(response: Mapping[str, Any],
                           fact_packet: Mapping[str, Any]) -> bool:
    """Reports whether any stated reason names something the Fact Packet does not contain."""
    reasons = response.get("reasons")
    if not isinstance(reasons, list):
        return False
    for reason in reasons:
        if not isinstance(reason, Mapping):
            continue
        if str(reason.get("kind") or "").upper() != "OBJECT":
            continue
        fact_id, _ = _resolve_reason(reason, fact_packet)
        if fact_id is None:
            return True
    return False


def score_action_agreement(response: Mapping[str, Any],
                           fact_packet: Mapping[str, Any]) -> Optional[bool]:
    """Compares the recommended action and sector with the deterministic oracle.

    Returns None when the reply does not state an action the oracle vocabulary recognises, which
    is recorded as unscoreable rather than as disagreement.
    """
    action = response.get("recommended_action")
    sector = response.get("recommended_sector")
    if action not in hdsg.RESTRICTION_ORDER:
        return None
    if sector not in ("LEFT", "CENTRE", "RIGHT", "NONE"):
        return None
    deterministic = fact_packet["deterministic"]
    return bool(
        action == deterministic["motion_decision"]
        and sector == deterministic["selected_sector"]
    )


def score_distance_estimates(response: Mapping[str, Any],
                             fact_packet: Mapping[str, Any]) -> dict:
    """Compares estimated per-sector distances with the measured clearances.

    Reported per sector and in aggregate. `within_factor_two` is included because it is the
    comparison Chapter 2 cites from the spatial-reasoning literature, which lets the run be read
    directly against that reported figure.
    """
    estimates = response.get("sector_distance_m")
    per_sector: dict[str, dict] = {}
    errors: list[float] = []
    within_two = 0
    compared = 0

    for name in SECTOR_NAMES:
        measured = fact_packet["sectors"][name].get("clearance_m")
        estimated = None
        if isinstance(estimates, Mapping):
            raw = estimates.get(name)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool):
                estimated = float(raw)
        entry: dict[str, Any] = {"measured_m": measured, "estimated_m": estimated}
        if measured is not None and estimated is not None and estimated > 0:
            error = abs(estimated - measured)
            ratio = max(estimated / measured, measured / estimated)
            entry.update({"absolute_error_m": round(error, 3),
                          "ratio": round(ratio, 3),
                          "within_factor_two": ratio <= 2.0})
            errors.append(error)
            compared += 1
            within_two += 1 if ratio <= 2.0 else 0
        else:
            entry["comparable"] = False
        per_sector[name] = entry

    return {
        "per_sector": per_sector,
        "compared": compared,
        "mean_absolute_error_m": round(sum(errors) / len(errors), 3) if errors else None,
        "within_factor_two": within_two,
        "within_factor_two_rate": round(within_two / compared, 3) if compared else None,
    }


def derive_action_from_estimates(response: Mapping[str, Any],
                                 fact_packet: Mapping[str, Any],
                                 **authority_kwargs: Any) -> Optional[dict]:
    """Runs the deterministic policy over the model's estimated distances.

    This answers a narrower question than `score_action_agreement`: not what the model advised,
    but what the walker would have decided had it trusted the model's numbers in place of the
    depth camera. It holds the policy constant so the only thing varying is measurement accuracy.

    It is not a substitute for the advisory comparison. A model can estimate a distance badly and
    still give sound advice, since avoiding a chair does not require knowing how far away it is.
    """
    estimates = response.get("sector_distance_m")
    if not isinstance(estimates, Mapping):
        return None
    thresholds = fact_packet["configuration"]["thresholds_m"]
    clear_at = float(thresholds["sector_clear_at_or_above"])
    blocked_below = float(thresholds["sector_blocked_below"])

    lane_depths = []
    lane_status = []
    for name in SECTOR_NAMES:
        raw = estimates.get(name)
        value = float(raw) if isinstance(raw, (int, float)) and not isinstance(raw, bool) else None
        if value is None or value <= 0:
            lane_depths.append(None)
            lane_status.append("UNKNOWN")
            continue
        lane_depths.append(value)
        if value < blocked_below:
            lane_status.append("BLOCKED")
        elif value < clear_at:
            lane_status.append("CONSTRAINED")
        else:
            lane_status.append("CLEAR")

    sectors = hdsg.sectors_from_lane_state({"depths": lane_depths, "status": lane_status})
    return hdsg.determine_authority(
        fact_packet["interaction"]["intent"],
        fact_packet["deterministic"]["object_advisory"],
        sectors,
        **authority_kwargs,
    )


def score_baseline_event(condition: str, response: Optional[Mapping[str, Any]],
                         fact_packet: Mapping[str, Any],
                         parse_failure: Optional[str] = None) -> dict:
    """Scores one offline baseline event across every Table 5-9 measure."""
    if response is None:
        return {
            "condition": condition,
            "event_id": fact_packet["identity"]["event_id"],
            "scoreable": False,
            "parse_failure": parse_failure or "no response",
            "cause_outcome": CauseOutcome.UNSCOREABLE.value,
            "scene_reason_required": bool(
                fact_packet["deterministic"].get("scene_fact_required")
            ),
            "scene_reason_outcome": (
                CauseOutcome.UNSCOREABLE.value
                if fact_packet["deterministic"].get("scene_fact_required") else None
            ),
            "redirect_complete": None,
            "unsupported_reason_present": None,
            "action_agrees": None,
            "distance": None,
            "derived_action": None,
        }

    cause = score_cause(response, fact_packet)
    scene = score_scene_reason(response, fact_packet)
    derived = derive_action_from_estimates(response, fact_packet)
    deterministic = fact_packet["deterministic"]
    return {
        "condition": condition,
        "event_id": fact_packet["identity"]["event_id"],
        "scoreable": True,
        "parse_failure": None,
        "cause_outcome": cause.outcome.value,
        "cause_detail": cause.detail,
        "cause_named_fact_id": cause.named_fact_id,
        "scene_reason_required": scene is not None,
        "scene_reason_outcome": None if scene is None else scene.outcome.value,
        "redirect_complete": score_redirect_completeness(response, fact_packet),
        "unsupported_reason_present": has_unsupported_reason(response, fact_packet),
        "action_agrees": score_action_agreement(response, fact_packet),
        "distance": score_distance_estimates(response, fact_packet),
        "derived_action": None if derived is None else {
            "motion_decision": derived["motion_decision"],
            "selected_sector": derived["selected_sector"],
            "agrees_with_measured": (
                derived["motion_decision"] == deterministic["motion_decision"]
                and derived["selected_sector"] == deterministic["selected_sector"]
            ),
        },
        "advisory_text": response.get("advisory_text"),
        "schema_version": BASELINE_RESPONSE_SCHEMA,
    }


def _rate(numerator: int, denominator: int) -> Optional[float]:
    return round(numerator / denominator, 4) if denominator else None


def summarise_condition(scored_events: list[Mapping[str, Any]]) -> dict:
    """Aggregates scored events into the rows of Chapter 5 Table 5-9 and Table 5-11.

    Every rate carries its own denominator. They differ deliberately: the conditional scene
    reason is scored only over events that required one, and redirect completeness only over
    redirects, so a single event count would misstate all but the first two rows.

    Unscoreable events are excluded from a measure's numerator and denominator and counted
    separately, which is what section 5.6.1 requires: they are neither passes nor failures, and
    the denominator they were removed from has to be reportable.
    """
    total = len(scored_events)
    scoreable = [item for item in scored_events if item.get("scoreable")]

    action_scored = [item for item in scoreable if item.get("action_agrees") is not None]
    cause_scored = [
        item for item in scoreable
        if item.get("cause_outcome") != CauseOutcome.UNSCOREABLE.value
    ]
    scene_required = [item for item in scored_events if item.get("scene_reason_required")]
    scene_scored = [
        item for item in scene_required
        if item.get("scene_reason_outcome") != CauseOutcome.UNSCOREABLE.value
    ]
    redirects = [item for item in scoreable if item.get("redirect_complete") is not None]
    unsupported_scored = [
        item for item in scoreable if item.get("unsupported_reason_present") is not None
    ]

    binding = sum(1 for item in cause_scored
                  if item["cause_outcome"] == CauseOutcome.BINDING.value)
    unrelated = sum(1 for item in cause_scored
                    if item["cause_outcome"] == CauseOutcome.UNRELATED_TRUE.value)
    silent = sum(1 for item in cause_scored
                 if item["cause_outcome"] == CauseOutcome.SILENT.value)
    unsupported_cause = sum(1 for item in cause_scored
                            if item["cause_outcome"] == CauseOutcome.UNSUPPORTED.value)

    distances = [item["distance"] for item in scoreable if item.get("distance")]
    compared = sum(entry["compared"] for entry in distances)
    within_two = sum(entry["within_factor_two"] for entry in distances)
    errors = [
        sector["absolute_error_m"]
        for entry in distances for sector in entry["per_sector"].values()
        if "absolute_error_m" in sector
    ]
    derived = [item["derived_action"] for item in scoreable if item.get("derived_action")]
    would_change = sum(1 for entry in derived if not entry["agrees_with_measured"])

    return {
        "events": total,
        "scoreable_events": len(scoreable),
        "parse_failures": total - len(scoreable),
        # Table 5-9
        "guidance_agreement": {
            "n": sum(1 for item in action_scored if item["action_agrees"]),
            "d": len(action_scored),
            "rate": _rate(sum(1 for item in action_scored if item["action_agrees"]),
                          len(action_scored)),
            "unscoreable": len(scoreable) - len(action_scored),
        },
        "binding_causal_reason": {
            "n": binding, "d": len(cause_scored), "rate": _rate(binding, len(cause_scored)),
            "unscoreable": len(scoreable) - len(cause_scored),
        },
        "conditional_scene_reason": {
            "n": sum(1 for item in scene_scored
                     if item["scene_reason_outcome"] == CauseOutcome.BINDING.value),
            "d": len(scene_scored),
            "rate": _rate(sum(1 for item in scene_scored
                              if item["scene_reason_outcome"] == CauseOutcome.BINDING.value),
                          len(scene_scored)),
            "unscoreable": len(scene_required) - len(scene_scored),
        },
        "redirect_completeness": {
            "n": sum(1 for item in redirects if item["redirect_complete"]),
            "d": len(redirects),
            "rate": _rate(sum(1 for item in redirects if item["redirect_complete"]),
                          len(redirects)),
        },
        "unsupported_reason": {
            "n": sum(1 for item in unsupported_scored if item["unsupported_reason_present"]),
            "d": len(unsupported_scored),
            "rate": _rate(sum(1 for item in unsupported_scored
                              if item["unsupported_reason_present"]),
                          len(unsupported_scored)),
        },
        "silent_reason": {
            "n": silent, "d": len(cause_scored), "rate": _rate(silent, len(cause_scored)),
        },
        # Reported separately because a true but non-causal reason is neither grounded nor false,
        # and collapsing it into either would hide the failure mode it names.
        "unrelated_true_reason": {
            "n": unrelated, "d": len(cause_scored), "rate": _rate(unrelated, len(cause_scored)),
        },
        "unsupported_action_cause": {
            "n": unsupported_cause, "d": len(cause_scored),
            "rate": _rate(unsupported_cause, len(cause_scored)),
        },
        # Table 5-11
        "distance": {
            "compared": compared,
            "within_factor_two": within_two,
            "within_factor_two_rate": _rate(within_two, compared),
            "median_absolute_error_m": (
                round(sorted(errors)[len(errors) // 2], 3) if errors else None
            ),
            "decision_would_change_if_trusted": {
                "n": would_change, "d": len(derived), "rate": _rate(would_change, len(derived)),
            },
        },
    }
