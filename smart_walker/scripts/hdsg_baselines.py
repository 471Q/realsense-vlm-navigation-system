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

from . import hdsg_runtime as hdsg


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
 "cause": {"kind": "SECTOR" | "OBJECT" | "NONE",
           "sector": "LEFT" | "CENTRE" | "RIGHT" | null,
           "object_label": <short lower-case label or null>},
 "advisory_text": "<one sentence you would say to the walker user>"}

sector_distance_m is your estimate in metres of the nearest obstruction in each third of the
view. cause names the single thing that most justifies your recommended action."""

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


def score_cause(response: Mapping[str, Any], fact_packet: Mapping[str, Any]) -> CauseScore:
    """Scores whether the stated reason names the measured cause of the decision.

    This is the primary measure: it is the one that tests grounded navigational transparency,
    the half of the aim that Chapter 1 says can only be established by measurement.
    """
    cause = response.get("cause")
    if not isinstance(cause, Mapping):
        return CauseScore(CauseOutcome.UNSCOREABLE, "no structured cause field")

    kind = str(cause.get("kind") or "").upper()
    binding = _binding_fact_ids(fact_packet)

    if kind == "NONE":
        return CauseScore(CauseOutcome.SILENT, "no reason offered")

    if kind == "SECTOR":
        fact_id = _sector_fact_id(cause.get("sector"))
        if fact_id is None:
            return CauseScore(CauseOutcome.UNSCOREABLE, "sector not recognisable")
        if fact_id in binding:
            return CauseScore(CauseOutcome.BINDING, "named the binding sector", fact_id)
        return CauseScore(
            CauseOutcome.UNRELATED_TRUE, "named a measured sector that did not cause the decision",
            fact_id,
        )

    if kind == "OBJECT":
        fact_id = _match_object(fact_packet, cause.get("object_label"), cause.get("sector"))
        if fact_id is None:
            return CauseScore(
                CauseOutcome.UNSUPPORTED, "named an object the Fact Packet does not contain"
            )
        if fact_id in binding:
            return CauseScore(CauseOutcome.BINDING, "named the binding object", fact_id)
        return CauseScore(
            CauseOutcome.UNRELATED_TRUE, "named a detected object that did not cause the decision",
            fact_id,
        )

    return CauseScore(CauseOutcome.UNSCOREABLE, f"unrecognised cause kind {kind!r}")


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
            "action_agrees": None,
            "distance": None,
            "derived_action": None,
        }

    cause = score_cause(response, fact_packet)
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
