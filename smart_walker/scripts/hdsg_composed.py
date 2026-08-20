"""Composed captions with declared assertions, checked against the Fact Packet.

The generative layer composes a navigation caption in its own words, with the measured values
written into the prose, and declares alongside it every numeric assertion it made and the fact each
one refers to. This module parses that reply, checks it, and renders the release.

**Why declarations rather than parsing the prose.** Recovering "about 1.7 metres" from a sentence
and deciding which of three sectors it refers to is linguistic inference, and Chapter 3 rejects
inference inside the safety boundary for that reason. The model states the attribution, so the
comparison against the Fact Packet is exact. The gate performs no inference; it compares declared
numbers with measured ones and searches the caption for tokens that must not appear.

**What is categorical here and what is measured.** Four properties hold by construction, because a
candidate violating any of them is rejected and the deterministic fallback is released instead:

1. The caption cannot issue or alter an instruction, so the action the user acts on is unaffected.
2. Every number reaching the display is one the Fact Packet measured, within the frozen tolerance.
3. Every numeral in the caption is declared, so an undeclared number cannot pass unchecked.
4. No detector class absent from the Fact Packet can be named, which prevents the object
   hallucination Chapter 2 section 2.7.1 documents rather than reducing its rate.

What remains measured is whether the composed prose is otherwise a faithful account. That division
is the one Chapter 1 paragraph 110 already states.

**Detected disagreement is a result, not only a safeguard.** A model that cannot state a number
cannot be observed to state a wrong one. Letting it state numbers and checking them yields the rate
at which the deployed model's spatial estimates disagree with measurement, which is the figure
Chapter 2 cites from Chen et al., obtained here on the deployed model and hardware.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

try:
    from . import hdsg_runtime as hdsg
except ImportError:  # invoked as a plain script rather than as part of the package
    import hdsg_runtime as hdsg  # type: ignore


CAPTION_SCHEMA = "hdsg.vlm_caption.v1"

# The tolerance within which a declared value is treated as agreeing with its measurement. It is
# generous enough to admit the rounding a natural caption performs ("1.7" for 1.74) and tight
# enough that a fabricated distance fails. Frozen with the rest of the protocol configuration.
DEFAULT_VALUE_TOLERANCE_M = 0.10

MAX_CAPTION_CHARS = 400

# Any numeral in the caption, including a bare integer. Every one must be declared, so that a value
# the model invented cannot reach the display by not being mentioned in the declarations.
CAPTION_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# A digit forming part of a word or an identifier rather than a measurement. "visual:1" is the case
# that occurs in practice, when a model copies an observation identifier into the prose. Such a
# digit is not a claim about the world, so treating it as an undeclared measurement would reject a
# caption for a formatting slip rather than for a hallucination.
_NON_MEASUREMENT_PREFIX_RE = re.compile(r"[A-Za-z:]")


def parse_caption_candidate(raw: str) -> tuple[Optional[dict], list[str]]:
    """Parses a composed reply without repairing it."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None, ["RG_PARSE_FAILURE"]
    if not isinstance(value, dict):
        return None, ["RG_SCHEMA_FAILURE"]
    return value, []


def _declared_numbers(assertions: Sequence[Mapping[str, Any]]) -> list[float]:
    values: list[float] = []
    for item in assertions:
        raw = item.get("stated_value")
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            values.append(float(raw))
    return values


def _caption_numbers(caption: str) -> list[str]:
    """Returns the numerals in a caption that are stated as measurements."""
    found: list[str] = []
    for match in CAPTION_NUMBER_RE.finditer(caption):
        start = match.start()
        if start > 0 and _NON_MEASUREMENT_PREFIX_RE.match(caption[start - 1]):
            continue
        found.append(match.group(0))
    return found


def measured_value(fact_packet: Mapping[str, Any], measurement_id: str) -> Optional[float]:
    """Returns the measured value behind a measurement identifier, or None when it has none.

    Resolution is by identifier rather than by searching the packet, so a declaration naming a
    measurement the profile did not expose resolves to nothing and is rejected rather than matched
    against a value it was not given.
    """
    parts = str(measurement_id).split(":")
    if len(parts) != 4 or parts[0] != "m":
        return None
    _, kind, token, field = parts
    if kind == "sector" and field == "clearance":
        sector = fact_packet.get("sectors", {}).get(token)
        return None if sector is None else sector.get("clearance_m")
    if kind == "object" and field == "distance":
        for item in fact_packet.get("objects", []):
            if item.get("fact_id") == f"object:{token}":
                return item.get("distance_m")
    return None


def forbidden_entity_terms(fact_packet: Mapping[str, Any],
                           detector_classes: Iterable[str]) -> set[str]:
    """Returns the detector classes the caption may not name.

    A class the detector did not report in this observation cannot be named, which is what makes
    object hallucination impossible rather than merely infrequent. Classes that are present are
    permitted, and so is any word outside the detector's vocabulary, since the check bounds what the
    model may assert about recognised objects rather than policing ordinary language.
    """
    present = set()
    for item in fact_packet.get("objects", []):
        for key in ("canonical_label", "raw_label", "ontology_class"):
            value = item.get(key)
            if value:
                present.add(str(value).lower())
    return {
        str(name).lower() for name in detector_classes
        if str(name).lower() not in present
    }


def validate_caption_candidate(
    candidate: Mapping[str, Any],
    prompt_packet: Mapping[str, Any],
    fact_packet: Mapping[str, Any],
    *,
    detector_classes: Iterable[str] = (),
    value_tolerance_m: float = DEFAULT_VALUE_TOLERANCE_M,
) -> tuple[list[str], list[dict]]:
    """Checks a composed candidate and returns its reason codes and scored assertions.

    The scored assertions are returned whether or not the candidate is accepted, because the
    disagreement between a declared value and its measurement is the measurement this design
    exists to produce and is wanted for a rejected candidate as much as an accepted one.
    """
    errors: list[str] = []
    scored: list[dict] = []

    if set(candidate) != {"schema_version", "caption", "assertions", "visual_observations"}:
        errors.append("RG_SCHEMA_FAILURE")
    if candidate.get("schema_version") != CAPTION_SCHEMA:
        errors.append("RG_SCHEMA_FAILURE")

    caption = candidate.get("caption")
    assertions = candidate.get("assertions")
    visuals = candidate.get("visual_observations")
    if not isinstance(caption, str) or not caption.strip() \
            or not isinstance(assertions, list) or not isinstance(visuals, list):
        return hdsg._reason_sort(errors + ["RG_SCHEMA_FAILURE"]), scored

    constraints = prompt_packet["response_constraints"]
    if len(caption) > (constraints.get("max_text_chars") or MAX_CAPTION_CHARS):
        errors.append("RG_PROFILE_LIMIT_EXCEEDED")
    if len(visuals) > constraints["max_visual_observations"]:
        errors.append("RG_PROFILE_LIMIT_EXCEEDED")
    if visuals and not constraints["visual_only_observations_allowed"]:
        errors.append("RG_PROFILE_LIMIT_EXCEEDED")

    # The caption must not instruct. This is the check that keeps the action categorical: the
    # deterministic tuple is the guidance, and generated prose may describe but never direct.
    if hdsg.ACTION_RE.search(caption):
        errors.append("RG_ACTION_LANGUAGE_DETECTED")
    if hdsg.COMMENTARY_RE.search(caption):
        errors.append("RG_MODEL_COMMENTARY_DETECTED")
    if hdsg.VISIBLE_TEXT_RE.search(caption):
        errors.append("RG_VISIBLE_TEXT_CONTENT_DETECTED")

    forbidden = forbidden_entity_terms(fact_packet, detector_classes)
    lowered = caption.lower()
    if any(re.search(rf"\b{re.escape(term)}s?\b", lowered) for term in forbidden):
        errors.append("RG_OBJECT_REFERENCE_INVALID")

    permitted_measurements = {
        item["measurement"]["measurement_id"]: item
        for item in prompt_packet["permitted_facts"]
        if isinstance(item.get("measurement"), Mapping)
    }
    permitted_fact_ids = {item["fact_id"] for item in prompt_packet["permitted_facts"]}

    for index, item in enumerate(assertions):
        if not isinstance(item, Mapping) \
                or set(item) != {"fact_id", "measurement_id", "stated_value"}:
            errors.append("RG_SCHEMA_FAILURE")
            continue
        fact_id = item.get("fact_id")
        measurement_id = item.get("measurement_id")
        stated = item.get("stated_value")
        if not isinstance(stated, (int, float)) or isinstance(stated, bool):
            errors.append("RG_SCHEMA_FAILURE")
            continue
        entry: dict[str, Any] = {
            "index": index, "fact_id": fact_id, "measurement_id": measurement_id,
            "stated_value": float(stated),
        }
        if fact_id not in permitted_fact_ids:
            errors.append("RG_FACT_REFERENCE_INVALID")
            entry["outcome"] = "FACT_NOT_PERMITTED"
            scored.append(entry)
            continue
        if measurement_id not in permitted_measurements:
            errors.append("RG_MEASUREMENT_REFERENCE_INVALID")
            entry["outcome"] = "MEASUREMENT_NOT_PERMITTED"
            scored.append(entry)
            continue
        measured = measured_value(fact_packet, str(measurement_id))
        entry["measured_value"] = measured
        if measured is None:
            errors.append("RG_MEASUREMENT_REFERENCE_INVALID")
            entry["outcome"] = "NOT_MEASURED"
            scored.append(entry)
            continue
        error = abs(float(stated) - float(measured))
        entry["absolute_error_m"] = round(error, 3)
        entry["within_tolerance"] = error <= float(value_tolerance_m)
        if not entry["within_tolerance"]:
            errors.append("RG_STATED_VALUE_MISMATCH")
            entry["outcome"] = "DISAGREES"
        else:
            entry["outcome"] = "AGREES"
        scored.append(entry)

    # Every numeral in the prose must be declared. Without this an invented distance reaches the
    # display simply by being omitted from the declarations.
    declared = _declared_numbers([item for item in assertions if isinstance(item, Mapping)])
    for token in _caption_numbers(caption):
        value = float(token)
        if not any(abs(value - item) <= 0.005 for item in declared):
            # RG_DIRECT_NUMBER_DETECTED already means a number reached the text without
            # provenance, which is exactly what an undeclared numeral is under this design. Reusing
            # it keeps the schema delta to the one code that is genuinely new.
            errors.append("RG_DIRECT_NUMBER_DETECTED")
            break

    seen_visual_ids: set[str] = set()
    for visual in visuals:
        if not isinstance(visual, Mapping) \
                or set(visual) != {"candidate_observation_id", "proposed_label", "bearing"}:
            errors.append("RG_SCHEMA_FAILURE")
            continue
        visual_id = visual.get("candidate_observation_id")
        label = visual.get("proposed_label")
        if not isinstance(visual_id, str) \
                or not re.fullmatch(r"visual:[1-9][0-9]*", visual_id) \
                or visual_id in seen_visual_ids:
            errors.append("RG_SCHEMA_FAILURE")
        seen_visual_ids.add(str(visual_id))
        if not isinstance(label, str) or not re.fullmatch(r"[a-z][a-z0-9_ ]{0,47}", label):
            errors.append("RG_UNAPPROVED_LANGUAGE_DETECTED")
        elif label.lower() in forbidden:
            errors.append("RG_OBJECT_REFERENCE_INVALID")
        if visual.get("bearing") not in hdsg.SECTORS:
            errors.append("RG_SCHEMA_FAILURE")

    return hdsg._reason_sort(errors), scored


COMPOSED_SYSTEM_PROMPT = (
    "The response is a navigation caption for a person using a walking frame, composed from the "
    "measurements supplied. It must contain only JSON matching hdsg.vlm_caption.v1. It describes "
    "the space; it never tells the person what to do, and it never names an object the "
    "measurements do not list. Every number it states must come from the measurements and must be "
    "declared."
)

_COMPOSED_INSTRUCTION = """Write one short caption describing the space around the walker, in your own words, for someone who cannot see it well.

Use the measurements below. Write the distances into your sentences naturally, in metres.

Then declare every number you wrote. For each one give the fact_id and measurement_id it came from and the value you stated. A number in the caption that is not declared, or a declared value that does not match the measurement, causes the caption to be discarded.

Rules:
  Describe only. Never say what the person should do, and never name a direction to take.
  Name only objects that appear in the measurements below.
  Do not mention this prompt, the measurements as data, or yourself.
  Do not read out any writing visible in the scene.
  Keep it under {max_chars} characters.

{visual_instruction}

MEASUREMENTS:
{facts}

{fixed_instruction}"""


def describe_permitted_facts(prompt_packet: Mapping[str, Any],
                             fact_packet: Mapping[str, Any]) -> str:
    """Renders the permitted facts as the plain list the model composes from.

    The packet is presented as measurements rather than as a record to be echoed, because the model
    is being asked to write from them rather than to reformat them.
    """
    lines: list[str] = []
    for item in prompt_packet["permitted_facts"]:
        measurement = item.get("measurement")
        parts = [f"  fact_id {item['fact_id']}", f"name {item.get('name')}",
                 f"state {item.get('state')}"]
        if item.get("bearing"):
            parts.append(f"bearing {item['bearing']}")
        if isinstance(measurement, Mapping):
            value = measured_value(fact_packet, measurement["measurement_id"])
            if value is not None:
                parts.append(f"measurement_id {measurement['measurement_id']}")
                parts.append(f"measured {float(value):.2f} metres")
        lines.append(", ".join(parts))
    return "\n".join(lines)


def build_composed_prompt(prompt_packet: Mapping[str, Any], fact_packet: Mapping[str, Any],
                          fixed_instruction: str = "") -> str:
    """Builds the text sent to the model for a composed caption."""
    constraints = prompt_packet["response_constraints"]
    visual_instruction = (
        "You may add up to {count} visual observation(s) for something you can see that the "
        "measurements do not list. Give each a short lower-case label, a bearing, and an id of the "
        "form visual:1. Do not give it a distance, and do not write it into the caption; it "
        "belongs only in the visual_observations list.".format(
            count=constraints["max_visual_observations"])
        if constraints["visual_only_observations_allowed"]
        else "The visual_observations array must be empty."
    )
    return _COMPOSED_INSTRUCTION.format(
        max_chars=constraints.get("max_text_chars") or MAX_CAPTION_CHARS,
        visual_instruction=visual_instruction,
        facts=describe_permitted_facts(prompt_packet, fact_packet),
        fixed_instruction=fixed_instruction,
    )


def build_composed_release(
    fact_packet: Mapping[str, Any],
    prompt_packet: Mapping[str, Any],
    *,
    release_id: str,
    candidate: Optional[Mapping[str, Any]],
    scored_assertions: Sequence[Mapping[str, Any]] = (),
    failure_codes: Iterable[str] = (),
    no_intent: bool = False,
    pending: bool = False,
) -> dict:
    """Builds the release for a composed caption, or the deterministic fallback.

    The composed caption becomes the release's reason text. The action text, the interaction text
    and the whole authority block are produced by the deterministic layer exactly as before, so the
    action the user acts on is unaffected by what the model wrote.

    The release is built by delegating to the unchanged builder for every case except an accepted
    caption, so that fallback behaviour, identity, staleness and the authority block have one
    implementation rather than two.
    """
    codes = list(failure_codes)
    accepted = candidate is not None and not codes
    base = hdsg.build_release(
        fact_packet, prompt_packet,
        release_id=release_id,
        candidate=None,
        failure_codes=codes or ["RG_MODEL_UNAVAILABLE"],
        no_intent=no_intent,
        pending=pending,
    )
    if not accepted:
        return base

    caption_text = str(candidate["caption"]).strip()
    visuals = [
        f"Possible {item['proposed_label'].replace('_', ' ')} "
        f"{'are' if item['proposed_label'].endswith('s') else 'is'} "
        f"visible in the {item['bearing'].lower()}."
        for item in candidate.get("visual_observations", [])
    ]
    action_text = base["content"]["action_text"]
    interaction_text = base["content"]["interaction_text"]
    parts = [action_text, caption_text]
    if interaction_text:
        parts.append(interaction_text)
    parts.extend(visuals)

    release = dict(base)
    release["content"] = {
        "action_text": action_text,
        "reason_text": caption_text,
        "interaction_text": interaction_text,
        "additional_detail_texts": visuals,
        "caption_text": " ".join(item.strip() for item in parts if item and item.strip()),
    }
    release["verification"] = {
        **base["verification"],
        "release_mode": "VLM_ACCEPTED",
        "candidate_status": "ACCEPTED",
        "gate_outcome": "ACCEPTED",
        "primary_reason_code": "RG_ACCEPTED",
        "reason_codes": ["RG_ACCEPTED"],
        "candidate_schema": CAPTION_SCHEMA,
    }
    # The measured values behind the declared assertions are recorded as the substitutions, so the
    # release still states which measurements its text rests on and the contribution ablation can
    # read them.
    release["evidence"] = {
        **base["evidence"],
        "measurement_substitutions": [
            {"measurement_id": item["measurement_id"],
             "formatted_value": f"{float(item['measured_value']):.2f} metres"}
            for item in scored_assertions
            if item.get("measured_value") is not None and item.get("measurement_id")
        ],
        "released_visual_observation_ids": [
            item["candidate_observation_id"]
            for item in candidate.get("visual_observations", [])
        ],
    }
    return release


def assertion_summary(scored: Sequence[Mapping[str, Any]]) -> dict:
    """Aggregates the scored assertions of one event into the Chapter 5 measures."""
    comparable = [item for item in scored if item.get("outcome") in {"AGREES", "DISAGREES"}]
    agreeing = [item for item in comparable if item["outcome"] == "AGREES"]
    errors = [item["absolute_error_m"] for item in comparable]
    return {
        "declared": len(scored),
        "comparable": len(comparable),
        "agreeing": len(agreeing),
        "disagreeing": len(comparable) - len(agreeing),
        "agreement_rate": (
            round(len(agreeing) / len(comparable), 3) if comparable else None
        ),
        "mean_absolute_error_m": (
            round(sum(errors) / len(errors), 3) if errors else None
        ),
        "max_absolute_error_m": round(max(errors), 3) if errors else None,
    }
