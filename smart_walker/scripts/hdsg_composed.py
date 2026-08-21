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
3. Every number the caption states is declared, so an undeclared one cannot pass unchecked. This
   covers a distance written in words as well as one written in digits: scanning only for digits
   left "roughly five metres" unchecked against a measured 1.74 m, which is the failure mode the
   property exists to exclude.
4. No detector class absent from the Fact Packet can be named, in any of its written forms, which
   prevents the object hallucination Chapter 2 section 2.7.1 documents rather than reducing its
   rate. Plural forms are enumerated rather than guessed by suffix, since a suffix misses "people"
   and person is the class a walker is most often wrong about.

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

# Any numeral in the caption, including a bare integer and a value written without its leading zero.
# Every one must be declared, so that a value the model invented cannot reach the display by not
# being mentioned in the declarations.
CAPTION_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|\.\d+")

# A digit forming part of a word or an identifier rather than a measurement. "visual:1" is the case
# that occurs in practice, when a model copies an observation identifier into the prose. Such a
# digit is not a claim about the world, so treating it as an undeclared measurement would reject a
# caption for a formatting slip rather than for a hallucination.
_NON_MEASUREMENT_PREFIX_RE = re.compile(r"[A-Za-z:]")

# A distance can be stated in words as readily as in digits, and "roughly five metres" is a claim
# about the world in exactly the way "5.0 metres" is. Scanning only for digits left the whole class
# undeclared and unchecked, so a spelled-out distance reached the display without ever meeting a
# measurement.
_WORD_NUMBER_VALUES = {
    "zero": 0.0, "one": 1.0, "two": 2.0, "three": 3.0, "four": 4.0, "five": 5.0,
    "six": 6.0, "seven": 7.0, "eight": 8.0, "nine": 9.0, "ten": 10.0,
    "eleven": 11.0, "twelve": 12.0, "thirteen": 13.0, "fourteen": 14.0, "fifteen": 15.0,
    "sixteen": 16.0, "seventeen": 17.0, "eighteen": 18.0, "nineteen": 19.0, "twenty": 20.0,
}

_DISTANCE_UNIT = r"(?:metres?|meters?|centimetres?|centimeters?|cm|mm)"

# A word number counts as a stated distance only when a unit follows it, which is what separates
# "two metres" from "no one ahead" and "one of the chairs". A bare digit needs no such test, because
# a digit in a navigation caption is a measurement and an ordinary English word is not.
_WORD_NUMBER_RE = re.compile(
    r"\b(" + "|".join(_WORD_NUMBER_VALUES) + r")\b"
    r"(?=\s+(?:and\s+a\s+half\s+)?" + _DISTANCE_UNIT + r"\b)",
    re.IGNORECASE,
)


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


def _caption_numbers(caption: str) -> list[tuple[str, float, int]]:
    """Returns the numbers a caption states as measurements, in digits or in words.

    Each entry is the text as written, its value, and the number of decimal places it was written
    to. The precision is carried because it decides how far the written number may sit from the
    value it declares: "1.7" is a faithful rendering of 1.74 and "2" is not.
    """
    found: list[tuple[str, float, int]] = []
    for match in CAPTION_NUMBER_RE.finditer(caption):
        start = match.start()
        if start > 0 and _NON_MEASUREMENT_PREFIX_RE.match(caption[start - 1]):
            continue
        token = match.group(0)
        decimals = len(token.partition(".")[2])
        found.append((token, float(token), decimals))
    for match in _WORD_NUMBER_RE.finditer(caption):
        token = match.group(1)
        found.append((token, _WORD_NUMBER_VALUES[token.lower()], 0))
    return found


def _rounding_allowance(decimals: int, value_tolerance_m: float) -> float:
    """How far a written number may sit from the value it declares, given how it was written.

    A caption rounds, and the design intends it to: the value tolerance exists so that "1.7" passes
    against a measured 1.74. Matching the written number to the declaration within a fixed hundredth
    of a metre contradicted that, and rejected the caption the prompt asks for.

    The allowance is half a unit of the last written place, so one decimal admits 0.05 and two admit
    0.005. It is capped at the value tolerance, because a whole number written for 1.74 is not a
    rounding the reader can discount: "2 metres" overstates the clearance by more than the gate
    allows a declaration to be wrong, and the overstatement is the direction that matters.
    """
    return min(max(0.005, 0.5 * (10.0 ** -decimals)), float(value_tolerance_m))


def undeclared_numbers(caption: str, assertions: Sequence[Mapping[str, Any]],
                       value_tolerance_m: float = DEFAULT_VALUE_TOLERANCE_M) -> list[str]:
    """Returns the numbers a caption states that no declaration accounts for.

    Exposed so an analysis of an archived run can report which numbers went undeclared, rather than
    only that some did. The release records the reason code; this recovers the token behind it.
    """
    declared = _declared_numbers([item for item in assertions if isinstance(item, Mapping)])
    missing: list[str] = []
    for token, value, decimals in _caption_numbers(caption):
        allowance = _rounding_allowance(decimals, value_tolerance_m)
        if not any(abs(value - item) <= allowance for item in declared):
            missing.append(token)
    return missing


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


# Plurals that appending "s" does not produce. "person" is the one that matters: it is the class a
# walker is most often wrong about, and a caption saying "two people" when the detector reported
# nobody went unchecked while the same caption saying "a person" was rejected.
_IRREGULAR_PLURALS = {
    "person": "people",
    "knife": "knives",
    "mouse": "mice",
    "sheep": "sheep",
}

# A class ending in a sibilant takes "es", so "bus" pluralises to "buses" and not to "buss".
_SIBILANT_ENDINGS = ("s", "x", "z", "ch", "sh")


def _surface_forms(term: str) -> set[str]:
    """Returns the written forms of one detector class, singular and plural."""
    lowered = str(term).lower().strip()
    if not lowered:
        return set()
    forms = {lowered}
    irregular = _IRREGULAR_PLURALS.get(lowered)
    if irregular:
        forms.add(irregular)
    elif lowered.endswith(_SIBILANT_ENDINGS):
        forms.add(lowered + "es")
    elif lowered.endswith("y") and len(lowered) > 1 and lowered[-2] not in "aeiou":
        forms.add(lowered[:-1] + "ies")
    else:
        forms.add(lowered + "s")
    return forms


def forbidden_entity_terms(fact_packet: Mapping[str, Any],
                           detector_classes: Iterable[str]) -> set[str]:
    """Returns the written forms the caption may not use, singular and plural.

    A class the detector did not report in this observation cannot be named, which is what makes
    object hallucination impossible rather than merely infrequent. Classes that are present are
    permitted, and so is any word outside the detector's vocabulary, since the check bounds what the
    model may assert about recognised objects rather than policing ordinary language.

    Plurals are enumerated here rather than by appending "s" at the point of search, because a
    single suffix does not cover the vocabulary: it misses "people" entirely and turns "bus" into
    "buss".
    """
    present: set[str] = set()
    for item in fact_packet.get("objects", []):
        for key in ("canonical_label", "raw_label", "ontology_class"):
            value = item.get(key)
            if value:
                present.update(_surface_forms(str(value)))
    forbidden: set[str] = set()
    for name in detector_classes:
        forms = _surface_forms(str(name))
        if forms & present:
            continue
        forbidden.update(forms)
    return forbidden


def _names_forbidden_term(text: str, forbidden: Iterable[str]) -> bool:
    """True when the text names one of the forbidden forms as a whole word.

    Used for the caption and for a visual observation's label alike. The label was previously
    compared by equality, so a forbidden class sitting inside a longer label passed unnoticed while
    the same class alone was rejected.
    """
    lowered = str(text).lower()
    return any(re.search(rf"\b{re.escape(term)}\b", lowered) for term in forbidden)


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
    if _names_forbidden_term(caption, forbidden):
        errors.append("RG_OBJECT_REFERENCE_INVALID")

    permitted_measurements = {
        item["measurement"]["measurement_id"]
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

    # Every number the prose states must be declared, whether written in digits or in words.
    # Without this an invented distance reaches the display simply by being omitted from the
    # declarations. RG_DIRECT_NUMBER_DETECTED already means a number reached the text without
    # provenance, which is exactly what an undeclared number is under this design, so reusing it
    # keeps the reason-code enumeration unchanged.
    if undeclared_numbers(caption, assertions, value_tolerance_m):
        errors.append("RG_DIRECT_NUMBER_DETECTED")

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
        elif _names_forbidden_term(label.replace("_", " "), forbidden):
            errors.append("RG_OBJECT_REFERENCE_INVALID")
        elif (hdsg.NUMBER_RE.search(label) or hdsg.ACTION_RE.search(label)
              or hdsg.COMMENTARY_RE.search(label)
              or hdsg.VISUAL_LABEL_PROHIBITED_RE.search(label)):
            # The label is rendered into the release as "Possible <label> is visible in the
            # <bearing>", so it reaches the display as prose and is screened as prose. The
            # character-class pattern above admits spaces and therefore admits a phrase, which is
            # why an instruction or a distance can appear in a label that is otherwise well formed.
            errors.append("RG_UNAPPROVED_LANGUAGE_DETECTED")
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

Then declare every number you wrote. For each one give the fact_id and the measurement_id it came from, exactly as they appear below, and the value you stated. They are different: fact_id looks like sector:centre, measurement_id looks like m:sector:centre:clearance. For a clearance of 1.74 metres at the centre, the declaration is:

  {{"fact_id": "sector:centre", "measurement_id": "m:sector:centre:clearance", "stated_value": 1.74}}

A number in the caption that is not declared, or a declared value that does not match the measurement, causes the caption to be discarded. This applies to a distance written as a word as much as to one written in digits: "two metres" needs its declaration exactly as "2.0 metres" does. Round if it reads better, and declare the value you wrote rather than the one you rounded from.

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
    #
    # Deduplicated by identifier, as the deterministic builder also does. A caption may legitimately
    # state one measurement twice, and without this the release carried the same substitution twice
    # and violated the uniqueItems constraint its own frozen schema imposes.
    substitutions = {
        str(item["measurement_id"]):
            {"measurement_id": str(item["measurement_id"]),
             "formatted_value": f"{float(item['measured_value']):.2f} metres"}
        for item in scored_assertions
        if item.get("measured_value") is not None and item.get("measurement_id")
    }
    release["evidence"] = {
        **base["evidence"],
        "measurement_substitutions": list(substitutions.values()),
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
