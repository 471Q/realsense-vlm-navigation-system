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
2. Every number reaching the display is a measured value written character for character as the
   deterministic renderer would display it. There is no tolerance and no alternative spelling, so
   one sensor reading reaches the user in one form whether the gate accepted a caption or fell
   back. A measurement of 2.00 is written "2.00"; "2 metres" and "two metres" are rejected.
3. Every number the caption states is declared, so an undeclared one cannot pass unchecked. This
   covers a distance written in words as well as one written in digits: scanning only for digits
   left "roughly five metres" unchecked against a measured 1.74 m, which is the failure mode the
   property exists to exclude. An unquantified phrase such as "a few metres" is a qualitative
   assertion rather than a number, and falls under the open decision recorded in
   HDSG_VERIFIED_GENERATION_POLICY.md section 7.2.
4. No detector class absent from the Fact Packet can be named, in any of its written forms, which
   prevents the object hallucination Chapter 2 section 2.7.1 documents rather than reducing its
   rate. Plural forms are enumerated rather than guessed by suffix, since a suffix misses "people"
   and person is the class a walker is most often wrong about.

**What these properties do not cover.** Each one constrains a value or a name in isolation. None
constrains which fact a sentence attaches a value to, because the declaration states the attribution
and the prose is not read. A caption that declares all three sector clearances correctly and then
writes each one against the wrong sector passes every check: every number is measured, every number
is declared, every declaration agrees with its measurement, and the sentence is false. Detecting it
would require reading the prose to recover the attribution, which is the linguistic inference
Chapter 3 excludes from the safety boundary. The attribution of prose to fact therefore remains
measured rather than guaranteed, and `stated_in_caption` is recorded on each declaration so an
analysis can at least separate a declaration the caption used from one it did not.

What remains measured is whether the composed prose is otherwise a faithful account. That division
is the one Chapter 1 paragraph 110 already states.

**Detected disagreement is a result, not only a safeguard.** A model that cannot state a number
cannot be observed to state a wrong one. Letting it state numbers and checking them yields the rate
at which the deployed model alters a value it was handed, and the size of each alteration is
recorded whether or not the caption is released.

That rate is not the same quantity as the spatial-estimation error Chapter 2 cites from Chen et al.
Here the model is given the measurement, so a disagreement is a transcription failure. The
estimation error is what `C0_VLM_ONLY` and `C1_GROUNDED_UNGATED` measure, where the model has no
measurements to transcribe. The two must not be reported as one figure.
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

MAX_CAPTION_CHARS = 400


def _display_string(value: Any) -> str:
    """Returns the digits a caption must use for a measurement, without the unit.

    A caption states a measurement as the system displays it, character for character. There is no
    tolerance, and there is no second acceptable spelling of the same value: a measurement of 1.74
    is written "1.74", and nothing else is that measurement.

    The comparison is made on the written token rather than on the number behind it, and that is
    what makes the property hold. Parsing the token and comparing the resulting float admitted
    three ways for a caption to display a figure the system never measured. "1.7449 metres" parsed
    to a value that rounded onto 1.74 and was released with the invented precision intact. "2
    metres" and "two metres" both parsed onto a measurement the deterministic renderer displays as
    "2.00 metres", so the same sensor reading reached the user in a different form depending on
    whether the gate happened to accept a caption. An earlier design went further and allowed a
    0.10 m window as well, which composed with a second window between the caption and its
    declaration and put numbers on the display 0.15 m from the truth.

    The magnitude of a disagreement is still recorded on every declaration, so the size of an error
    remains reportable although any error rejects.
    """
    return f"{float(value):.{hdsg.MEASUREMENT_DECIMALS}f}"


# Any numeral in the caption, including a bare integer and a value written without its leading zero.
# Every one must be a measurement written as displayed, so that a value the model invented cannot
# reach the display by going unmentioned in the declarations.
CAPTION_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?|\.\d+")

# The identifiers this system uses, which carry digits that are not claims about the world. A model
# does copy one into the prose, and rejecting a caption for that would report a formatting slip as a
# hallucination. Each is masked out before the caption is scanned.
#
# The previous exclusion tested the single character preceding a digit and skipped the digit when
# that character was a letter or a colon. It therefore skipped every digit in "Clearance:9.99 metres
# ahead" and in "about9.99 metres", so an invented distance passed the undeclared-number check
# untouched and reached the display. The shapes are enumerated instead, so a colon the model wrote
# for its own reasons grants no cover.
_IDENTIFIER_RE = re.compile(
    r"\b(?:m:)?(?:visual|object|sector):[A-Za-z0-9_]+(?::[a-z_]+)?\b", re.IGNORECASE
)

_DISTANCE_UNIT = r"(?:metres?|meters?|centimetres?|centimeters?|cm|mm)"

# A distance can be stated in words as readily as in digits, and "roughly five metres" is a claim
# about the world in exactly the way "5.00 metres" is. Scanning only for digits left the whole class
# unchecked, so a spelled-out distance reached the display without ever meeting a measurement.
#
# No word is the displayed form of a measurement, so finding one of these is finding a violation.
# The vocabulary exists to detect the phrase and to name it in the record, not to value it, which is
# why the words carry no numeric values. The range runs to twenty because that spans the distances a
# walker's depth camera reports; a compound above it, "one hundred metres", is outside the range and
# outside what the sensor can produce.
_WORD_NUMBERS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen", "twenty",
)

# A word number counts as a stated distance only when a unit follows it, which is what separates
# "two metres" from "no one ahead" and "one of the chairs". A bare digit needs no such test, because
# a digit in a navigation caption is a measurement and an ordinary English word is not. A trailing
# half is taken into the phrase so that the token reported is the whole of what was written.
_WORD_NUMBER_RE = re.compile(
    r"\b(?:" + "|".join(_WORD_NUMBERS) + r")\b(?:\s+and\s+a\s+half)?"
    r"(?=\s+" + _DISTANCE_UNIT + r"\b)",
    re.IGNORECASE,
)

# "half a metre" states a distance without any number word before it.
_BARE_HALF_RE = re.compile(r"\bhalf\s+(?:a|an)\s+(?=" + _DISTANCE_UNIT + r"\b)", re.IGNORECASE)


def parse_caption_candidate(raw: str) -> tuple[Optional[dict], list[str]]:
    """Parses a composed reply without repairing it."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None, ["RG_PARSE_FAILURE"]
    if not isinstance(value, dict):
        return None, ["RG_SCHEMA_FAILURE"]
    return value, []


def _caption_numbers(caption: str) -> list[str]:
    """Returns every number a caption states, as written.

    Identifiers are masked before the scan, so the digit inside "visual:1" is not read as a
    measurement. A number written in words is returned as the phrase that was written, because the
    token is what the check compares and reporting the phrase names what the caption actually said.
    """
    masked = _IDENTIFIER_RE.sub(lambda match: "#" * len(match.group(0)), caption)
    found = [match.group(0) for match in CAPTION_NUMBER_RE.finditer(masked)]
    found.extend(match.group(0) for match in _WORD_NUMBER_RE.finditer(masked))
    found.extend(match.group(0).strip() for match in _BARE_HALF_RE.finditer(masked))
    return found


def grounding_values(assertions: Sequence[Mapping[str, Any]],
                     fact_packet: Mapping[str, Any]) -> list[float]:
    """Returns the measured values behind a candidate's declarations.

    The measured value rather than the declared one, because that is what the caption's prose is
    checked against.
    """
    values: list[float] = []
    for item in assertions:
        if not isinstance(item, Mapping):
            continue
        measurement_id = item.get("measurement_id")
        if not measurement_id:
            continue
        measured = measured_value(fact_packet, str(measurement_id))
        if measured is not None:
            values.append(float(measured))
    return values


def declared_display_strings(assertions: Sequence[Mapping[str, Any]],
                             fact_packet: Mapping[str, Any]) -> set[str]:
    """Returns the digit strings a caption may use, one for each measurement it declared."""
    return {_display_string(value) for value in grounding_values(assertions, fact_packet)}


def undeclared_numbers(caption: str, assertions: Sequence[Mapping[str, Any]],
                       fact_packet: Mapping[str, Any]) -> list[str]:
    """Returns the numbers a caption states that no declared measurement accounts for.

    **The comparison is against the measurement, not against the declared value.** Comparing the
    prose against the declaration and the declaration against the measurement let two windows
    compose, so a caption could put a number on the display further from the truth than either
    check permitted on its own.

    Exposed so an analysis of an archived run can report which numbers went undeclared, rather than
    only that some did. The release records the reason code; this recovers the token behind it.
    """
    permitted = declared_display_strings(assertions, fact_packet)
    return [token for token in _caption_numbers(caption) if token not in permitted]


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

# Singular nouns that end in "s". A visual label is rendered into the release with a verb, and
# testing only for a trailing "s" produced "Possible bus are visible in the left".
_SINGULAR_S_ENDINGS = ("ss", "us", "is", "as", "os")


def _label_is_plural(label: str) -> bool:
    """True when a visual observation's label takes a plural verb in the rendered sentence."""
    words = str(label).split()
    if not words:
        return False
    last = words[-1].lower()
    return last.endswith("s") and not last.endswith(_SINGULAR_S_ENDINGS)


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
    detector_classes: Iterable[str],
) -> tuple[list[str], list[dict]]:
    """Checks a composed candidate and returns its reason codes and scored assertions.

    A declared value must be the measured value at the precision it is displayed to. There is no
    tolerance, because the model is handed the measurement and restating it is transcription rather
    than estimation: a number the model altered is a number the Fact Packet did not supply, however
    small the alteration.

    Every declaration appears in the scored list, including one the schema check refused, so the
    count of what the model declared is the count of what it declared. The scored assertions are
    returned whether or not the candidate is accepted, because the disagreement between a declared
    value and its measurement is the measurement this design exists to produce and is wanted for a
    rejected candidate as much as an accepted one. `absolute_error_m` is recorded on every
    comparable declaration, so the size of a disagreement stays reportable although any
    disagreement rejects.

    `detector_classes` carries no default. An empty vocabulary turns the object check into a
    no-operation, and a property that holds by construction must not be able to switch itself off
    because a caller left an argument out.
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

    # Scanned once, then used both to score each declaration and to find prose numbers that no
    # declaration accounts for.
    caption_tokens = set(_caption_numbers(caption))

    for index, item in enumerate(assertions):
        if not isinstance(item, Mapping) \
                or set(item) != {"fact_id", "measurement_id", "stated_value"}:
            errors.append("RG_SCHEMA_FAILURE")
            scored.append({"index": index, "outcome": "MALFORMED"})
            continue
        fact_id = item.get("fact_id")
        measurement_id = item.get("measurement_id")
        stated = item.get("stated_value")
        if not isinstance(stated, (int, float)) or isinstance(stated, bool):
            errors.append("RG_SCHEMA_FAILURE")
            scored.append({"index": index, "fact_id": fact_id,
                           "measurement_id": measurement_id, "outcome": "MALFORMED"})
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
        entry["absolute_error_m"] = round(abs(float(stated) - float(measured)), 3)
        # The declared value must be the measurement as displayed, not a value that rounds onto it.
        # Rounding both sides scored a declaration of 1.7449 against a measurement of 1.74 as
        # agreement, so a model altering a value below the display precision was counted as
        # transcribing it faithfully and the agreement rate reported in Chapter 5 was inflated.
        entry["exact"] = float(stated) == hdsg.display_value(measured)
        # Whether the prose actually carries the value this declaration claims to account for. A
        # declaration the caption never states widens the set of numbers the prose is allowed to
        # contain without the model having written anything, so the two are recorded separately.
        entry["stated_in_caption"] = _display_string(measured) in caption_tokens
        if not entry["exact"]:
            errors.append("RG_STATED_VALUE_MISMATCH")
            entry["outcome"] = "DISAGREES"
        else:
            entry["outcome"] = "AGREES"
        scored.append(entry)

    # Every number the prose states must be a declared measurement written as the system displays
    # it, whether the model wrote it in digits or in words. Without this an invented distance
    # reaches the display simply by being omitted from the declarations.
    # RG_DIRECT_NUMBER_DETECTED already means a number reached the text without provenance, which
    # is exactly what an undeclared number is under this design, so reusing it keeps the
    # reason-code enumeration unchanged.
    if caption_tokens - declared_display_strings(assertions, fact_packet):
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

Use the measurements below. Write the distances into your sentences in metres, exactly as they are given, digit for digit. Do not round them, do not approximate them, do not drop a trailing zero, and do not write them as words. A measurement given as 1.74 metres is written as 1.74 metres. A measurement given as 2.00 metres is written as 2.00 metres, not as 2 metres and not as two metres.

Then declare every number you wrote. For each one give the fact_id and the measurement_id it came from, exactly as they appear below, and the value you stated. They are different: fact_id looks like sector:centre, measurement_id looks like m:sector:centre:clearance. For a clearance of 1.74 metres at the centre, the declaration is:

  {{"fact_id": "sector:centre", "measurement_id": "m:sector:centre:clearance", "stated_value": 1.74}}

A number in the caption that is not declared, or any number that differs from its measurement, causes the caption to be discarded. This applies to a distance written as a word as much as to one written in digits.

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
                # Presented through the shared formatter, so what the model is asked to write is
                # the string the deterministic renderer would have displayed for the same value.
                parts.append(f"measured {hdsg._format_measurement(value)}")
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
        f"{'are' if _label_is_plural(item['proposed_label']) else 'is'} "
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
             "formatted_value": hdsg._format_measurement(item["measured_value"])}
        for item in scored_assertions
        if item.get("measured_value") is not None and item.get("measurement_id")
    }
    # The scene binding names the facts the released reason text rests on, so for an accepted
    # caption it is the facts the caption declared. Inheriting it from the base release recorded the
    # binding of the deterministic sentence that was composed and then discarded, which is a
    # different set: a caption describing the centre was released carrying "sector:right" as its
    # evidence. The action binding is left as the deterministic layer produced it, because the
    # action really is the deterministic one.
    scene_ids = list(dict.fromkeys(
        str(item["fact_id"]) for item in scored_assertions if item.get("fact_id")
    ))
    release["evidence"] = {
        **base["evidence"],
        "scene_binding_fact_ids": scene_ids,
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
