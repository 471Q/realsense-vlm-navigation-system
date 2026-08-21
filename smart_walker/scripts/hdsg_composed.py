"""Composed captions with declared assertions, checked against the Fact Packet.

The generative layer composes a navigation caption in its own words, with the measured values
written into the prose, and declares alongside it every numeric assertion it made and the fact each
one refers to. This module parses that reply, checks it, and renders the release.

**Why declarations rather than parsing the prose.** Recovering "about 1.7 metres" from a sentence
and deciding which of three sectors it refers to is linguistic inference, and Chapter 3 rejects
inference inside the safety boundary for that reason. The model states the attribution, so the
comparison against the Fact Packet is exact.

The prose is nonetheless read. It is scanned for instruction words, for commentary, for quoted text,
for the names of absent detector classes, for the numbers it states, and for where it names each
fact. What matters is not how often it is read but what a reading is permitted to do: every one of
those scans can only add a refusal to a candidate the declarations had already permitted, and none
can admit one they had not. No released caption owes its release to a linguistic judgement.

**What is categorical here and what is measured.** Four properties hold by construction, because a
candidate violating any of them is rejected and the deterministic fallback is released instead:

1. The action the user acts on is produced by the deterministic layer and is not derived from the
   caption, so no caption can alter it. The caption is separately screened against a fixed
   vocabulary of instruction words, which lowers the chance of a caption reading as an instruction
   without excluding it: a phrasing outside that vocabulary, such as "the wider side is the easier
   one", is not caught. Only the first half of this property is categorical.
2. Every number reaching the display is a measured value written character for character as the
   deterministic renderer would display it. There is no tolerance and no alternative spelling, so
   one sensor reading reaches the user in one form whether the gate accepted a caption or fell
   back. A measurement of 2.00 is written "2.00"; "2 metres" and "two metres" are rejected.
3. Every number the caption states is declared, so an undeclared one cannot pass unchecked. Two
   checks hold this between them. Every numeral must be a declared measurement written as
   displayed, and every unit of length must carry a numeral immediately in front of it. The second
   is what covers a distance written without digits, and it screens the unit rather than the
   number because enumerating the ways English states a quantity does not terminate: a list of
   twenty number words and six unit spellings let "two m", "a metre", "a two-metre gap", "two feet"
   and "half of a metre" through undeclared. A hedged quantity such as "a few metres" is refused by
   the same rule, which settles part of the qualitative-assertion question recorded in
   HDSG_VERIFIED_GENERATION_POLICY.md section 7.2: a hedge attached to a unit is refused, while a
   qualitative claim carrying no unit, "the left is the wider side", is untouched.
4. No detector class absent from the Fact Packet can be named in its singular or its plural form,
   which prevents the object hallucination Chapter 2 section 2.7.1 documents rather than reducing
   its rate. Plurals are enumerated rather than guessed by suffix, since a suffix misses "people"
   and person is the class a walker is most often wrong about. Those two forms are the whole of the
   screen: a compound such as "armchair" carries no word boundary before the class name and is not
   caught, so the property covers naming a class and not every way of alluding to one.

**The fifth property, which is a filter rather than a construction.** The four above each constrain
a value or a name in isolation, and none of them constrains which fact a sentence attaches a value
to. A caption declaring all three sector clearances correctly and then writing each one against the
wrong sector satisfied every one of them: every number measured, every number declared, every
declaration in agreement, and all three statements false. The fifth check therefore reads the prose,
and a number must stand nearer to the fact it was declared against than to any other.

The reading is mechanical and its result is used in one direction only. Every place the caption
names a fact is located, with overlapping names resolved in favour of the longer, so that "right in
front of the walker" names the centre rather than the right. Each number is then attributed to the
nearest of those names lying in the number's own sentence. Three cases are refused: the nearest name
is not a fact the number was declared against, two different facts are equally near, or the sentence
names no fact at all. The vocabulary is closed, eleven words and phrases across the three sectors,
and for an object it is the detector's label for it.

Confining the search to one sentence is what allows a caption of more than one sentence. Without it
"The left is open for 3.13 metres. The right is tighter at 1.16 metres." put 3.13 exactly as far
from "left" as from "right" and was refused as ambiguous on account of a word in the sentence after
it.

Nearness rather than clause membership, and the difference is not cosmetic. The first form of this
check cut the caption on punctuation and coordinating words and required the fact to be named inside
the same piece. It refused four of ten naturally worded truthful captions, because a subordinator is
exactly where a subject stops being repeated: "The centre, which is 1.74 metres" leaves the number
in a fragment carrying no subject, and so does every appositive. Distance to the nearest name does
not depend on where a clause was judged to begin.

Attribution is still taken from the declaration, so nothing is accepted on the strength of a
linguistic reading; the reading can only refuse. Chapter 3 paragraph 740 objects to attribution
being inferred, and it is not: the inference here subtracts from what the declarations already
permit.

What the filter does not reach is recorded rather than hidden. A negated sentence naming its fact
passes, a falsehood carrying no number is untouched, and the construction nobody anticipated is by
definition not covered. The residual rate is measured by hand against a sample of accepted captions,
because only a reader finds what the check does not model.

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


# Taken from the runtime rather than repeated. Written out here as a second literal, the two were
# free to disagree, and an identifier the frozen schema set pins is the last thing that should have
# two definitions.
CAPTION_SCHEMA = hdsg.CAPTION_SCHEMA

# Used when the profile states no limit of its own, which at present is always: no profile sets
# max_text_chars. Read with an explicit test for absence rather than through `or`, so that a profile
# setting a limit of zero is honoured instead of being silently replaced by this default.
MAX_CAPTION_CHARS = 400


def caption_char_limit(prompt_packet: Mapping[str, Any]) -> int:
    """Returns the character limit a caption is held to."""
    limit = prompt_packet["response_constraints"].get("max_text_chars")
    return MAX_CAPTION_CHARS if limit is None else int(limit)


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

# Every way of writing a unit of length. The abbreviations are admitted only where a digit or a
# space precedes them, which is what keeps the "m" of "warm" and the "in" of "into" out.
_DISTANCE_UNIT_RE = re.compile(
    r"\b(?:metres?|meters?|centimetres?|centimeters?|millimetres?|millimeters?|kilometres?"
    r"|kilometers?|feet|foot|inches|inch|yards?)\b"
    r"|(?<=[0-9\s])(?:cm|mm|km|m)\b",
    re.IGNORECASE,
)

# A numeral written immediately before a unit, allowing for the space or hyphen between them.
_VALUE_BEFORE_UNIT_RE = re.compile(r"(?:\d+(?:\.\d+)?|\.\d+)[\s\-]*$")


def unquantified_units(caption: str) -> list[str]:
    """Returns each place the caption names a unit of length without a numeral before it.

    **The units are screened, not the numbers.** An earlier design hunted for the numbers instead,
    matching digits and a list of twenty number words followed by a list of unit spellings. A list
    of spellings is always shorter than English, and five ways of stating a distance went through
    the gate undetected and undeclared: "two m", because "m" was not in the unit list; "a metre",
    because "a" was not in the number list; "a two-metre gap", because the pattern demanded a space;
    "two feet", because the units were metric only; and "half of a metre", because the phrase did
    not match the one spelling of a bare half that was anticipated. Extending the lists does not
    close the class, since the next phrasing is not in them either.

    Screening the unit inverts the problem. A distance is stated by naming a unit, so every unit in
    the caption must carry a numeral immediately in front of it, and that numeral is then held to
    the same exactness as any other. Anything else, a word, an article, a hedge or nothing at all,
    is refused without needing to have been foreseen.

    The offending text is returned rather than a count, so an analysis of an archived run can quote
    what the model wrote.
    """
    masked = _mask_identifiers(caption)
    found: list[str] = []
    for match in _DISTANCE_UNIT_RE.finditer(masked):
        if _VALUE_BEFORE_UNIT_RE.search(masked[:match.start()]):
            continue
        found.append(caption[max(0, match.start() - 20):match.end()].strip())
    return found


def parse_caption_candidate(raw: str) -> tuple[Optional[dict], list[str]]:
    """Parses a composed reply without repairing it."""
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None, ["RG_PARSE_FAILURE"]
    if not isinstance(value, dict):
        return None, ["RG_SCHEMA_FAILURE"]
    return value, []


def _mask_identifiers(caption: str) -> str:
    """Blanks out every identifier, preserving length so offsets into the original still hold."""
    return _IDENTIFIER_RE.sub(lambda match: "#" * len(match.group(0)), caption)


def _caption_number_spans(caption: str) -> list[tuple[int, int, str]]:
    """Returns the position and text of every numeral a caption states.

    Positions are offsets into the caption as given, which the attribution check needs in order to
    measure how far a number sits from the fact it names. Identifiers are masked before the scan,
    with the mask the same length as what it replaces, so the digit inside "visual:1" is not read as
    a measurement and every remaining offset is still an offset into the original string.

    Numerals only. A distance written in words carries no numeral and is caught by
    `unquantified_units`, which screens the unit rather than the number.
    """
    masked = _mask_identifiers(caption)
    return [(m.start(), m.end(), m.group(0)) for m in CAPTION_NUMBER_RE.finditer(masked)]


def _caption_numbers(caption: str) -> list[str]:
    """Returns every number a caption states, as written.

    A number written in words is returned as the phrase that was written, because the token is what
    the check compares and reporting the phrase names what the caption actually said.
    """
    return [token for _, _, token in _caption_number_spans(caption)]


def grounded_display_strings(scored: Sequence[Mapping[str, Any]]) -> set[str]:
    """Returns the digits a caption may use, one for each declaration the gate resolved.

    **Built from the scored declarations, not from the raw ones.** An earlier form resolved every
    measurement identifier the candidate mentioned, whether or not the prompt had offered it and
    whether or not it belonged to the fact the declaration named. A declaration the gate was about
    to refuse therefore still licensed its value to appear in the prose, so the run recorded the
    reference error and not the undeclared number that came with it.
    """
    return {
        _display_string(item["measured_value"]) for item in scored
        if item.get("measured_value") is not None
    }


def undeclared_numbers(caption: str, scored: Sequence[Mapping[str, Any]]) -> list[str]:
    """Returns the numerals a caption states that no accepted declaration accounts for.

    **The comparison is against the measurement, not against the declared value.** A caption is
    held to what the sensor reported, so a declaration that misstates its own measurement cannot
    license the misstatement in the prose as well.

    Exposed so an analysis of an archived run can report which numbers went undeclared, rather than
    only that some did. The release records the reason code; this recovers the token behind it.
    """
    grounded = grounded_display_strings(scored)
    return [token for token in _caption_numbers(caption) if token not in grounded]


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


# The words that denote each sector, as a closed list rather than as a similarity test. A caption
# describing the space in front of the walker says "ahead" far more often than "centre", so the
# variants are enumerated; omitting them refuses ordinary phrasing. "right in front" and "right
# ahead" are listed against the centre because the longest match wins, which is what stops the
# "right" inside them from being read as the right-hand sector.
SECTOR_TERMS = {
    "sector:left": ("left",),
    "sector:centre": ("centre", "center", "ahead", "forward", "in front", "straight ahead",
                      "directly ahead", "right in front", "right ahead"),
    "sector:right": ("right",),
}


def _fact_terms(fact_id: str, fact_packet: Mapping[str, Any]) -> tuple[str, ...]:
    """Returns the words a caption may use to name one fact."""
    if fact_id in SECTOR_TERMS:
        return SECTOR_TERMS[fact_id]
    for item in fact_packet.get("objects", []):
        if item.get("fact_id") == fact_id:
            label = item.get("canonical_label") or item.get("raw_label")
            return (str(label).lower(),) if label else ()
    return ()


def _fact_mentions(caption: str, fact_packet: Mapping[str, Any]) -> list[tuple[int, int, str]]:
    """Returns where the caption names each fact, as (start, end, fact_id).

    Overlapping matches are resolved in favour of the longer term, which is how "right in front of
    the walker" is read as one mention of the centre rather than as a mention of the right.

    Identifiers are masked first, so a model quoting "sector:centre" into the prose does not thereby
    name the centre. Quoting an identifier is not describing a place.
    """
    masked = _mask_identifiers(caption).lower()
    fact_ids = list(SECTOR_TERMS) + [
        str(item["fact_id"]) for item in fact_packet.get("objects", []) if item.get("fact_id")
    ]
    found: list[tuple[int, int, str]] = []
    for fact_id in fact_ids:
        for term in _fact_terms(fact_id, fact_packet):
            if not term:
                continue
            found.extend((m.start(), m.end(), fact_id)
                         for m in re.finditer(rf"\b{re.escape(term)}\b", masked))
    found.sort(key=lambda span: (span[0], span[0] - span[1]))
    kept: list[tuple[int, int, str]] = []
    for span in found:
        if any(span[0] < other[1] and other[0] < span[1] for other in kept):
            continue
        kept.append(span)
    return kept


# Sentence boundaries, used to stop a fact named in one sentence from competing for a number stated
# in another. A full stop between two digits is not a boundary, since it is a decimal point.
_SENTENCE_END_RE = re.compile(r"(?<!\d)[.;!?](?!\d)")


def _sentence_bounds(caption: str, position: int) -> tuple[int, int]:
    """Returns the span of the sentence containing a position."""
    start = 0
    for match in _SENTENCE_END_RE.finditer(caption):
        if match.end() > position:
            return start, match.start()
        start = match.end()
    return start, len(caption)


def _distance(number: tuple[int, int, str], mention: tuple[int, int, str]) -> int:
    if mention[1] <= number[0]:
        return number[0] - mention[1]
    if number[1] <= mention[0]:
        return mention[0] - number[1]
    return 0


def attribution_failures(caption: str, scored: Sequence[Mapping[str, Any]],
                         fact_packet: Mapping[str, Any]) -> list[dict]:
    """Returns the numbers a caption states away from the fact they were declared against.

    Each number is attributed to the nearest fact the caption names, measured in characters, and
    that fact must be one the number was declared against. A number naming no fact at all cannot be
    attributed, and a number equidistant between two different facts is ambiguous; both are
    reported.

    **Nearest mention rather than clause membership.** The first implementation cut the caption into
    clauses on punctuation and coordinating words and required the fact to be named inside the same
    clause. That refused four of ten naturally worded truthful captions, because a subordinator is
    exactly where the subject stops being repeated: "The centre, which is 1.74 metres" leaves the
    number in a fragment carrying no subject at all, and so do "The left, at 3.13 metres" and every
    other appositive. Distance to the nearest mention does not depend on where a clause was judged
    to begin, and it still refuses the swap the check exists for, because in "the left side measures
    1.16 metres" the nearest fact named is the left and 1.16 belongs to the right.

    Reported rather than raised, so an analysis of an archived run can quote what failed.
    """
    owners: dict[str, list[str]] = {}
    for entry in scored:
        measured = entry.get("measured_value")
        if measured is None or not entry.get("fact_id"):
            continue
        owners.setdefault(_display_string(measured), []).append(str(entry["fact_id"]))
    if not owners:
        return []

    mentions = _fact_mentions(caption, fact_packet)
    failures: list[dict] = []
    for number in _caption_number_spans(caption):
        token = number[2]
        candidates = owners.get(token, [])
        if not candidates:
            continue  # an undeclared number, refused by the check above and not attributable here
        # Only the sentence the number sits in is searched. Without this, "The left is open for
        # 3.13 metres. The right is tighter at 1.16 metres." put 3.13 exactly as far from "left" as
        # from "right", and a truthful caption was refused as ambiguous because of a fact named in
        # the sentence after it.
        low, high = _sentence_bounds(caption, number[0])
        local = [m for m in mentions if m[0] >= low and m[1] <= high]
        if not local:
            failures.append({"number": token, "declared_for": candidates,
                             "reason": "FACT_NOT_NAMED"})
            continue
        shortest = min(_distance(number, mention) for mention in local)
        nearest = {mention[2] for mention in local if _distance(number, mention) == shortest}
        if len(nearest) > 1:
            failures.append({"number": token, "declared_for": candidates,
                             "reason": "AMBIGUOUS", "nearest": sorted(nearest)})
        elif not nearest & set(candidates):
            failures.append({"number": token, "declared_for": candidates,
                             "reason": "MISATTRIBUTED", "nearest": sorted(nearest)})
    return failures


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
    if len(caption) > caption_char_limit(prompt_packet):
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

    # The measurement each permitted fact carries, so a declaration can be checked against the
    # pairing the prompt offered rather than only against the two lists separately.
    measurement_of_fact = {
        item["fact_id"]: (item["measurement"]["measurement_id"]
                          if isinstance(item.get("measurement"), Mapping) else None)
        for item in prompt_packet["permitted_facts"]
    }
    permitted_measurements = {value for value in measurement_of_fact.values() if value}
    permitted_fact_ids = set(measurement_of_fact)

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
        # The measurement must be the one this fact carries. Checking the two lists separately let
        # a candidate pair any permitted fact with any permitted measurement, and that pairing is
        # the whole of the attribution: declaring fact_id sector:left against
        # m:sector:centre:clearance made 1.74 the left sector's measured value as far as every
        # later check was concerned, so "The left side is clear for 1.74 metres" was released
        # against a left sector measured at 3.13. Nothing else detects it. The scored outcome is
        # kept distinct from a measurement the prompt never offered, because a model pairing two
        # real identifiers wrongly is not doing the same thing as one inventing an identifier.
        if measurement_id != measurement_of_fact[fact_id]:
            errors.append("RG_MEASUREMENT_REFERENCE_INVALID")
            entry["outcome"] = "MEASUREMENT_NOT_FOR_FACT"
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
    #
    # The shared helper is called rather than the same set difference written out again. The two
    # had been written twice, which is how a check and the analysis that reports on it drift apart:
    # a run would have recorded the code while the helper listing the offending tokens disagreed
    # about which they were.
    if undeclared_numbers(caption, scored):
        errors.append("RG_DIRECT_NUMBER_DETECTED")

    # And every unit of length must carry a numeral in front of it. The check above sees numerals
    # only, so a distance written without one, "two m", "a metre", "a two-metre gap", passed it
    # untouched and reached the display undeclared. Screening the unit catches the class rather than
    # the spellings of it. Same code: a distance with no provenance is what both checks find.
    if unquantified_units(caption):
        errors.append("RG_DIRECT_NUMBER_DETECTED")

    # A number must sit in a clause that names the fact it was declared against. Without this a
    # caption could declare all three sector clearances correctly and write each one against the
    # wrong sector, which every other check passes and which is false in every clause.
    #
    # RG_SUBJECT_MISMATCH is the code for it. Under the templated contract it meant a clause whose
    # subject was not the fact the clause was chosen for, which is the same failure read off a
    # sentence the runtime had written rather than one the model composed. Reviving it keeps the
    # enumeration unchanged and names the failure accurately.
    if attribution_failures(caption, scored, fact_packet):
        errors.append("RG_SUBJECT_MISMATCH")

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

Keep each distance in the same clause as the thing it belongs to, and name that thing in words: write "the centre is clear for 1.74 metres", not "the centre is the widest, at 1.74 metres". A distance in a clause that does not name what it measures causes the caption to be discarded.

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
        max_chars=caption_char_limit(prompt_packet),
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
    scored_assertions: Sequence[Mapping[str, Any]],
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

    `scored_assertions` carries no default. The accepted release's evidence is built from it, so a
    caller omitting it produced a release recording no scene binding and no measurement
    substitutions for a caption that rested on both, and nothing failed.
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
        # A declaration the caption never states widens the set of numbers the prose is allowed to
        # contain without the model having written anything. The flag was recorded on each entry
        # and then left out of the aggregate, so the one place Chapter 5 reads could not see it.
        "not_stated_in_caption": sum(
            1 for item in comparable if item.get("stated_in_caption") is False
        ),
        "agreement_rate": (
            round(len(agreeing) / len(comparable), 3) if comparable else None
        ),
        "mean_absolute_error_m": (
            round(sum(errors) / len(errors), 3) if errors else None
        ),
        "max_absolute_error_m": round(max(errors), 3) if errors else None,
    }
