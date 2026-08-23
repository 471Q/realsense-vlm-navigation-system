"""The composed caption's decoding constraint, against what the runtime actually produces.

The grammar decides what the model is physically able to write. Everything about meaning belongs to
the gate, so this module asserts one property only: that every identifier the runtime can put in
front of the model is one the model is able to write back.

That property failed until 23 August 2026. An object is named by its tracker identifier where it has
one and by its position in the detection list otherwise, and that position counts from zero, but the
grammar required object numbers to begin with a digit from one to nine. `object:0` was therefore a
permitted fact the decoder could not produce. The model was forced onto some other digit, naming an
object that either did not exist or was not the one measured, and the caption was rejected as an
invalid reference or a value mismatch, with the reason code blaming the model. No archived run
contains `object:0`, the tracker having issued identifiers from one throughout, so it had never
fired. It is reachable whenever the tracker holds no identifier, which is the opening frames of a
run and any recovery after it loses every track.

The rules are read out of the grammar file and turned into matchers rather than transcribed, because
a transcription is a second copy that can disagree with the first without anything saying so.
"""

from __future__ import annotations

import json
import re
import unittest

from support import (  # noqa: F401
    CONFIG,
    SCHEMAS,
    caption,
    declares_object,
    detected_object,
    event,
    gate,
)

GRAMMAR = CONFIG / "hdsg.vlm_caption.v1.gbnf"
SCHEMA = SCHEMAS / "hdsg.vlm_caption.v1.schema.json"


def grammar_rules() -> dict[str, str]:
    """The grammar's rules, by name. Comment lines and blank lines are dropped."""
    rules = {}
    for line in GRAMMAR.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "::=" not in line:
            continue
        name, _, body = line.partition("::=")
        rules[name.strip()] = body.strip()
    return rules


def as_regex(rule_name: str, rules: dict[str, str], depth: int = 0) -> str:
    """Turns one grammar rule into an equivalent regular expression.

    Handles only what the identifier rules use: quoted literals, character classes, the `*`
    repetition, alternation and references to other rules. It is not a general GBNF reader and is not
    meant to be. Anything it cannot interpret raises, so a rule that grows beyond this shape fails
    the tests rather than being silently skipped.
    """
    if depth > 10:
        raise AssertionError(f"rule {rule_name} recurses")
    body = rules[rule_name]
    out = []
    index = 0
    while index < len(body):
        char = body[index]
        if char == " ":
            index += 1
        elif char == "|":
            out.append("|")
            index += 1
        elif char == '"':
            end = body.index('"', index + 1)
            out.append(re.escape(body[index + 1:end].replace('\\"', '"')))
            index = end + 1
        elif char == "[":
            end = body.index("]", index)
            out.append(body[index:end + 1])
            index = end + 1
        elif char == "*":
            out.append("*")
            index += 1
        elif char == "(":
            out.append("(?:")
            index += 1
        elif char == ")":
            out.append(")")
            index += 1
        else:
            match = re.match(r"[A-Za-z][A-Za-z0-9_-]*", body[index:])
            if match is None:
                raise AssertionError(f"rule {rule_name} uses syntax this reader does not handle: "
                                     f"{body[index:]!r}")
            out.append("(?:" + as_regex(match.group(0), rules, depth + 1) + ")")
            index += len(match.group(0))
    return "".join(out)


def admits(rule_name: str, text: str) -> bool:
    """Whether the grammar rule can produce exactly this text."""
    return re.fullmatch(as_regex(rule_name, grammar_rules()), text) is not None


class GrammarReaderTests(unittest.TestCase):
    """Guards on the reader above. Without these, a reader that quietly produced a regex matching
    everything would make every test in this module pass."""

    def test_the_grammar_declares_the_identifier_rules(self):
        for name in ("fact-id", "measurement-id", "object-index", "sector-name"):
            with self.subTest(name):
                self.assertIn(name, grammar_rules())

    def test_the_reader_refuses_text_outside_a_rule(self):
        self.assertFalse(admits("sector-name", "rear"))
        self.assertFalse(admits("fact-id", "sector:diagonal"))

    def test_the_reader_accepts_text_inside_a_rule(self):
        self.assertTrue(admits("sector-name", "centre"))


class ObjectIndexTests(unittest.TestCase):
    """The defect and its fix."""

    def test_the_grammar_admits_an_object_numbered_zero(self):
        self.assertTrue(admits("fact-id", "object:0"))
        self.assertTrue(admits("measurement-id", "m:object:0:distance"))

    def test_the_grammar_still_admits_ordinary_object_numbers(self):
        for number in ("1", "4", "56", "137"):
            with self.subTest(number):
                self.assertTrue(admits("fact-id", f"object:{number}"))
                self.assertTrue(admits("measurement-id", f"m:object:{number}:distance"))

    def test_an_object_number_is_not_padded_or_signed(self):
        """`object-index` permits a bare zero, not a zero in front of another digit. Otherwise
        object:01 and object:1 name the same object by two spellings."""
        for bad in ("01", "007", "-1", "1.0", ""):
            with self.subTest(bad):
                self.assertFalse(admits("fact-id", f"object:{bad}"))

    def test_a_visual_observation_number_still_starts_at_one(self):
        """Visual numbering is the model's own invention and the gate requires visual:[1-9][0-9]*,
        so `positive-integer` is correct there and must not have been widened with the object rule."""
        self.assertTrue(admits("positive-integer", "1"))
        self.assertFalse(admits("positive-integer", "0"))


class RuntimeAgreementTests(unittest.TestCase):
    """Every identifier the runtime can offer the model, against what the model may write."""

    def test_an_untracked_object_is_expressible(self):
        """The case that produced the defect. A detection the tracker holds no identifier for is
        numbered by its position in the list, and the first position is zero."""
        objects = detected_object(track_id=None)
        self.assertEqual("object:0", objects[0]["fact_id"])
        self.assertTrue(admits("fact-id", objects[0]["fact_id"]))

    def test_every_permitted_fact_in_a_scene_is_expressible(self):
        """The general form. A fact offered to the model that the model cannot name is a caption
        that cannot be written, whatever the fact happens to be."""
        packet, prompt = event(objects=detected_object(track_id=None))
        offered = sorted({fact_id for requirement in prompt["requirements"]
                          for fact_id in requirement["fact_ids"]})
        self.assertTrue(offered, "the scene offered no facts, so this asserts nothing")
        for fact_id in offered:
            with self.subTest(fact_id):
                self.assertTrue(admits("fact-id", fact_id),
                                f"the grammar cannot express the permitted fact {fact_id}")

    def test_the_three_sector_names_agree_with_the_runtime(self):
        for name in ("left", "centre", "right"):
            with self.subTest(name):
                self.assertTrue(admits("fact-id", f"sector:{name}"))
                self.assertTrue(admits("measurement-id", f"m:sector:{name}:clearance"))


class GateAgreementTests(unittest.TestCase):
    """The grammar's output through the gate. Widening the grammar is worth nothing if the gate
    refuses what it now permits."""

    def test_the_gate_accepts_a_declaration_naming_object_zero(self):
        objects = detected_object(track_id=None)
        distance = objects[0]["distance_m"]
        packet, prompt = event(objects=objects)
        candidate = caption(f"A chair is {distance:.2f} metres away on the left.",
                            [declares_object(0, round(distance, 2))])
        codes, scored = gate(candidate, packet, prompt)
        self.assertEqual([], codes)
        self.assertEqual(["object:0"], [item["fact_id"] for item in scored])


class SchemaAgreementTests(unittest.TestCase):
    """The schema against the grammar and against the gate it describes."""

    def schema(self):
        return json.loads(SCHEMA.read_text(encoding="utf-8"))

    def test_the_schema_accepts_an_object_numbered_zero(self):
        pattern = self.schema()["definitions"]["fact_id"]["pattern"]
        self.assertIsNotNone(re.fullmatch(pattern, "object:0"))

    def test_the_schema_does_not_describe_a_tolerance(self):
        """The tolerance was removed and replaced by an exact match on the written token, but the
        schema went on describing it until 23 August 2026. A schema that describes a rule the code
        does not implement is worse than no description, because it is built against.
        """
        described = self.schema()["definitions"]["assertion"]["properties"]["stated_value"]
        self.assertNotIn("more than the frozen tolerance", described["description"])
        self.assertIn("no tolerance", described["description"])


class ThreeStatementsOfOneRuleTests(unittest.TestCase):
    """The same field shape, stated three times in three notations, compared by behaviour.

    A field's permitted shape is written in the grammar, which stops the model producing anything
    else; in the schema, which is the frozen record an archived caption is validated against; and in
    `hdsg_composed`, which screens the parsed candidate. None can be removed. The grammar must be
    GBNF because the decoder reads it, the schema must be JSON Schema because a validator reads it,
    and the third runs in Python on an object that already exists.

    Comparing the three by their text is not possible, the notations being different. They are
    compared by what they accept: each probe string is offered to all three, and the three must
    agree. Object numbering disagreed between the grammar and the schema from the day both were
    written until 23 August 2026, and the only reason it was found is that somebody read both files
    on the same afternoon.
    """

    def schema(self):
        return json.loads(SCHEMA.read_text(encoding="utf-8"))

    def assert_all_three_agree(self, probes, rule_name, schema_field, python_expression,
                               grammar_bounds_length=True):
        """`schema_field` is the whole subschema, so `maxLength` counts alongside `pattern`.

        Comparing the pattern alone would have missed a length disagreement entirely, the schema
        stating its limit in a separate keyword from its shape.

        `grammar_bounds_length` is false where the grammar imposes no length of its own. The limit
        is then a decision of the gate rather than a disagreement, and an over-long probe is skipped
        rather than counted as a conflict.
        """
        pattern = schema_field["pattern"]
        limit = schema_field.get("maxLength")
        for probe in probes:
            with self.subTest(probe):
                if not grammar_bounds_length and limit is not None and len(probe) > limit:
                    self.assertIsNone(python_expression.fullmatch(probe),
                                      "the gate must enforce the length the schema states")
                    continue
                verdicts = {
                    "grammar": admits(rule_name, probe),
                    "schema": (re.fullmatch(pattern, probe) is not None
                               and (limit is None or len(probe) <= limit)),
                    "hdsg_composed": python_expression.fullmatch(probe) is not None,
                }
                self.assertEqual(1, len(set(verdicts.values())),
                                 f"the three statements disagree on {probe!r}: {verdicts}")

    def test_the_three_agree_on_a_visual_observation_number(self):
        from scripts import hdsg_composed as composed
        self.assert_all_three_agree(
            ["visual:1", "visual:2", "visual:17", "visual:0", "visual:01", "visual:",
             "visual:-1", "visual: 1", "sighting:1", "visual:1x"],
            "visual-identifier",
            self.schema()["definitions"]["visual_observation"]
                ["properties"]["candidate_observation_id"],
            composed.VISUAL_ID_RE,
        )

    def test_the_three_agree_on_a_proposed_label(self):
        from scripts import hdsg_composed as composed
        self.assert_all_three_agree(
            ["doorway", "ceiling light", "bright spot", "step_edge", "a", "Doorway",
             "door way sign post", "door-way", "2 steps", "", " doorway", "doorway ",
             "door  way", "a" * 48, "a" * 49, " ".join(["word"] * 12)],
            "proposed-label",
            self.schema()["definitions"]["visual_observation"]["properties"]["proposed_label"],
            composed.PROPOSED_LABEL_RE,
            # The grammar states no length. The 48 character limit is the schema's and the gate's.
            grammar_bounds_length=False,
        )

    def test_the_grammar_and_the_schema_agree_on_a_fact_identifier(self):
        """`hdsg_composed` is absent from this one deliberately. It does not screen the shape of a
        fact identifier at all: it checks membership of the facts this frame actually offers, which
        is the stronger test, since a correctly shaped identifier can still name nothing. That is why
        the gate accepted object:0 throughout the period the grammar refused it.
        """
        pattern = self.schema()["definitions"]["fact_id"]["pattern"]
        for probe in ["sector:left", "sector:centre", "sector:right", "object:0", "object:1",
                      "object:56", "condition:no_clear_sector", "sector:rear", "object:", ""]:
            with self.subTest(probe):
                self.assertEqual(re.fullmatch(pattern, probe) is not None,
                                 admits("fact-id", probe),
                                 f"the grammar and the schema disagree on {probe!r}")

    def test_the_grammar_and_the_schema_agree_on_a_bearing(self):
        allowed = self.schema()["definitions"]["visual_observation"]["properties"]["bearing"]["enum"]
        for probe in allowed + ["FRONT", "left", "REAR", ""]:
            with self.subTest(probe):
                self.assertEqual(probe in allowed, admits("bearing-value", probe))


if __name__ == "__main__":
    unittest.main()
