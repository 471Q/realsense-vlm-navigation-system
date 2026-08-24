"""What a model reply is allowed to contain before anything scores it.

Python's JSON decoder accepts three bare tokens the format does not define, NaN, Infinity and
-Infinity, and its encoder writes them out again. A reply carrying one therefore parsed, scored, and
was written into the telemetry as a bare word, leaving a log line that Python reads back and a
strict reader cannot. That file is the evidence Chapter 5 rests on.

A non-finite declared value also poisons the event's `mean_absolute_error_m`, and one such event
turns an average across events into NaN or drops it, depending on the tool.

The generation grammar cannot produce any of the three: its number rule is a sign, digits, and an
optional fractional part. The exposure is the paths carrying no grammar by design, the
`--unconstrained` diagnostic mode and the C0 and C1 baselines, the latter also carrying no gate.
Those are the runs whose figures sit beside the gated ones.
"""

from __future__ import annotations

import json
import unittest

import support
from support import caption, event
from scripts import hdsg_runtime as hdsg  # noqa: E402
from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_baselines as baselines  # noqa: E402
from scripts import hdsg_questions as questions  # noqa: E402

NOT_JSON_VALUES = ("NaN", "Infinity", "-Infinity")


class StrictLoader(unittest.TestCase):

    def test_the_three_undefined_tokens_are_refused(self):
        for token in NOT_JSON_VALUES:
            with self.subTest(token):
                with self.assertRaises(ValueError):
                    hdsg.loads_strict('{"value": %s}' % token)

    def test_ordinary_json_is_unaffected(self):
        self.assertEqual({"value": 1.74}, hdsg.loads_strict('{"value": 1.74}'))
        self.assertEqual({"value": None}, hdsg.loads_strict('{"value": null}'))

    def test_a_number_whose_digits_overflow_to_infinity_is_refused(self):
        """The second way in, and the one the grammar can produce. Its number rule places no bound
        on the digits, so a value of "1" followed by four hundred zeros and a fractional part
        converts to inf without complaint. A bare integer of the same length is safe, Python's
        integers being arbitrary precision, which is why the fractional part matters."""
        with self.assertRaises(ValueError):
            hdsg.loads_strict('{"value": 1' + "0" * 400 + '.0}')

    def test_a_large_finite_number_still_parses(self):
        """A finite value is a disagreement to be measured, not a reply to be refused."""
        self.assertEqual(1e300, hdsg.loads_strict('{"value": 1e300}')["value"])
        self.assertEqual(10 ** 400, hdsg.loads_strict('{"value": 1' + "0" * 400 + "}")["value"])

    def test_the_failure_is_the_one_callers_already_handle(self):
        """`json.JSONDecodeError` subclasses `ValueError`, so a caller catching a parse failure
        catches this without change."""
        with self.assertRaises(ValueError):
            hdsg.loads_strict("{not json")


class TheThreeReplyParsers(unittest.TestCase):
    """Every place a model's reply is read. Two of the three carry no grammar behind them."""

    def test_the_caption_gate_reports_a_parse_failure(self):
        for token in NOT_JSON_VALUES:
            with self.subTest(token):
                value, codes = composed.parse_caption_candidate('{"caption": %s}' % token)
                self.assertIsNone(value)
                self.assertEqual(["RG_PARSE_FAILURE"], codes)

    def test_the_question_router_reports_a_parse_failure(self):
        route, reason = questions.parse_route('{"schema_version": "x", "route": NaN}')
        self.assertIsNone(route)
        self.assertIn("not JSON", reason)

    def test_the_baselines_refuse_it_wrapped_in_prose_as_well(self):
        """The baselines tolerate commentary around the object, so the second parse needs the same
        strictness as the first. Refusing only the first would let a wrapped reply through."""
        self.assertEqual((None, "JSON object did not parse"),
                         baselines.parse_baseline_response('{"a": NaN}'))
        self.assertEqual((None, "JSON object did not parse"),
                         baselines.parse_baseline_response('here it is {"a": Infinity} ok'))

    def test_an_ordinary_baseline_reply_still_reads(self):
        self.assertEqual(({"a": 1.5}, None), baselines.parse_baseline_response('{"a": 1.5}'))


class TheScorerStandsBehindTheParse(unittest.TestCase):
    """A caller that parsed loosely must not be able to poison an aggregate.

    The parse is the first defence and this is the second. Both are wanted: the guarantee should not
    rest on every future caller having used the strict loader.
    """

    def setUp(self):
        self.packet, self.prompt = event()

    def score(self, value):
        candidate = caption("The centre is clear for 1.74 metres.",
                            [{"fact_id": "sector:centre",
                              "measurement_id": "m:sector:centre:clearance",
                              "stated_value": value}])
        return composed.validate_caption_candidate(
            candidate, self.prompt, self.packet, detector_classes=["chair"])

    def test_a_non_finite_declaration_is_a_schema_failure(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value):
                codes, _ = self.score(value)
                self.assertIn("RG_SCHEMA_FAILURE", codes)

    def test_the_event_summary_stays_writable_as_json(self):
        """It was written with a bare NaN, which Python reads back and a strict reader does not."""
        _, scored = self.score(float("nan"))
        summary = composed.assertion_summary(scored)
        self.assertIsNone(summary["mean_absolute_error_m"])
        json.loads(json.dumps(summary),
                   parse_constant=lambda token: self.fail(f"the log carries a bare {token}"))

    def test_a_finite_declaration_is_still_scored(self):
        """The check must refuse non-finite values without refusing a large wrong one, which is a
        disagreement this design exists to measure."""
        codes, scored = self.score(9999.0)
        self.assertIn("RG_STATED_VALUE_MISMATCH", codes)
        self.assertEqual(9997.26, composed.assertion_summary(scored)["mean_absolute_error_m"])


class WhatTheGrammarCanAndCannotProduce(unittest.TestCase):
    """Which paths were exposed to which of the two routes.

    The bare tokens are unreachable under the grammar and reachable on the `--unconstrained` mode
    and the two baselines. The overflow is reachable everywhere, including the release path, because
    the number rule bounds the shape of a number and not its length. The first reading of this
    block reported the release path as safe on the strength of the rule admitting digits only, which
    was true of the tokens and not of the arithmetic.
    """

    def setUp(self):
        self.grammar = (support.CONFIG / "hdsg.vlm_caption.v1.gbnf").read_text(encoding="utf-8")

    def test_the_number_rule_admits_digits_only(self):
        self.assertIn('number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)?', self.grammar)
        for token in NOT_JSON_VALUES:
            self.assertNotIn(token, self.grammar)

    def test_the_number_rule_places_no_bound_on_the_digits(self):
        """Which is what makes the overflow reachable under the grammar. Recorded rather than
        changed: a length bound in the grammar is a fourth statement of a rule already written in
        three notations, and the parser refuses the value that results."""
        self.assertIn("[0-9]*", self.grammar)
        self.assertNotIn("[0-9]{", self.grammar)


if __name__ == "__main__":
    unittest.main()
