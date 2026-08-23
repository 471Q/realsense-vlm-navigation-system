"""The admission classifier's three outcomes, across the four places they are written.

A typed question is read by a model call whose decoder is constrained by a grammar. The grammar
admits exactly one of three tokens and nothing else, which is what makes the call safe to point at
the person's raw text: the model reads free text, but it cannot emit anything outside the closed
set, so a crafted message can at worst be admitted when it should have been declined. It cannot
cause the classifier to produce content.

Those three tokens are written out four times: in `config/hdsg.question_route.v1.gbnf`, in the enum
of `schemas/hdsg.question_route.v1.schema.json`, as `ROUTES` in `hdsg_questions`, and in the two
fixtures. Nothing checked that they agreed until 23 August 2026, and neither way of disagreeing
raises an error. An outcome added to the code alone is unreachable, since the decoder cannot emit
it. An outcome added to the grammar alone has every reply rejected by `parse_route` and every
question answered as out of scope. Both are silent.
"""

from __future__ import annotations

import json
import re
import unittest

from support import CONFIG, SCHEMAS  # noqa: F401
from scripts import hdsg_questions as questions

GRAMMAR = CONFIG / "hdsg.question_route.v1.gbnf"
SCHEMA = SCHEMAS / "hdsg.question_route.v1.schema.json"


def grammar_tokens() -> list[str]:
    """The alternatives on the grammar's `route` rule, in the order written.

    Read from the file rather than transcribed. A transcription cannot go out of date visibly, which
    is the failure this module exists to catch.
    """
    text = GRAMMAR.read_text(encoding="utf-8")
    match = re.search(r"^route\s*::=(.*)$", text, re.MULTILINE)
    if match is None:
        raise AssertionError("the grammar has no `route` rule")
    return re.findall(r'"\\"([A-Z_]+)\\""', match.group(1))


def schema_tokens() -> list[str]:
    return json.loads(SCHEMA.read_text(encoding="utf-8"))["properties"]["route"]["enum"]


class AgreementTests(unittest.TestCase):
    """The four lists against each other."""

    def test_the_grammar_declares_three_outcomes(self):
        """A guard on the reader above. A regex that silently matched nothing would make every other
        test here pass by comparing empty sets."""
        self.assertEqual(3, len(grammar_tokens()))

    def test_the_grammar_and_the_schema_agree(self):
        self.assertEqual(schema_tokens(), grammar_tokens())

    def test_the_grammar_and_the_code_agree(self):
        self.assertEqual(list(questions.ROUTES), grammar_tokens())

    def test_the_schema_version_in_the_grammar_matches_the_code(self):
        """The grammar writes the version as a literal string, so a schema rename leaves the model
        emitting the old one and `parse_route` refusing every reply."""
        self.assertIn(f'"\\"{questions.ROUTE_SCHEMA}\\""',
                      GRAMMAR.read_text(encoding="utf-8"))


class AdmissibleRepliesTests(unittest.TestCase):
    """Everything the grammar permits, through the code that reads it.

    The grammar's language is small enough to enumerate: three payloads, plus whitespace. Testing
    the whole language rather than a sample is possible here and is done.
    """

    def replies(self):
        return [('{"schema_version": "%s", "route": "%s"}' % (questions.ROUTE_SCHEMA, token), token)
                for token in grammar_tokens()]

    def test_every_admissible_reply_parses_to_its_own_outcome(self):
        for payload, token in self.replies():
            with self.subTest(token):
                route, error = questions.parse_route(payload)
                self.assertIsNone(error)
                self.assertEqual(token, route)

    def test_every_admissible_reply_validates_against_the_schema(self):
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")
        schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
        for payload, token in self.replies():
            with self.subTest(token):
                jsonschema.validate(json.loads(payload), schema)

    def test_the_whitespace_the_grammar_permits_is_accepted(self):
        """`ws` allows spaces, tabs and newlines at every join, so a reply the decoder can produce
        must not be refused for its formatting."""
        compact = '{"schema_version":"%s","route":"IN_SCOPE"}' % questions.ROUTE_SCHEMA
        spread = '  {\n "schema_version" : "%s",\n "route" : "IN_SCOPE"\n }\n' % questions.ROUTE_SCHEMA
        for payload in (compact, spread):
            with self.subTest(payload[:20]):
                self.assertEqual("IN_SCOPE", questions.parse_route(payload)[0])


class RefusalTests(unittest.TestCase):
    """What `parse_route` refuses. The grammar cannot produce any of these, so each stands against
    a reply arriving from somewhere the grammar did not constrain."""

    def test_an_unknown_outcome_is_refused(self):
        payload = '{"schema_version": "%s", "route": "ALLOW"}' % questions.ROUTE_SCHEMA
        route, error = questions.parse_route(payload)
        self.assertIsNone(route)
        self.assertIn("ALLOW", error)

    def test_a_wrong_schema_version_is_refused(self):
        payload = '{"schema_version": "hdsg.question_route.v2", "route": "IN_SCOPE"}'
        self.assertIsNone(questions.parse_route(payload)[0])

    def test_a_reply_that_is_not_json_is_refused(self):
        self.assertIsNone(questions.parse_route("IN_SCOPE")[0])

    def test_a_reply_that_is_not_an_object_is_refused(self):
        self.assertIsNone(questions.parse_route('["IN_SCOPE"]')[0])


class FixtureTests(unittest.TestCase):
    """The two fixtures the manifest freezes alongside the schema."""

    def test_the_valid_fixture_parses(self):
        payload = (SCHEMAS / "fixtures" / "valid" / "hdsg.question_route.v1.json").read_text(
            encoding="utf-8")
        route, error = questions.parse_route(payload)
        self.assertIsNone(error)
        self.assertIn(route, grammar_tokens())

    def test_the_invalid_fixture_is_refused(self):
        payload = (SCHEMAS / "fixtures" / "invalid" /
                   "hdsg.question_route.v1.unknown_route.json").read_text(encoding="utf-8")
        self.assertIsNone(questions.parse_route(payload)[0])


if __name__ == "__main__":
    unittest.main()
