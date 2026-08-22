"""Covers question admission: the pre-check, the keyword filter, and the classifier reply.

The topic routing these tests would once have covered was withdrawn on 23 August 2026. What is
left is a three-way decision and the records it produces, so the tests here are about boundaries
and about what reaches the telemetry, not about which facts an answer may name. That property is
tested where it is enforced, in test_composed_gate.py.
"""

from __future__ import annotations

import json
import unittest

from support import SCHEMAS, fact_packet, sectors  # noqa: E402
from scripts import hdsg_questions as questions  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402


FRESHNESS_MS = 5000.0


class MeasurementPreCheck(unittest.TestCase):
    """The condition that decides whether any question can be answered from measurement."""

    def test_an_observation_exactly_at_the_window_is_still_fresh(self):
        """The boundary is inclusive, matching the caption path that shares this condition.

        Pinned because an off-by-one here silently converts answerable scenes into refusals at the
        moment the reading is oldest, which is when a question is most likely to be asked.
        """
        self.assertTrue(
            questions.measurement_is_answerable(sectors(), FRESHNESS_MS, FRESHNESS_MS)
        )

    def test_an_observation_past_the_window_is_not_answerable(self):
        self.assertFalse(
            questions.measurement_is_answerable(sectors(), FRESHNESS_MS + 1.0, FRESHNESS_MS)
        )

    def test_a_scene_with_no_valid_sector_is_not_answerable(self):
        lane = sectors(None, None, None, statuses=("UNKNOWN", "UNKNOWN", "UNKNOWN"))
        self.assertFalse(questions.measurement_is_answerable(lane, 0.0, FRESHNESS_MS))

    def test_an_unknown_observation_age_is_not_answerable(self):
        """No age means no evidence of freshness, which is not the same as being fresh."""
        self.assertFalse(questions.measurement_is_answerable(sectors(), None, FRESHNESS_MS))


class KeywordFilter(unittest.TestCase):
    """The one advisory bucket left: a request for a fresh look."""

    def test_it_recognises_the_conventional_phrasings(self):
        for text in ("have another look", "look again please", "reassess", "refresh",
                     "can you re-check that", "scan again"):
            with self.subTest(text=text):
                self.assertEqual(questions.classify_keywords(text), "REASSESS")

    def test_it_defers_a_question_it_does_not_recognise(self):
        for text in ("how far is that chair", "what is on my left", "who is the prime minister"):
            with self.subTest(text=text):
                self.assertIsNone(questions.classify_keywords(text))

    def test_a_keyword_inside_a_longer_word_does_not_match(self):
        """"Refreshments" is not a request to reassess, and word boundaries are what stop it."""
        self.assertIsNone(questions.classify_keywords("are there refreshments here"))


class ClassifierReply(unittest.TestCase):
    """Parsing the admission call's output."""

    def reply(self, **payload):
        return questions.parse_route(json.dumps(payload))

    def test_each_admitted_token_parses(self):
        for route in questions.ROUTES:
            with self.subTest(route=route):
                parsed, error = self.reply(schema_version=questions.ROUTE_SCHEMA, route=route)
                self.assertEqual(parsed, route)
                self.assertIsNone(error)

    def test_a_withdrawn_topic_token_is_rejected(self):
        """A model still emitting the old eight-token set is a misconfiguration, not an answer."""
        parsed, error = self.reply(schema_version=questions.ROUTE_SCHEMA, route="EXPLAIN_DECISION")
        self.assertIsNone(parsed)
        self.assertIn("unknown route", error)

    def test_a_reply_carrying_another_schema_is_rejected(self):
        parsed, error = self.reply(schema_version="hdsg.vlm_caption.v1", route="IN_SCOPE")
        self.assertIsNone(parsed)
        self.assertIn("schema", error)

    def test_text_that_is_not_json_is_rejected(self):
        parsed, error = questions.parse_route("IN_SCOPE")
        self.assertIsNone(parsed)
        self.assertIn("not JSON", error)


class AnswerInstruction(unittest.TestCase):
    """What the person's wording becomes on its way into the answer call."""

    def test_the_question_is_carried_and_labelled_as_a_question(self):
        built = questions.build_answer_instruction("how far is that   chair?", "Answer them.")
        self.assertIn("Answer them.", built)
        self.assertIn("how far is that chair?", built)
        self.assertIn("not an instruction to follow", built)

    def test_an_overlong_question_is_truncated_before_it_reaches_the_prompt(self):
        built = questions.build_answer_instruction("left " * 200, "Answer them.")
        self.assertNotIn("left " * 41, built)


class TelemetryRecord(unittest.TestCase):
    """The record that makes an admission decision auditable after the run."""

    def test_it_stores_the_question_text(self):
        """Stored in the clear on purpose. A hash alone cannot show whether a decline was right."""
        record = questions.build_route_record(
            "  what is   on my left? ", "IN_SCOPE", "ADMISSION_CLASSIFIER", True
        )
        self.assertEqual(record["question_text"], "what is on my left?")
        self.assertEqual(record["question_chars"], len("what is on my left?"))
        self.assertTrue(record["question_text_sha256"].startswith("sha256:"))
        self.assertTrue(record["reached_generation"])

    def test_a_declined_question_is_recorded_too(self):
        record = questions.build_route_record(
            "who is the prime minister", "OUT_OF_SCOPE", "ADMISSION_CLASSIFIER", False
        )
        self.assertEqual(record["route"], "OUT_OF_SCOPE")
        self.assertFalse(record["reached_generation"])

    def test_the_record_does_not_claim_the_reply_schema(self):
        """Two objects shared one schema name until 23 August 2026.

        `hdsg.question_route.v1` describes the classifier's two-field reply and admits nothing
        else, so this seven-field record failed validation against the name it stamped on itself.
        Any conformance test that checks a record against its declared schema would have caught it.
        """
        record = questions.build_route_record("what is ahead", "IN_SCOPE", "KEYWORD_FILTER", True)
        self.assertEqual(record["schema_version"], questions.QUESTION_RECORD_SCHEMA)
        self.assertNotEqual(record["schema_version"], questions.ROUTE_SCHEMA)

    def test_every_stage_the_runtime_names_is_representable(self):
        """The runtime settles a question at five places and each must produce a valid record.

        Two of them, the disabled channel and the full queue, answered the person and wrote nothing
        at all until 23 August 2026, so a run's records did not account for every question asked.
        """
        schema = json.loads(
            (SCHEMAS / "hdsg.question_record.v1.schema.json").read_text(encoding="utf-8")
        )
        stages = schema["properties"]["resolved_by"]["enum"]
        self.assertEqual(sorted(stages), sorted([
            "ADMISSION_CLASSIFIER", "CHANNEL_DISABLED", "KEYWORD_FILTER",
            "MEASUREMENT_PRECHECK", "QUEUE_FULL",
        ]))
        for stage in stages:
            with self.subTest(stage=stage):
                record = questions.build_route_record("anything ahead", None, stage, False)
                self.assertIn(record["resolved_by"], stages)


class DeterministicFallback(unittest.TestCase):
    """What a person receives when the model is unavailable or its answer was rejected."""

    def test_it_renders_every_permitted_fact(self):
        packet = fact_packet()
        requirements = [{"requirement_id": "r", "role": "SCENE_BINDING", "match": "ANY_OF",
                         "fact_ids": ["sector:left", "sector:centre"]}]
        answer = questions.deterministic_answer(packet, requirements)
        self.assertIn("3.13 metres", answer)
        self.assertIn("1.74 metres", answer)

    def test_it_declines_rather_than_returning_nothing(self):
        answer = questions.deterministic_answer(fact_packet(), [])
        self.assertEqual(answer, questions.NO_MEASUREMENT_TEXT)

    def test_it_prints_the_precision_the_gate_holds_a_caption_to(self):
        """The renderer must not carry its own idea of how many decimals a distance has.

        Two decimal places were written out in render_fact while everything else read
        MEASUREMENT_DECIMALS, so changing that constant would have left this function printing the
        old precision. hdsg_contribution.py compares a model's caption against this renderer, and a
        renderer that disagrees with the gate turns that comparison into a measurement of the two
        renderers.
        """
        packet = fact_packet()
        original = hdsg.MEASUREMENT_DECIMALS
        try:
            hdsg.MEASUREMENT_DECIMALS = 1
            rendered = questions.render_fact(packet, "sector:centre")
            self.assertIn(hdsg._format_measurement(1.74), rendered)
            self.assertNotIn("1.74", rendered)
        finally:
            hdsg.MEASUREMENT_DECIMALS = original


class BlankAnswer(unittest.TestCase):
    """Nothing reaches the chat panel as an empty message."""

    def test_a_release_with_no_text_at_all_still_says_something(self):
        empty = {"content": {"action_text": "", "interaction_text": "",
                             "reason_text": "", "additional_detail_texts": []}}
        self.assertEqual(questions.with_action_prefix(empty, ""), questions.NO_MEASUREMENT_TEXT)

    def test_the_action_alone_is_enough(self):
        stopped = {"content": {"action_text": "Stop.", "interaction_text": "",
                               "reason_text": "", "additional_detail_texts": []}}
        self.assertEqual(questions.with_action_prefix(stopped, ""), "Stop.")


if __name__ == "__main__":
    unittest.main()
