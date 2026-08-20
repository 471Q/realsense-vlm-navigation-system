"""Tests for composed captions with declared assertions.

Four properties are meant to hold by construction under this design, and the tests that matter are
the ones showing the gate rejects a candidate violating each: a caption cannot instruct, cannot
carry a number that disagrees with its measurement, cannot carry a number it did not declare, and
cannot name a detector class the Fact Packet does not hold.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import hdsg_composed as comp  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402


DETECTOR_CLASSES = ("chair", "person", "door", "tv", "bed", "couch", "bicycle", "bottle")


def make_event(objects=(), depths=(3.13, 1.74, 1.16),
               statuses=("CLEAR", "CONSTRAINED", "CONSTRAINED")):
    root = Path(__file__).resolve().parents[1]
    sectors = hdsg.sectors_from_lane_state({"depths": list(depths), "status": list(statuses)})
    authority = hdsg.determine_authority(
        "FORWARD", "CAUTION", sectors,
        previous_selected_sector=None, sector_choice_tolerance_m=0.10,
    )
    fact_packet = hdsg.build_fact_packet(
        run_id="run_t", event_id="evt_1", observation_id="obs_1", ticket_id=None,
        timestamp_ms=1.0, intent="FORWARD", trigger_type="USER_REQUESTED",
        request_id="MORE_DETAIL", response_mode="MORE_DETAIL", previous_signature=None,
        objects=list(objects), sectors=sectors, authority=authority, mirror_view=False,
        detector_model="yolov8n.pt", detector_confidence=0.35,
        pipeline_config_path=root / "config" / "pipeline.yaml",
        ontology_path=root / "config" / "ontology.yaml",
        clear_threshold_m=2.0, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.10,
        motion_tracker=hdsg.MotionTracker(),
    )
    prompt_packet = hdsg.build_prompt_packet(
        fact_packet, prompt_id="prompt_1", model_id="m", model_hash=hdsg.sha256_text("m"),
        quantisation=None, temperature=0.2, top_p=0.9, max_tokens=400,
        system_prompt="s", constraint_hash=hdsg.sha256_text("g"),
    )
    return fact_packet, prompt_packet


def chair_object():
    return hdsg.normalise_objects([{
        "id": 0, "track_id": 3, "raw_label": "chair", "canonical_class": "chair",
        "ontology_class": "furniture", "conf": 0.8, "bbox_xyxy": [10, 10, 90, 200],
        "bearing": "LEFT", "distance_m": 1.62, "distance_method": "D455F_LOWER_BBOX_MEDIAN",
    }])


def candidate(caption, assertions=(), visuals=()):
    return {
        "schema_version": comp.CAPTION_SCHEMA,
        "caption": caption,
        "assertions": [dict(item) for item in assertions],
        "visual_observations": [dict(item) for item in visuals],
    }


def sector_assertion(name, value):
    return {"fact_id": f"sector:{name}",
            "measurement_id": f"m:sector:{name}:clearance",
            "stated_value": value}


def check(cand, fact_packet, prompt_packet, classes=DETECTOR_CLASSES):
    return comp.validate_caption_candidate(
        cand, prompt_packet, fact_packet, detector_classes=classes
    )


class AcceptanceTests(unittest.TestCase):
    def test_a_faithful_composed_caption_is_accepted(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate(
            "Ahead the space narrows to 1.74 metres, with 3.13 metres of room to the left "
            "and 1.16 metres to the right.",
            [sector_assertion("centre", 1.74), sector_assertion("left", 3.13),
             sector_assertion("right", 1.16)],
        )
        errors, scored = check(cand, fact_packet, prompt_packet)
        self.assertEqual(errors, [])
        self.assertEqual(comp.assertion_summary(scored)["agreement_rate"], 1.0)

    def test_rounding_within_tolerance_is_accepted(self):
        # A natural caption rounds. The tolerance exists to admit that without admitting invention.
        fact_packet, prompt_packet = make_event()
        cand = candidate("The way ahead narrows to about 1.7 metres.",
                         [sector_assertion("centre", 1.7)])
        errors, scored = check(cand, fact_packet, prompt_packet)
        self.assertEqual(errors, [])
        self.assertEqual(scored[0]["outcome"], "AGREES")
        self.assertAlmostEqual(scored[0]["absolute_error_m"], 0.04, places=3)

    def test_prose_naming_a_non_detector_word_is_allowed(self):
        # The check bounds what may be asserted about recognised objects, not ordinary language.
        fact_packet, prompt_packet = make_event()
        cand = candidate("A wall runs across the space 1.74 metres ahead.",
                         [sector_assertion("centre", 1.74)])
        self.assertEqual(check(cand, fact_packet, prompt_packet)[0], [])


class HallucinationDetectionTests(unittest.TestCase):
    """The four properties the design claims to hold by construction."""

    def test_a_declared_value_disagreeing_with_measurement_is_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("There is 0.40 metres of space ahead.",
                         [sector_assertion("centre", 0.40)])
        errors, scored = check(cand, fact_packet, prompt_packet)
        self.assertIn("RG_STATED_VALUE_MISMATCH", errors)
        self.assertEqual(scored[0]["outcome"], "DISAGREES")
        self.assertAlmostEqual(scored[0]["absolute_error_m"], 1.34, places=3)

    def test_an_undeclared_number_is_rejected(self):
        # Without this, an invented distance reaches the display simply by being left out of the
        # declarations.
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead is 1.74 metres, and a step lies 0.80 metres beyond it.",
                         [sector_assertion("centre", 1.74)])
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", check(cand, fact_packet, prompt_packet)[0])

    def test_a_caption_that_instructs_is_rejected(self):
        # This is the check that keeps the action categorical. The deterministic tuple is the
        # guidance; generated prose may describe but never direct.
        fact_packet, prompt_packet = make_event()
        for caption in ("There is 3.13 metres to the left, so take the left path.",
                        "Ahead narrows to 1.74 metres. Turn towards the left.",
                        "With 1.74 metres ahead you should slow down."):
            cand = candidate(caption, [sector_assertion("centre", 1.74),
                                       sector_assertion("left", 3.13)])
            self.assertIn("RG_ACTION_LANGUAGE_DETECTED",
                          check(cand, fact_packet, prompt_packet)[0], caption)

    def test_naming_a_detector_class_absent_from_the_packet_is_rejected(self):
        # Object hallucination becomes impossible rather than infrequent: a class the detector did
        # not report cannot appear in released text. Chapter 2 section 2.7.1.
        fact_packet, prompt_packet = make_event()
        cand = candidate("A person stands 1.74 metres ahead.",
                         [sector_assertion("centre", 1.74)])
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", check(cand, fact_packet, prompt_packet)[0])

    def test_the_plural_of_an_absent_class_is_also_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("Two chairs sit 1.74 metres ahead.", [sector_assertion("centre", 1.74)])
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", check(cand, fact_packet, prompt_packet)[0])

    def test_a_class_present_in_the_packet_may_be_named(self):
        fact_packet, prompt_packet = make_event(objects=chair_object())
        cand = candidate("A chair sits 1.62 metres to the left.",
                         [{"fact_id": "object:3", "measurement_id": "m:object:3:distance",
                           "stated_value": 1.62}])
        self.assertEqual(check(cand, fact_packet, prompt_packet)[0], [])


class ReferenceTests(unittest.TestCase):
    def test_an_unexposed_fact_is_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("An object sits 1.20 metres away.",
                         [{"fact_id": "object:99", "measurement_id": "m:object:99:distance",
                           "stated_value": 1.20}])
        self.assertIn("RG_FACT_REFERENCE_INVALID", check(cand, fact_packet, prompt_packet)[0])

    def test_an_unexposed_measurement_is_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("The rear is 1.74 metres away.",
                         [{"fact_id": "sector:centre", "measurement_id": "m:sector:rear:clearance",
                           "stated_value": 1.74}])
        self.assertIn("RG_MEASUREMENT_REFERENCE_INVALID",
                      check(cand, fact_packet, prompt_packet)[0])

    def test_measured_value_resolves_by_identifier(self):
        fact_packet, _ = make_event(objects=chair_object())
        self.assertAlmostEqual(
            comp.measured_value(fact_packet, "m:sector:centre:clearance"), 1.74, places=2)
        self.assertAlmostEqual(
            comp.measured_value(fact_packet, "m:object:3:distance"), 1.62, places=2)
        self.assertIsNone(comp.measured_value(fact_packet, "m:sector:rear:clearance"))
        self.assertIsNone(comp.measured_value(fact_packet, "nonsense"))


class CaptionNumberTests(unittest.TestCase):
    def test_an_observation_identifier_is_not_read_as_a_measurement(self):
        # A model copying "visual:1" into the prose is a formatting slip, not a claim about the
        # world, and rejecting the caption for it would inflate the fallback rate.
        self.assertEqual(comp._caption_numbers("Shadows are visible, visual:1."), [])

    def test_a_stated_distance_is_read(self):
        self.assertEqual(comp._caption_numbers("Ahead is 1.74 metres."), ["1.74"])

    def test_a_bare_integer_is_read(self):
        self.assertEqual(comp._caption_numbers("There are 3 metres of room."), ["3"])


class ReleaseTests(unittest.TestCase):
    def build(self, cand, errors, scored, fact_packet, prompt_packet):
        return comp.build_composed_release(
            fact_packet, prompt_packet, release_id="release_1", candidate=cand,
            scored_assertions=scored, failure_codes=errors,
        )

    def test_an_accepted_caption_becomes_the_reason_text(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead narrows to 1.74 metres.", [sector_assertion("centre", 1.74)])
        errors, scored = check(cand, fact_packet, prompt_packet)
        release = self.build(cand, errors, scored, fact_packet, prompt_packet)
        self.assertEqual(release["verification"]["release_mode"], "VLM_ACCEPTED")
        self.assertEqual(release["content"]["reason_text"], "Ahead narrows to 1.74 metres.")

    def test_the_action_text_is_deterministic_and_unaffected(self):
        # The whole point. Whatever the model wrote, the action the user acts on comes from the
        # rule engine.
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead narrows to 1.74 metres.", [sector_assertion("centre", 1.74)])
        errors, scored = check(cand, fact_packet, prompt_packet)
        accepted = self.build(cand, errors, scored, fact_packet, prompt_packet)
        rejected = self.build(None, ["RG_MODEL_UNAVAILABLE"], [], fact_packet, prompt_packet)
        self.assertEqual(accepted["content"]["action_text"], rejected["content"]["action_text"])
        self.assertEqual(accepted["authority"], rejected["authority"])

    def test_a_rejected_caption_falls_back_to_the_deterministic_rendering(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("There is 0.40 metres ahead.", [sector_assertion("centre", 0.40)])
        errors, scored = check(cand, fact_packet, prompt_packet)
        release = self.build(cand, errors, scored, fact_packet, prompt_packet)
        self.assertEqual(release["verification"]["release_mode"], "DETERMINISTIC_FALLBACK")
        self.assertIn("RG_STATED_VALUE_MISMATCH", release["verification"]["reason_codes"])
        self.assertNotIn("0.40", release["content"]["caption_text"])

    def test_the_measured_values_are_recorded_as_the_substitutions(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead narrows to 1.74 metres.", [sector_assertion("centre", 1.74)])
        errors, scored = check(cand, fact_packet, prompt_packet)
        release = self.build(cand, errors, scored, fact_packet, prompt_packet)
        substitutions = release["evidence"]["measurement_substitutions"]
        self.assertEqual(substitutions[0]["measurement_id"], "m:sector:centre:clearance")
        self.assertEqual(substitutions[0]["formatted_value"], "1.74 metres")

    def test_visual_observations_are_appended_and_recorded(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead narrows to 1.74 metres.", [sector_assertion("centre", 1.74)],
                         visuals=[{"candidate_observation_id": "visual:1",
                                   "proposed_label": "doorway", "bearing": "LEFT"}])
        errors, scored = check(cand, fact_packet, prompt_packet)
        release = self.build(cand, errors, scored, fact_packet, prompt_packet)
        self.assertEqual(errors, [])
        self.assertIn("Possible doorway is visible in the left.",
                      release["content"]["additional_detail_texts"])
        self.assertEqual(release["evidence"]["released_visual_observation_ids"], ["visual:1"])


class SchemaTests(unittest.TestCase):
    def test_a_reply_missing_a_field_is_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = {"schema_version": comp.CAPTION_SCHEMA, "caption": "Ahead is open."}
        self.assertIn("RG_SCHEMA_FAILURE", check(cand, fact_packet, prompt_packet)[0])

    def test_a_wrong_schema_version_is_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead is open.")
        cand["schema_version"] = "something.else"
        self.assertIn("RG_SCHEMA_FAILURE", check(cand, fact_packet, prompt_packet)[0])

    def test_non_json_is_a_parse_failure(self):
        self.assertEqual(comp.parse_caption_candidate("not json")[1], ["RG_PARSE_FAILURE"])

    def test_an_overlong_caption_is_rejected(self):
        fact_packet, prompt_packet = make_event()
        cand = candidate("Ahead is open. " * 60, [])
        self.assertIn("RG_PROFILE_LIMIT_EXCEEDED", check(cand, fact_packet, prompt_packet)[0])


class GrammarTests(unittest.TestCase):
    def test_no_rule_continues_onto_a_following_line(self):
        path = Path(__file__).resolve().parents[1] / "config" / "hdsg.vlm_caption.v1.gbnf"
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            self.assertFalse(line.lstrip().startswith("|"), f"line {number}")

    def test_the_two_identifier_fields_are_shaped_by_the_grammar(self):
        """Their shapes are fixed rather than observation-dependent, so the decoder enforces them.

        Left as free strings, the model put the measurement identifier into the fact_id field on
        two of three trials against Qwen3-VL-4B. The values it stated were correct and the
        candidate was discarded for a field mix-up, which is a rejection the design should not be
        spending.
        """
        path = Path(__file__).resolve().parents[1] / "config" / "hdsg.vlm_caption.v1.gbnf"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn('assertion-fact-field ::= "\\"fact_id\\"" ws ":" ws json-string', text)
        self.assertIn('fact-id ::= "sector:" sector-name', text)
        self.assertIn('measurement-id ::= "m:sector:" sector-name ":clearance"', text)

    def test_the_grammar_leaves_the_caption_free(self):
        # The design requires composition rather than selection, so the caption must not be
        # constrained to a fixed shape by the grammar.
        path = Path(__file__).resolve().parents[1] / "config" / "hdsg.vlm_caption.v1.gbnf"
        text = path.read_text(encoding="utf-8")
        self.assertIn('caption-field ::= "\\"caption\\"" ws ":" ws json-string', text)


class SummaryTests(unittest.TestCase):
    def test_only_comparable_assertions_enter_the_rate(self):
        scored = [
            {"outcome": "AGREES", "absolute_error_m": 0.01},
            {"outcome": "DISAGREES", "absolute_error_m": 1.30},
            {"outcome": "FACT_NOT_PERMITTED"},
        ]
        summary = comp.assertion_summary(scored)
        self.assertEqual(summary["declared"], 3)
        self.assertEqual(summary["comparable"], 2)
        self.assertEqual(summary["agreement_rate"], 0.5)
        self.assertEqual(summary["max_absolute_error_m"], 1.3)

    def test_no_comparable_assertions_reports_no_rate(self):
        self.assertIsNone(comp.assertion_summary([])["agreement_rate"])


if __name__ == "__main__":
    unittest.main()
