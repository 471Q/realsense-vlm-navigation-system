"""Tests for the offline comparison conditions and the frozen explanation scorer.

The scorer decides what Chapter 5 Table 5-9 reports, so its edge cases are the substance here:
a reason that is true but not causal must not score as grounded, and wording the scorer cannot
resolve must not be given a favourable reading.
"""

import json
from pathlib import Path
import unittest

from scripts import hdsg_baselines as base
from scripts import hdsg_runtime as hdsg


class BaselineFixtureMixin:
    def build(self, intent="FORWARD", objects=None, sectors=None):
        sectors = sectors or {
            "left": {"fact_id": "sector:left", "clearance_m": 0.49, "valid": True,
                     "invalid_reason_codes": [], "status": "BLOCKED"},
            "centre": {"fact_id": "sector:centre", "clearance_m": 2.42, "valid": True,
                       "invalid_reason_codes": [], "status": "CLEAR"},
            "right": {"fact_id": "sector:right", "clearance_m": 2.17, "valid": True,
                      "invalid_reason_codes": [], "status": "CLEAR"},
        }
        objects = objects or []
        authority = hdsg.determine_authority(intent, "SAFE", sectors, objects=objects)
        return hdsg.build_fact_packet(
            run_id="run_t", event_id="evt_t", observation_id="obs_t", ticket_id="tk",
            timestamp_ms=0.0, intent=intent, trigger_type="MOTION_INTENT_STARTED",
            request_id="AUTO_GUIDANCE", response_mode="AUTOMATIC", previous_signature=None,
            objects=objects, sectors=sectors, authority=authority, mirror_view=False,
            detector_model="yolov8n.pt", detector_confidence=0.25,
            pipeline_config_path=Path("x"), ontology_path=Path("y"),
            clear_threshold_m=1.8, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.1,
            motion_tracker=hdsg.MotionTracker(),
        )


class PromptTests(BaselineFixtureMixin, unittest.TestCase):
    def test_c0_is_not_given_the_measurements_and_c1_is(self):
        packet = self.build()
        c0 = base.build_baseline_prompt(base.CONDITION_C0, packet)
        c1 = base.build_baseline_prompt(base.CONDITION_C1, packet)
        self.assertNotIn("2.42", c0)
        self.assertNotIn("depth camera on the walker", c0)
        self.assertIn("2.42", c1)
        self.assertIn("BLOCKED", c1)
        # The task and the requested response shape must be identical, so a difference in
        # results is attributable to grounding rather than to prompt design.
        self.assertIn("recommended_action", c0)
        self.assertIn("recommended_action", c1)
        self.assertIn("intends to move forward", c0)
        self.assertIn("intends to move forward", c1)

    def test_c2_is_not_an_offline_prompt(self):
        with self.assertRaises(ValueError):
            base.build_baseline_prompt(base.CONDITION_C2, self.build())


class ParseTests(unittest.TestCase):
    def test_plain_json_parses(self):
        value, failure = base.parse_baseline_response('{"recommended_action":"STOP"}')
        self.assertEqual(value, {"recommended_action": "STOP"})
        self.assertIsNone(failure)

    def test_json_wrapped_in_prose_still_parses(self):
        """The baseline is unconstrained, so refusing a reply wrapped in commentary would report
        a formatting artefact as a content failure."""
        value, failure = base.parse_baseline_response(
            'Sure! Here is my assessment:\n{"recommended_action":"SLOW"}\nHope that helps.'
        )
        self.assertEqual(value, {"recommended_action": "SLOW"})
        self.assertIsNone(failure)

    def test_prose_without_json_is_a_parse_failure(self):
        value, failure = base.parse_baseline_response("The path ahead looks a little tight.")
        self.assertIsNone(value)
        self.assertIsNotNone(failure)


class CauseScoringTests(BaselineFixtureMixin, unittest.TestCase):
    def test_naming_the_binding_sector_scores_binding(self):
        packet = self.build(intent="LEFT")
        self.assertIn("sector:left", packet["deterministic"]["action_binding"]["accepted_fact_ids"])
        score = base.score_cause({"reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "LEFT"}]}, packet)
        self.assertEqual(score.outcome, base.CauseOutcome.BINDING)

    def test_a_true_but_non_causal_sector_is_not_scored_as_grounded(self):
        """Reporting a real fact that did not cause the decision is the failure mode that a
        naive 'is it true?' scorer would pass. HDSG_EXPLANATION_BINDING_POLICY.md section 7
        requires it to be counted separately."""
        packet = self.build(intent="LEFT")
        score = base.score_cause({"reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "RIGHT"}]}, packet)
        self.assertEqual(score.outcome, base.CauseOutcome.UNRELATED_TRUE)
        self.assertNotEqual(score.outcome, base.CauseOutcome.BINDING)

    def test_an_object_absent_from_the_packet_is_unsupported(self):
        packet = self.build()
        score = base.score_cause(
            {"reasons": [{"role": "ACTION", "kind": "OBJECT", "object_label": "staircase", "sector": "CENTRE"}]}, packet
        )
        self.assertEqual(score.outcome, base.CauseOutcome.UNSUPPORTED)

    def test_a_detected_object_that_caused_the_decision_scores_binding(self):
        objects = hdsg.normalise_objects([{
            "id": 3, "raw_label": "chair", "canonical_class": "chair",
            "ontology_class": "static_obstacle", "conf": 0.9, "bbox_xyxy": [1, 2, 3, 4],
            "distance_m": 0.55, "bearing": "centre", "motion_state": "STATIONARY",
        }])
        packet = self.build(objects=objects)
        binding = packet["deterministic"]["action_binding"]["accepted_fact_ids"]
        self.assertTrue(any(item.startswith("object:") for item in binding), binding)
        score = base.score_cause(
            {"reasons": [{"role": "ACTION", "kind": "OBJECT", "object_label": "chair", "sector": "CENTRE"}]}, packet
        )
        self.assertEqual(score.outcome, base.CauseOutcome.BINDING)

    def test_no_reason_scores_silent(self):
        score = base.score_cause({"reasons": []}, self.build())
        self.assertEqual(score.outcome, base.CauseOutcome.SILENT)

    def test_an_unresolvable_cause_is_unscoreable_not_a_failure(self):
        """Section 5.6.1 requires unresolvable wording to be recorded as unscoreable rather than
        assigned a favourable, or an unfavourable, interpretation."""
        packet = self.build()
        self.assertEqual(
            base.score_cause({"reasons": [{"role": "ACTION", "kind": "VIBES"}]}, packet).outcome,
            base.CauseOutcome.UNSCOREABLE,
        )
        self.assertEqual(
            base.score_cause({}, packet).outcome, base.CauseOutcome.UNSCOREABLE
        )
        self.assertEqual(
            base.score_cause({"reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "sideways"}]}, packet).outcome,
            base.CauseOutcome.UNSCOREABLE,
        )


class SceneAndRedirectTests(BaselineFixtureMixin, unittest.TestCase):
    def _redirect_packet(self):
        sectors = {
            "left": {"fact_id": "sector:left", "clearance_m": 2.30, "valid": True,
                     "invalid_reason_codes": [], "status": "CLEAR"},
            "centre": {"fact_id": "sector:centre", "clearance_m": 0.55, "valid": True,
                       "invalid_reason_codes": [], "status": "BLOCKED"},
            "right": {"fact_id": "sector:right", "clearance_m": 0.50, "valid": True,
                      "invalid_reason_codes": [], "status": "BLOCKED"},
        }
        packet = self.build(intent="FORWARD", sectors=sectors)
        self.assertEqual(packet["deterministic"]["motion_decision"], "REDIRECT")
        return packet

    def test_a_redirect_naming_only_the_blockage_is_incomplete(self):
        """Explaining half a redirect tells the user to change direction without saying where to."""
        packet = self._redirect_packet()
        response = {"reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"}]}
        self.assertFalse(base.score_redirect_completeness(response, packet))

    def test_a_redirect_naming_both_halves_is_complete(self):
        packet = self._redirect_packet()
        response = {"reasons": [
            {"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"},
            {"role": "ALTERNATIVE", "kind": "SECTOR", "sector": "LEFT"},
        ]}
        self.assertTrue(base.score_redirect_completeness(response, packet))

    def test_a_non_redirect_stays_out_of_the_denominator(self):
        self.assertIsNone(base.score_redirect_completeness({"reasons": []}, self.build()))

    def test_an_event_not_requiring_a_scene_reason_stays_out_of_the_denominator(self):
        """Events that never needed a scene reason are excluded rather than counted as passes."""
        packet = self.build(intent="LEFT")
        self.assertFalse(packet["deterministic"]["scene_fact_required"])
        self.assertIsNone(base.score_scene_reason({"reasons": []}, packet))

    def test_a_missing_but_required_scene_reason_is_silent(self):
        sectors = {
            "left": {"fact_id": "sector:left", "clearance_m": 0.49, "valid": True,
                     "invalid_reason_codes": [], "status": "BLOCKED"},
            "centre": {"fact_id": "sector:centre", "clearance_m": 2.42, "valid": True,
                       "invalid_reason_codes": [], "status": "CLEAR"},
            "right": {"fact_id": "sector:right", "clearance_m": 2.17, "valid": True,
                      "invalid_reason_codes": [], "status": "CLEAR"},
        }
        packet = self.build(intent="FORWARD", sectors=sectors)
        self.assertTrue(packet["deterministic"]["scene_fact_required"])
        score = base.score_scene_reason(
            {"reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"}]}, packet
        )
        self.assertEqual(score.outcome, base.CauseOutcome.SILENT)

    def test_unsupported_detection_covers_every_role_not_just_the_action(self):
        packet = self.build()
        self.assertTrue(base.has_unsupported_reason({"reasons": [
            {"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"},
            {"role": "SCENE", "kind": "OBJECT", "object_label": "escalator", "sector": "LEFT"},
        ]}, packet))
        self.assertFalse(base.has_unsupported_reason({"reasons": [
            {"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"},
        ]}, packet))


class SummaryTests(BaselineFixtureMixin, unittest.TestCase):
    def test_denominators_differ_per_measure_and_exclude_unscoreables(self):
        """Each row carries its own denominator: a single event count would misstate every row
        except the first two, and an unscoreable event must leave the denominator entirely."""
        packet = self.build(intent="LEFT")
        good = base.score_baseline_event(base.CONDITION_C1, {
            "sector_distance_m": {"left": 0.5, "centre": 2.4, "right": 2.2},
            "recommended_action": packet["deterministic"]["motion_decision"],
            "recommended_sector": packet["deterministic"]["selected_sector"],
            "reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "LEFT"}],
            "advisory_text": "ok",
        }, packet)
        unreadable = base.score_baseline_event(base.CONDITION_C1, {
            "reasons": [{"role": "ACTION", "kind": "VIBES"}],
        }, packet)
        failed = base.score_baseline_event(
            base.CONDITION_C1, None, packet, parse_failure="no JSON object found"
        )

        summary = base.summarise_condition([good, unreadable, failed])
        self.assertEqual(summary["events"], 3)
        self.assertEqual(summary["parse_failures"], 1)
        # The unreadable cause leaves the binding denominator rather than counting against it.
        self.assertEqual(summary["binding_causal_reason"]["d"], 1)
        self.assertEqual(summary["binding_causal_reason"]["n"], 1)
        self.assertEqual(summary["binding_causal_reason"]["unscoreable"], 1)
        # Only the complete event stated an action the oracle vocabulary recognises.
        self.assertEqual(summary["guidance_agreement"]["d"], 1)
        self.assertEqual(summary["distance"]["compared"], 3)

    def test_an_all_unscoreable_condition_reports_no_rate_rather_than_zero(self):
        """A rate of zero would read as a measured failure; None reports that nothing was scored."""
        packet = self.build()
        failed = [base.score_baseline_event(base.CONDITION_C0, None, packet, "empty response")
                  for _ in range(4)]
        summary = base.summarise_condition(failed)
        self.assertEqual(summary["parse_failures"], 4)
        self.assertIsNone(summary["binding_causal_reason"]["rate"])
        self.assertIsNone(summary["guidance_agreement"]["rate"])
        self.assertIsNone(summary["distance"]["within_factor_two_rate"])


class DistanceAndDerivedActionTests(BaselineFixtureMixin, unittest.TestCase):
    def test_distance_error_and_factor_of_two_are_reported(self):
        packet = self.build()
        result = base.score_distance_estimates(
            {"sector_distance_m": {"left": 0.50, "centre": 2.40, "right": 8.00}}, packet
        )
        self.assertEqual(result["compared"], 3)
        self.assertTrue(result["per_sector"]["left"]["within_factor_two"])
        self.assertTrue(result["per_sector"]["centre"]["within_factor_two"])
        self.assertFalse(result["per_sector"]["right"]["within_factor_two"])
        self.assertEqual(result["within_factor_two"], 2)

    def test_a_missing_estimate_is_not_counted_as_an_error(self):
        packet = self.build()
        result = base.score_distance_estimates(
            {"sector_distance_m": {"left": None, "centre": 2.40, "right": 2.20}}, packet
        )
        self.assertEqual(result["compared"], 2)
        self.assertFalse(result["per_sector"]["left"]["comparable"])

    def test_trusting_a_bad_estimate_would_have_changed_the_decision(self):
        """The concrete safety statement this measure exists to support: had the walker used the
        model's numbers instead of the depth camera, it would have decided differently."""
        packet = self.build()
        self.assertEqual(packet["deterministic"]["motion_decision"], "PROCEED")

        # The model badly overestimates a near obstruction directly ahead.
        derived = base.derive_action_from_estimates(
            {"sector_distance_m": {"left": 0.50, "centre": 0.40, "right": 2.20}}, packet
        )
        self.assertNotEqual(derived["motion_decision"], "PROCEED")

        # An accurate estimate reproduces the measured decision.
        faithful = base.derive_action_from_estimates(
            {"sector_distance_m": {"left": 0.49, "centre": 2.42, "right": 2.17}}, packet
        )
        self.assertEqual(faithful["motion_decision"], packet["deterministic"]["motion_decision"])
        self.assertEqual(faithful["selected_sector"], packet["deterministic"]["selected_sector"])


class EventScoringTests(BaselineFixtureMixin, unittest.TestCase):
    def test_a_complete_event_scores_every_measure(self):
        packet = self.build(intent="LEFT")
        response = {
            "sector_distance_m": {"left": 0.5, "centre": 2.4, "right": 2.2},
            "recommended_action": packet["deterministic"]["motion_decision"],
            "recommended_sector": packet["deterministic"]["selected_sector"],
            "reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "LEFT", "object_label": None}],
            "advisory_text": "The way to your left is blocked, so go straight on instead.",
        }
        scored = base.score_baseline_event(base.CONDITION_C1, response, packet)
        self.assertTrue(scored["scoreable"])
        self.assertEqual(scored["cause_outcome"], "BINDING")
        self.assertTrue(scored["action_agrees"])
        self.assertTrue(scored["derived_action"]["agrees_with_measured"])
        self.assertEqual(scored["distance"]["compared"], 3)
        json.dumps(scored)

    def test_a_missing_response_is_recorded_as_unscoreable(self):
        scored = base.score_baseline_event(
            base.CONDITION_C0, None, self.build(), parse_failure="no JSON object found"
        )
        self.assertFalse(scored["scoreable"])
        self.assertEqual(scored["cause_outcome"], "UNSCOREABLE")
        self.assertIsNone(scored["action_agrees"])
        self.assertEqual(scored["parse_failure"], "no JSON object found")

    def test_an_unrecognised_action_is_unscoreable_rather_than_disagreement(self):
        packet = self.build()
        self.assertIsNone(base.score_action_agreement(
            {"recommended_action": "EASE_OFF", "recommended_sector": "CENTRE"}, packet
        ))


if __name__ == "__main__":
    unittest.main()
