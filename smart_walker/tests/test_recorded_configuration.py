"""The record of how a run was configured, checked against what the run applied.

Chapter 5 is read off these records, so a number here that no code reads is a description of a run
that did not happen. This has been found twice in the same block: `binding_tie_margin` stated a tie
threshold of 0.15 that appeared only in the record, the schema and one fixture, and the two
transition timings were independent literals in two files that agreed only because nobody had
changed either.
"""

from __future__ import annotations

import ast
import unittest

import support
from support import event  # noqa: F401
from scripts import hdsg_runtime as hdsg  # noqa: E402


class TheTransitionTimingsHaveOneHome(unittest.TestCase):
    """HDSG_HYBRID_CAPTION_POLICY.md section 4 says the final persistence periods will be chosen
    during pilot testing. That is exactly when one of two copies moves and the other does not.
    """

    def test_the_record_states_the_constants(self):
        packet, _ = event()
        timing = packet["configuration"]["timing_ms"]
        self.assertEqual(hdsg.RESTRICTIVE_TRANSITION_PERSISTENCE_MS,
                         timing["restrictive_transition_persistence"])
        self.assertEqual(hdsg.RECOVERY_TRANSITION_PERSISTENCE_MS,
                         timing["recovery_transition_persistence"])

    def test_the_recorded_value_follows_the_argument(self):
        """The test above passes against a hardcoded literal too, since the literal matches the
        constant's current value. Only an explicit argument separates the two, and putting the
        literals back survived every check until this test existed.
        """
        packet = support.fact_packet()
        moved = hdsg.build_fact_packet(
            run_id="run_t", event_id="evt_1", observation_id="obs_1", ticket_id=None,
            timestamp_ms=1.0, intent="FORWARD", trigger_type="USER_REQUESTED",
            request_id=packet["interaction"]["request_id"], response_mode="MORE_DETAIL",
            previous_signature=None, objects=[], sectors=packet["sectors"],
            authority=packet["deterministic"], mirror_view=False,
            detector_model="yolov8n.pt", detector_confidence=0.35,
            pipeline_config_path=support.CONFIG / "pipeline.yaml",
            ontology_path=support.CONFIG / "ontology.yaml",
            clear_threshold_m=hdsg.SECTOR_CLEAR_AT_OR_ABOVE_M,
            blocked_threshold_m=hdsg.OBJECT_STOP_BELOW_M, sector_choice_tolerance_m=0.10,
            motion_tracker=hdsg.MotionTracker(),
            restrictive_transition_persistence_ms=275.0,
            recovery_transition_persistence_ms=825.0,
        )
        timing = moved["configuration"]["timing_ms"]
        self.assertEqual(275.0, timing["restrictive_transition_persistence"])
        self.assertEqual(825.0, timing["recovery_transition_persistence"])

    def test_escalation_is_faster_than_recovery(self):
        """HDSG_HYBRID_CAPTION_POLICY.md section 4. The values themselves are provisional and are
        deliberately not pinned here, since the same document says they will be chosen during pilot
        testing. The asymmetry is not provisional: a hazard must not wait as long as a recovery.
        """
        self.assertLess(hdsg.RESTRICTIVE_TRANSITION_PERSISTENCE_MS,
                        hdsg.RECOVERY_TRANSITION_PERSISTENCE_MS)

    def test_the_defaults_are_the_shared_constants(self):
        defaults = hdsg.build_fact_packet.__kwdefaults__ or {}
        self.assertEqual(hdsg.RESTRICTIVE_TRANSITION_PERSISTENCE_MS,
                         defaults["restrictive_transition_persistence_ms"])
        self.assertEqual(hdsg.RECOVERY_TRANSITION_PERSISTENCE_MS,
                         defaults["recovery_transition_persistence_ms"])

    def test_the_client_reads_the_same_constants(self):
        """Asserted against the source: the persistence choice sits inside the guidance loop and
        needs a camera and a model to reach.

        The client held `0.15 if current_rank > previous_rank else 0.50` in seconds while the record
        held 150 and 500 in milliseconds. Neither literal may return.
        """
        source = (support.ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(
            encoding="utf-8")
        self.assertIn("hdsg.RESTRICTIVE_TRANSITION_PERSISTENCE_MS", source)
        self.assertIn("hdsg.RECOVERY_TRANSITION_PERSISTENCE_MS", source)
        # Only the expression that assigns the persistence period is examined. Scanning every 0.15
        # and 0.5 in the file matched three OpenCV font scales, which is the same substring trap the
        # docstring scans in this suite already work around.
        assignment = next((node for node in ast.walk(ast.parse(source))
                           if isinstance(node, ast.Assign)
                           and any(isinstance(target, ast.Name) and target.id == "persistence"
                                   for target in node.targets)), None)
        self.assertIsNotNone(assignment, "the persistence assignment was renamed or removed")
        literals = [node.value for node in ast.walk(assignment)
                    if isinstance(node, ast.Constant) and isinstance(node.value, float)]
        self.assertEqual([1000.0], literals,
                         "the only number in the assignment is the millisecond conversion")


class TheThresholdsRecordedAreTheThresholdsApplied(unittest.TestCase):
    """Held from the `binding_tie_margin` finding: every number in the block must be one some line
    of code reads."""

    def test_every_recorded_threshold_matches_the_module_constant(self):
        packet, _ = event()
        thresholds = packet["configuration"]["thresholds_m"]
        self.assertEqual(hdsg.OBJECT_STOP_BELOW_M, thresholds["object_stop_below"])
        self.assertEqual(hdsg.OBJECT_CAUTION_BELOW_M, thresholds["object_caution_below"])
        self.assertEqual(hdsg.HAZARD_STOP_AT_OR_BELOW_M, thresholds["hazard_stop_at_or_below"])

    def test_no_tie_margin_returns(self):
        packet, _ = event()
        self.assertNotIn("binding_tie_margin", packet["configuration"]["thresholds_m"])
        self.assertNotIn("binding_tie_margin", packet["configuration"])


class TheCameraIsStatedRatherThanRead(unittest.TestCase):
    """A stated limitation on 25 August 2026, not a defect. The study runs on one D455f that is not
    changing, and the model name is a `const` in the schema so no other camera's record would
    validate. Held as a test so the choice is visible: the device identifier is a label for the
    study's camera and not the unit's serial number, so the archive does not attribute a run to a
    physical device.
    """

    def test_the_recorded_camera_is_the_one_the_schema_freezes(self):
        import json
        packet, _ = event()
        schema = json.loads((support.ROOT / "schemas" / "hdsg.fact_packet.v2.schema.json")
                            .read_text(encoding="utf-8"))
        frozen = schema["definitions"]["observation"]["properties"]["camera_model"]["const"]
        self.assertEqual(frozen, packet["observation"]["camera_model"])

    def test_the_device_identifier_is_a_label_and_not_a_serial(self):
        packet, _ = event()
        self.assertEqual("d455f_01", packet["observation"]["device_id"])


if __name__ == "__main__":
    unittest.main()
