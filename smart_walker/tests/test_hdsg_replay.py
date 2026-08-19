"""End-to-end tests for the replay driver, with no camera and no model server.

The driver is the piece that turns a recording into the scored population behind Chapter 5's
tables, so the tests here run the whole path: write a recording and a telemetry archive, replay
them under both offline conditions with a scripted responder, and check the scored output.
"""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts import hdsg_baselines as baselines
from scripts import hdsg_recording as recording
from scripts import hdsg_replay as replay
from scripts import hdsg_runtime as hdsg


def _sectors(left, centre, right):
    def one(name, value, status):
        return {"fact_id": f"sector:{name}", "clearance_m": value, "valid": True,
                "invalid_reason_codes": [], "status": status}
    return {
        "left": one("left", *left),
        "centre": one("centre", *centre),
        "right": one("right", *right),
    }


class ReplayHarnessTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.recording_dir = self.root / "run_x"
        self.telemetry = self.root / "run_x.jsonl"
        self.colour = np.zeros((32, 48, 3), np.uint8)

        recorder = recording.ObservationRecorder(self.recording_dir).start()
        lines = []
        self.packets = []
        for index, layout in enumerate([
            ((0.49, "BLOCKED"), (2.42, "CLEAR"), (2.17, "CLEAR")),
            ((2.40, "CLEAR"), (0.55, "BLOCKED"), (2.30, "CLEAR")),
            ((1.50, "CONSTRAINED"), (1.60, "CONSTRAINED"), (2.50, "CLEAR")),
        ], start=1):
            observation_id = hdsg.next_identifier("obs", index)
            sectors = _sectors(*layout)
            authority = hdsg.determine_authority("FORWARD", "SAFE", sectors)
            packet = hdsg.build_fact_packet(
                run_id="run_x", event_id=hdsg.next_identifier("evt", index),
                observation_id=observation_id, ticket_id="tk", timestamp_ms=float(index),
                intent="FORWARD", trigger_type="MOTION_INTENT_STARTED",
                request_id="AUTO_GUIDANCE", response_mode="AUTOMATIC", previous_signature=None,
                objects=[], sectors=sectors, authority=authority, mirror_view=False,
                detector_model="yolov8n.pt", detector_confidence=0.25,
                pipeline_config_path=Path("x"), ontology_path=Path("y"),
                clear_threshold_m=1.8, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.1,
                motion_tracker=hdsg.MotionTracker(), recording_dir="run_x",
            )
            self.packets.append(packet)
            depth = np.full((32, 48), layout[1][0], np.float32)
            recorder.record(observation_id, self.colour, depth, float(index))
            lines.append(json.dumps({"record_type": "full_fact_packet", "record": packet}))
        self.assertTrue(recorder.stop()["complete"])
        self.telemetry.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def tearDown(self):
        self._temp.cleanup()

    def _responder(self, reply):
        def respond(prompt, colour):
            self.assertIsInstance(prompt, str)
            self.assertEqual(colour.shape, self.colour.shape)
            return reply if isinstance(reply, str) else json.dumps(reply)
        return respond

    def test_archive_and_recording_pair_by_observation_id(self):
        packets = replay.read_events(self.telemetry)
        self.assertEqual(len(packets), 3)
        events = list(replay.pair_with_recording(packets, self.recording_dir))
        self.assertEqual(len(events), 3)
        self.assertEqual([event.observation_id for event in events],
                         ["obs_000001", "obs_000002", "obs_000003"])

    def test_an_event_without_a_recorded_frame_is_skipped_not_substituted(self):
        """Scoring a condition against a different observation is not a matched comparison."""
        orphan = json.loads(json.dumps(self.packets[0]))
        orphan["identity"]["observation_id"] = "obs_999999"
        orphan["identity"]["event_id"] = "evt_999999"
        with self.telemetry.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"record_type": "full_fact_packet", "record": orphan}) + "\n")

        results = replay.replay_run(
            self.telemetry, self.recording_dir, self._responder({"reasons": []}),
            conditions=(baselines.CONDITION_C0,),
        )
        self.assertEqual(results["archived_events"], 4)
        self.assertEqual(results["replayed_events"], 3)
        self.assertEqual(results["skipped_without_frame"], 1)

    def test_both_conditions_run_over_the_same_events(self):
        results = replay.replay_run(
            self.telemetry, self.recording_dir,
            self._responder({
                "sector_distance_m": {"left": 0.5, "centre": 2.4, "right": 2.2},
                "recommended_action": "PROCEED", "recommended_sector": "CENTRE",
                "reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"}],
                "advisory_text": "Carry on.",
            }),
        )
        self.assertEqual(sorted(results["scored"]), sorted(baselines.OFFLINE_CONDITIONS))
        for condition in baselines.OFFLINE_CONDITIONS:
            self.assertEqual(len(results["scored"][condition]), 3)
            self.assertEqual(results["summary"][condition]["events"], 3)

    def test_c2_cannot_be_replayed_offline(self):
        with self.assertRaises(ValueError):
            replay.replay_run(self.telemetry, self.recording_dir,
                              self._responder({"reasons": []}),
                              conditions=(baselines.CONDITION_C2,))

    def test_a_responder_failure_is_unscoreable_not_a_zero_score(self):
        def broken(prompt, colour):
            raise RuntimeError("model unavailable")
        results = replay.replay_run(
            self.telemetry, self.recording_dir, broken, conditions=(baselines.CONDITION_C0,)
        )
        summary = results["summary"][baselines.CONDITION_C0]
        self.assertEqual(summary["parse_failures"], 3)
        self.assertIsNone(summary["binding_causal_reason"]["rate"])
        self.assertIsNone(summary["guidance_agreement"]["rate"])

    def test_prose_that_carries_no_structure_is_unscoreable(self):
        results = replay.replay_run(
            self.telemetry, self.recording_dir,
            self._responder("I would take it slowly if I were you."),
            conditions=(baselines.CONDITION_C0,),
        )
        summary = results["summary"][baselines.CONDITION_C0]
        self.assertEqual(summary["parse_failures"], 3)
        for entry in results["scored"][baselines.CONDITION_C0]:
            self.assertEqual(entry["cause_outcome"], "UNSCOREABLE")
            self.assertIsNotNone(entry["raw_response"])

    def test_the_raw_reply_is_retained_for_every_event(self):
        results = replay.replay_run(
            self.telemetry, self.recording_dir,
            self._responder({"reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "LEFT"}]}),
            conditions=(baselines.CONDITION_C1,),
        )
        for entry in results["scored"][baselines.CONDITION_C1]:
            self.assertIn("reasons", entry["raw_response"])

    def test_results_are_written_as_jsonl_with_a_final_summary(self):
        results = replay.replay_run(
            self.telemetry, self.recording_dir,
            self._responder({
                "sector_distance_m": {"left": 0.5, "centre": 2.4, "right": 2.2},
                "recommended_action": "PROCEED", "recommended_sector": "CENTRE",
                "reasons": [{"role": "ACTION", "kind": "SECTOR", "sector": "CENTRE"}],
                "advisory_text": "Carry on.",
            }),
        )
        path = replay.write_results(results, self.root / "scored" / "out.jsonl")
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(lines), 3 * len(baselines.OFFLINE_CONDITIONS) + 1)
        self.assertEqual(lines[-1]["record_type"], "baseline_summary")
        self.assertEqual(lines[-1]["record"]["replayed_events"], 3)
        self.assertTrue(all(item["record_type"] == "baseline_event" for item in lines[:-1]))

    def test_c1_is_given_the_measurements_and_c0_is_not(self):
        """The only difference between the conditions must be grounding."""
        seen: dict[str, str] = {}

        def capture(prompt, colour):
            seen[str(len(seen))] = prompt
            return json.dumps({"reasons": []})

        events = list(replay.pair_with_recording(
            replay.read_events(self.telemetry), self.recording_dir
        ))
        replay.replay_event(events[0], baselines.CONDITION_C0, capture)
        replay.replay_event(events[0], baselines.CONDITION_C1, capture)
        c0, c1 = seen["0"], seen["1"]
        self.assertNotIn("2.42", c0)
        self.assertIn("2.42", c1)


if __name__ == "__main__":
    unittest.main()
