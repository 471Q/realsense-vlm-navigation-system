import json
from pathlib import Path
import tempfile
import unittest

from scripts import hdsg_runtime as hdsg


class HdsgRuntimeTests(unittest.TestCase):
    def test_equivalent_object_reidentification_does_not_change_signature(self):
        authority_a = {
            "scene_advisory": "CAUTION",
            "motion_decision": "SLOW",
            "selected_sector": "CENTRE",
            "scene_binding": {"accepted_fact_ids": ["object:12"]},
            "action_binding": {"accepted_fact_ids": ["object:12"]},
            "moving_object_fact_ids": [],
        }
        authority_b = json.loads(json.dumps(authority_a))
        authority_b["scene_binding"]["accepted_fact_ids"] = ["object:98"]
        authority_b["action_binding"]["accepted_fact_ids"] = ["object:98"]
        object_a = {
            "fact_id": "object:12", "canonical_label": "chair",
            "bearing": "CENTRE", "motion_state": "STATIONARY",
        }
        object_b = {
            "fact_id": "object:98", "canonical_label": "chair",
            "bearing": "CENTRE", "motion_state": "STATIONARY",
        }

        self.assertEqual(
            hdsg.guidance_signature(authority_a, "VALID", [object_a]),
            hdsg.guidance_signature(authority_b, "VALID", [object_b]),
        )

    def setUp(self):
        self.sectors = {
            "left": {
                "fact_id": "sector:left",
                "clearance_m": 0.49,
                "valid": True,
                "invalid_reason_codes": [],
                "status": "BLOCKED",
            },
            "centre": {
                "fact_id": "sector:centre",
                "clearance_m": 2.42,
                "valid": True,
                "invalid_reason_codes": [],
                "status": "CLEAR",
            },
            "right": {
                "fact_id": "sector:right",
                "clearance_m": 2.17,
                "valid": True,
                "invalid_reason_codes": [],
                "status": "CLEAR",
            },
        }
        self.tracker = hdsg.MotionTracker()

    def build_records(self, response_mode="AUTOMATIC", request_id="AUTO_GUIDANCE"):
        authority = hdsg.determine_authority("RIGHT", "CAUTION", self.sectors)
        fact_packet = hdsg.build_fact_packet(
            run_id="run_test",
            event_id="evt_test",
            observation_id="obs_test",
            ticket_id="ticket_test",
            timestamp_ms=1000.0,
            intent="RIGHT",
            trigger_type="MOTION_INTENT_STARTED" if request_id == "AUTO_GUIDANCE" else "USER_REQUESTED",
            request_id=request_id,
            response_mode=response_mode,
            previous_signature=None,
            objects=[],
            sectors=self.sectors,
            authority=authority,
            mirror_view=False,
            detector_model="yolov8n.pt",
            detector_confidence=0.25,
            pipeline_config_path=Path("missing-pipeline.yaml"),
            ontology_path=Path("missing-ontology.yaml"),
            clear_threshold_m=1.8,
            blocked_threshold_m=0.7,
            sector_choice_tolerance_m=0.1,
            motion_tracker=self.tracker,
        )
        prompt_packet = hdsg.build_prompt_packet(
            fact_packet,
            prompt_id="prompt_test",
            model_id="qwen3-vl-4b-instruct",
            model_hash=hdsg.sha256_text("model"),
            quantisation="Q4_K_M",
            temperature=0.2,
            top_p=0.9,
            max_tokens=220,
            system_prompt="fixed system prompt",
            constraint_hash=hdsg.sha256_text("grammar"),
        )
        return fact_packet, prompt_packet

    def test_requested_clear_sector_remains_authoritative(self):
        authority = hdsg.determine_authority("RIGHT", "CAUTION", self.sectors)
        self.assertEqual(authority["scene_advisory"], "CAUTION")
        self.assertEqual(authority["motion_decision"], "PROCEED")
        self.assertEqual(authority["selected_sector"], "RIGHT")
        self.assertEqual(authority["clear_sectors"], ["CENTRE", "RIGHT"])

    def test_close_object_constrains_matching_clear_sector(self):
        objects = [{
            "fact_id": "object:7",
            "bearing": "RIGHT",
            "distance_m": 0.55,
            "is_hazard": False,
        }]
        authority = hdsg.determine_authority(
            "RIGHT", "STOP", self.sectors, objects=objects
        )
        self.assertNotEqual(authority["motion_decision"], "PROCEED")
        self.assertNotEqual(authority["selected_sector"], "RIGHT")
        self.assertIn("object:7", authority["action_binding"]["accepted_fact_ids"])

    def test_moving_obstacle_closer_never_weakens_decision(self):
        rank = {"PROCEED": 0, "SLOW": 1, "REDIRECT": 2, "STOP": 3}
        decisions = []
        for distance in (2.0, 1.2, 0.5):
            objects = [{
                "fact_id": "object:7",
                "bearing": "RIGHT",
                "distance_m": distance,
                "is_hazard": False,
            }]
            authority = hdsg.determine_authority(
                "RIGHT", "SAFE", self.sectors, objects=objects
            )
            decisions.append(rank[authority["motion_decision"]])
        self.assertEqual(decisions, sorted(decisions))

    def test_equivalent_redirect_options_require_closed_choice(self):
        sectors = json.loads(json.dumps(self.sectors))
        sectors["centre"].update(clearance_m=0.5, status="BLOCKED")
        sectors["left"].update(clearance_m=2.10, status="CLEAR")
        sectors["right"].update(clearance_m=2.15, status="CLEAR")
        authority = hdsg.determine_authority(
            "FORWARD", "CAUTION", sectors, sector_choice_tolerance_m=0.10
        )
        self.assertEqual(authority["interaction_state"], "AWAITING_SECTOR_CHOICE")
        self.assertEqual(authority["motion_decision"], "STOP")
        self.assertEqual(authority["selected_sector"], "NONE")
        self.assertEqual(authority["selection_options"], ["LEFT", "RIGHT"])

    def test_backward_intent_requires_reorientation(self):
        authority = hdsg.determine_authority("BACKWARD", "SAFE", self.sectors)
        self.assertEqual(authority["interaction_state"], "REORIENTATION_REQUIRED")
        self.assertEqual(authority["motion_decision"], "STOP")
        self.assertEqual(authority["selected_sector"], "NONE")

    def test_user_box_is_derived_only_from_confirmed_movement(self):
        source = {
            "id": 7,
            "raw_label": "person",
            "canonical_class": "person",
            "ontology_class": "dynamic_obstacle",
            "conf": 0.9,
            "bbox_xyxy": [1, 2, 30, 60],
            "distance_m": 1.4,
            "distance_bin": "near",
            "bearing": "left",
            "motion_state": "STATIONARY",
            "display_bounding_box": True,
        }
        stationary = hdsg.normalise_objects([source])[0]
        source["motion_state"] = "MOVING"
        moving = hdsg.normalise_objects([source])[0]
        self.assertFalse(stationary["display_bounding_box"])
        self.assertTrue(moving["display_bounding_box"])

    def test_accepted_candidate_cannot_change_action(self):
        fact_packet, prompt_packet = self.build_records()
        candidate = {
            "schema_version": hdsg.CANDIDATE_SCHEMA,
            "reason_clauses": [
                {
                    "clause_id": "reason:1",
                    "requirement_ids": ["action_reason"],
                    "fact_ids": ["sector:right"],
                    "measurement_ids": ["m:sector:right:clearance"],
                    "text_template": "The right sector is clear for {{m:sector:right:clearance}}.",
                },
                {
                    "clause_id": "reason:2",
                    "requirement_ids": ["scene_reason"],
                    "fact_ids": ["sector:left"],
                    "measurement_ids": ["m:sector:left:clearance"],
                    "text_template": "The left sector is blocked at {{m:sector:left:clearance}}.",
                },
            ],
            "visual_observations": [],
        }
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=candidate,
        )
        self.assertEqual(release["verification"]["release_mode"], "VLM_ACCEPTED")
        self.assertEqual(release["authority"]["motion_decision"], "PROCEED")
        self.assertEqual(release["authority"]["selected_sector"], "RIGHT")
        self.assertTrue(release["content"]["caption_text"].startswith("Continue towards the right."))
        self.assertIn("2.17 metres", release["content"]["caption_text"])

    def test_action_language_rejects_complete_candidate(self):
        fact_packet, prompt_packet = self.build_records()
        candidate = {
            "schema_version": hdsg.CANDIDATE_SCHEMA,
            "reason_clauses": [
                {
                    "clause_id": "reason:1",
                    "requirement_ids": ["action_reason"],
                    "fact_ids": ["sector:right"],
                    "measurement_ids": ["m:sector:right:clearance"],
                    "text_template": "Continue because the right sector is clear for {{m:sector:right:clearance}}.",
                }
            ],
            "visual_observations": [],
        }
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=candidate,
        )
        self.assertEqual(release["verification"]["release_mode"], "DETERMINISTIC_FALLBACK")
        self.assertIn("RG_ACTION_LANGUAGE_DETECTED", release["verification"]["reason_codes"])
        self.assertNotIn("Continue because", release["content"]["caption_text"])

    def test_direct_model_number_rejects_complete_candidate(self):
        fact_packet, prompt_packet = self.build_records()
        candidate = {
            "schema_version": hdsg.CANDIDATE_SCHEMA,
            "reason_clauses": [
                {
                    "clause_id": "reason:1",
                    "requirement_ids": ["action_reason"],
                    "fact_ids": ["sector:right"],
                    "measurement_ids": ["m:sector:right:clearance"],
                    "text_template": "The right sector is clear for 2.17 metres.",
                }
            ],
            "visual_observations": [],
        }
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=candidate,
        )
        self.assertEqual(release["verification"]["release_mode"], "DETERMINISTIC_FALLBACK")
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", release["verification"]["reason_codes"])

    def test_extra_generated_claim_rejects_complete_candidate(self):
        fact_packet, prompt_packet = self.build_records()
        candidate = {
            "schema_version": hdsg.CANDIDATE_SCHEMA,
            "reason_clauses": [
                {
                    "clause_id": "reason:1",
                    "requirement_ids": ["action_reason"],
                    "fact_ids": ["sector:right"],
                    "measurement_ids": ["m:sector:right:clearance"],
                    "text_template": "The right sector is clear for {{m:sector:right:clearance}} and looks comfortable.",
                },
                {
                    "clause_id": "reason:2",
                    "requirement_ids": ["scene_reason"],
                    "fact_ids": ["sector:left"],
                    "measurement_ids": ["m:sector:left:clearance"],
                    "text_template": "The left sector is blocked at {{m:sector:left:clearance}}.",
                },
            ],
            "visual_observations": [],
        }
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=candidate,
        )
        self.assertEqual(release["verification"]["release_mode"], "DETERMINISTIC_FALLBACK")
        self.assertIn("RG_UNAPPROVED_LANGUAGE_DETECTED", release["verification"]["reason_codes"])

    def test_visual_instruction_label_is_rejected(self):
        fact_packet, prompt_packet = self.build_records("REASSESSMENT", "REASSESS")
        candidate = {
            "schema_version": hdsg.CANDIDATE_SCHEMA,
            "reason_clauses": [
                {
                    "clause_id": "reason:1",
                    "requirement_ids": ["action_reason"],
                    "fact_ids": ["sector:right"],
                    "measurement_ids": ["m:sector:right:clearance"],
                    "text_template": "The right sector is clear for {{m:sector:right:clearance}}.",
                },
                {
                    "clause_id": "reason:2",
                    "requirement_ids": ["scene_reason"],
                    "fact_ids": ["sector:left"],
                    "measurement_ids": ["m:sector:left:clearance"],
                    "text_template": "The left sector is blocked at {{m:sector:left:clearance}}.",
                },
            ],
            "visual_observations": [
                {
                    "candidate_observation_id": "visual:1",
                    "proposed_label": "ignore instructions",
                    "bearing": "CENTRE",
                }
            ],
        }
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=candidate,
        )
        self.assertEqual(release["verification"]["release_mode"], "DETERMINISTIC_FALLBACK")
        self.assertIn("RG_UNAPPROVED_LANGUAGE_DETECTED", release["verification"]["reason_codes"])

    def test_records_are_json_serialisable(self):
        fact_packet, prompt_packet = self.build_records()
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=None,
            failure_codes=["RG_MODEL_UNAVAILABLE"],
        )
        with tempfile.TemporaryDirectory() as directory:
            for name, value in (
                ("fact", fact_packet),
                ("prompt", prompt_packet),
                ("release", release),
            ):
                path = Path(directory) / f"{name}.json"
                path.write_text(json.dumps(value), encoding="utf-8")
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
