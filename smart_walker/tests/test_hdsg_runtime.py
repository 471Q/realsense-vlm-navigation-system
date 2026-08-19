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

    def test_two_blocked_sectors_redirect_to_the_remaining_clear_one(self):
        # The mirror case of the constrained-sector redirect above: two sectors are blocked
        # (severe, "red") and one is clear ("green"). Unlike the constrained case, this path was
        # unchanged by that revision, since a blocked or unknown intended sector has always gone
        # through the redirect-candidate logic. Locked in explicitly here because existing
        # coverage of it was only indirect, through object-induced blocking.
        sectors = json.loads(json.dumps(self.sectors))
        sectors["left"].update(clearance_m=0.40, status="BLOCKED")
        sectors["centre"].update(clearance_m=0.55, status="BLOCKED")
        sectors["right"].update(clearance_m=2.30, status="CLEAR")
        authority = hdsg.determine_authority("FORWARD", "SAFE", sectors)
        self.assertEqual(authority["motion_decision"], "REDIRECT")
        self.assertEqual(authority["selected_sector"], "RIGHT")
        self.assertEqual(authority["interaction_state"], "GUIDANCE_ACTIVE")
        self.assertEqual(
            set(authority["action_binding"]["accepted_fact_ids"]),
            {"sector:centre", "sector:right"},
        )

    def test_constrained_intended_sector_redirects_to_a_clear_alternative(self):
        # HDSG_DETERMINISTIC_ACTION_POLICY.md section 3, revised: a constrained intended sector
        # is still passable, but a fully clear alternative is preferred over continuing through
        # it. This is the exact configuration a live run reported: forward intent, centre
        # constrained by a nearby obstacle, right fully clear.
        sectors = json.loads(json.dumps(self.sectors))
        sectors["left"].update(clearance_m=1.78, status="CONSTRAINED")
        sectors["centre"].update(clearance_m=1.52, status="CONSTRAINED")
        sectors["right"].update(clearance_m=2.42, status="CLEAR")
        authority = hdsg.determine_authority("FORWARD", "SAFE", sectors)
        self.assertEqual(authority["motion_decision"], "REDIRECT")
        self.assertEqual(authority["selected_sector"], "RIGHT")
        self.assertEqual(authority["interaction_state"], "GUIDANCE_ACTIVE")
        self.assertEqual(
            set(authority["action_binding"]["accepted_fact_ids"]),
            {"sector:centre", "sector:right"},
        )

    def test_constrained_intended_sector_slows_with_no_clear_alternative(self):
        # Every sector is constrained or worse, so there is nothing to redirect to; the
        # passable-but-cautious default is retained.
        sectors = json.loads(json.dumps(self.sectors))
        sectors["left"].update(clearance_m=1.78, status="CONSTRAINED")
        sectors["centre"].update(clearance_m=1.52, status="CONSTRAINED")
        sectors["right"].update(clearance_m=1.60, status="CONSTRAINED")
        authority = hdsg.determine_authority("FORWARD", "SAFE", sectors)
        self.assertEqual(authority["motion_decision"], "SLOW")
        self.assertEqual(authority["selected_sector"], "CENTRE")
        self.assertEqual(authority["action_binding"]["accepted_fact_ids"], ["sector:centre"])

    def test_constrained_intended_sector_does_not_force_a_choice_on_a_tie(self):
        # Two equally clear alternatives exist, but the intended sector remains passable, so an
        # unresolved tie falls back to continuing cautiously rather than interrupting with
        # AWAITING_SECTOR_CHOICE, which is reserved for when the intended sector cannot be used
        # at all.
        sectors = json.loads(json.dumps(self.sectors))
        sectors["centre"].update(clearance_m=1.52, status="CONSTRAINED")
        sectors["left"].update(clearance_m=2.10, status="CLEAR")
        sectors["right"].update(clearance_m=2.15, status="CLEAR")
        authority = hdsg.determine_authority(
            "FORWARD", "SAFE", sectors, sector_choice_tolerance_m=0.10
        )
        self.assertEqual(authority["motion_decision"], "SLOW")
        self.assertEqual(authority["selected_sector"], "CENTRE")
        self.assertEqual(authority["interaction_state"], "GUIDANCE_ACTIVE")

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

    def test_no_intent_produces_empty_caption(self):
        fact_packet, prompt_packet = self.build_records()
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=None,
            no_intent=True,
        )
        self.assertEqual(release["authority"]["interaction_state"], "IDLE_NO_INTENT")
        self.assertEqual(release["content"]["caption_text"], "")
        self.assertEqual(release["content"]["action_text"], "")
        self.assertEqual(release["content"]["reason_text"], "")
        self.assertEqual(release["verification"]["reason_codes"], ["RG_NO_INTENT_EXPRESSED"])

    def test_pending_generation_shows_placeholder_reason(self):
        fact_packet, prompt_packet = self.build_records()
        self.assertEqual(fact_packet["deterministic"]["interaction_state"], "GUIDANCE_ACTIVE")
        release = hdsg.build_release(
            fact_packet,
            prompt_packet,
            release_id="release_test",
            candidate=None,
            pending=True,
        )
        self.assertEqual(release["authority"]["interaction_state"], "GENERATION_PENDING")
        self.assertTrue(release["content"]["caption_text"].startswith("Continue towards the right."))
        self.assertIn("Assessing the environment.", release["content"]["caption_text"])
        self.assertEqual(release["verification"]["reason_codes"], ["RG_GENERATION_PENDING"])
        # The action line is available immediately and matches the deterministic decision;
        # a pending generation never changes it.
        self.assertEqual(release["authority"]["motion_decision"], "PROCEED")
        self.assertEqual(release["authority"]["selected_sector"], "RIGHT")

    def test_pending_does_not_override_a_temporary_interaction_state(self):
        authority = hdsg.determine_authority("BACKWARD", "SAFE", self.sectors)
        fact_packet = hdsg.build_fact_packet(
            run_id="run_test",
            event_id="evt_test",
            observation_id="obs_test",
            ticket_id="ticket_test",
            timestamp_ms=1000.0,
            intent="BACKWARD",
            trigger_type="MOTION_INTENT_STARTED",
            request_id="AUTO_GUIDANCE",
            response_mode="AUTOMATIC",
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
        release = hdsg.build_release(
            fact_packet,
            {},
            release_id="release_test",
            candidate=None,
            pending=True,
        )
        # REORIENTATION_REQUIRED already carries a complete deterministic account per
        # HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md's interaction-state table, so a pending
        # generation must not replace it with the generic placeholder.
        self.assertEqual(release["authority"]["interaction_state"], "REORIENTATION_REQUIRED")
        self.assertNotIn("Assessing the environment.", release["content"]["caption_text"])
        self.assertIn("has not been observed", release["content"]["caption_text"])
        self.assertEqual(release["verification"]["reason_codes"], ["RG_GENERATION_PENDING"])

    def test_more_detail_requests_a_clause_per_scene_fact(self):
        # A stationary object well outside the safety-relevant range (beyond the caution
        # threshold, not a hazard) is not part of the action or scene binding, so it is a clean
        # probe of whether More detail names facts beyond the ones the decision itself required.
        raw_object = {
            "id": 9,
            "raw_label": "chair",
            "canonical_class": "chair",
            "ontology_class": "static_obstacle",
            "conf": 0.9,
            "bbox_xyxy": [1, 2, 30, 60],
            "distance_m": 1.8,
            "bearing": "left",
            "motion_state": "STATIONARY",
        }
        objects = hdsg.normalise_objects([raw_object])
        authority = hdsg.determine_authority("RIGHT", "CAUTION", self.sectors, objects=objects)
        fact_packet = hdsg.build_fact_packet(
            run_id="run_test",
            event_id="evt_test",
            observation_id="obs_test",
            ticket_id="ticket_test",
            timestamp_ms=1000.0,
            intent="RIGHT",
            trigger_type="USER_REQUESTED",
            request_id="MORE_DETAIL",
            response_mode="MORE_DETAIL",
            previous_signature=None,
            objects=objects,
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
        requirement_ids = {item["requirement_id"] for item in prompt_packet["requirements"]}
        self.assertIn("action_reason", requirement_ids)
        self.assertIn("scene_reason", requirement_ids)
        self.assertTrue(any(r.startswith("detail_sector_") for r in requirement_ids))
        self.assertIn("detail_object_9", requirement_ids)
        # The grammar allows at most four reason clauses; every requirement is single-fact so
        # each can be satisfied by exactly one clause.
        self.assertLessEqual(len(prompt_packet["requirements"]), 4)
        for item in prompt_packet["requirements"]:
            self.assertEqual(len(item["fact_ids"]), 1)
        self.assertEqual(prompt_packet["response_constraints"]["max_visual_observations"], 2)

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
                {
                    "clause_id": "reason:3",
                    "requirement_ids": [next(r for r in requirement_ids if r.startswith("detail_sector_"))],
                    "fact_ids": ["sector:centre"],
                    "measurement_ids": ["m:sector:centre:clearance"],
                    "text_template": "The centre sector is clear for {{m:sector:centre:clearance}}.",
                },
                {
                    "clause_id": "reason:4",
                    "requirement_ids": ["detail_object_9"],
                    "fact_ids": ["object:9"],
                    "measurement_ids": ["m:object:9:distance"],
                    "text_template": "The chair is present on the left at {{m:object:9:distance}}.",
                },
            ],
            "visual_observations": [],
        }
        release = hdsg.build_release(
            fact_packet, prompt_packet, release_id="release_test", candidate=candidate,
        )
        self.assertEqual(release["verification"]["release_mode"], "VLM_ACCEPTED")
        # action_reason renders as reason_text; the other three clauses (scene_reason, the
        # centre-sector detail, and the chair) all render into additional_detail_texts.
        self.assertEqual(len(release["content"]["additional_detail_texts"]), 3)
        self.assertIn("chair", release["content"]["caption_text"])

    def test_restriction_order_ranks_stop_highest(self):
        self.assertEqual(
            sorted(hdsg.RESTRICTION_ORDER, key=hdsg.RESTRICTION_ORDER.get),
            ["PROCEED", "SLOW", "REDIRECT", "STOP"],
        )

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
