"""Validates what the runtime emits against the frozen schemas.

`test_schema_freeze.py` checks that the frozen files are unaltered and that the fixtures pass the
executing validator. Neither establishes that the records the system actually produces conform,
and on 20 August 2026 four of them did not:

- the prompt packet carried `approved_text_templates` while the schema forbade extra properties,
  which had been true of every packet since v1;
- the release carried `RG_NO_INTENT_EXPRESSED` and `RG_GENERATION_PENDING`, absent from the reason
  code enumeration since the intent-triggered explanation policy introduced them;
- the release carried `IDLE_NO_INTENT` and `GENERATION_PENDING` interaction states, absent from
  their enumeration for the same reason;
- the idle release carried empty text where the schema required a minimum length.

All four survived because the fixtures were written by hand to match the schemas. A hand-written
fixture agrees with its schema by construction and proves nothing about the system. These tests
build records through the runtime and validate those, which is the check that was missing.
"""

import json
from pathlib import Path
import unittest

from scripts import hdsg_runtime as hdsg


SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"
FIXTURES = SCHEMA_ROOT / "fixtures"
CONFIG = Path(__file__).resolve().parents[1] / "config"


class RuntimeConformanceTests(unittest.TestCase):
    def setUp(self):
        try:
            from jsonschema import Draft7Validator
        except ImportError:  # pragma: no cover - only where the dependency is absent
            self.skipTest("jsonschema is not installed")
        self.validator_class = Draft7Validator

    def _schema(self, name):
        return json.loads((SCHEMA_ROOT / f"{name}.schema.json").read_text(encoding="utf-8"))

    def assertConforms(self, document, schema_name):
        errors = sorted(
            self.validator_class(self._schema(schema_name)).iter_errors(document),
            key=lambda error: list(error.path),
        )
        self.assertEqual(
            [], [f"{list(error.path)}: {error.message}" for error in errors],
            f"runtime output does not conform to {schema_name}",
        )

    def _event(self, expected_response_schema=None, **fact_kwargs):
        from scripts import hdsg_composed as composed
        sectors = hdsg.sectors_from_lane_state(
            {"depths": [3.13, 1.74, 0.55], "status": ["CLEAR", "CONSTRAINED", "BLOCKED"]}
        )
        authority = hdsg.determine_authority(
            "FORWARD", "CAUTION", sectors,
            previous_selected_sector=None, sector_choice_tolerance_m=0.10,
        )
        fact_packet = hdsg.build_fact_packet(
            run_id="run_t", event_id="evt_t", observation_id="obs_t", ticket_id=None,
            timestamp_ms=1.0, intent="FORWARD", trigger_type="USER_REQUESTED",
            request_id="MORE_DETAIL", response_mode="MORE_DETAIL", previous_signature=None,
            objects=[], sectors=sectors, authority=authority, mirror_view=False,
            detector_model="yolov8n.pt", detector_confidence=0.35,
            pipeline_config_path=CONFIG / "pipeline.yaml",
            ontology_path=CONFIG / "ontology.yaml",
            clear_threshold_m=2.0, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.10,
            motion_tracker=hdsg.MotionTracker(), **fact_kwargs,
        )
        extra = ({"expected_response_schema": expected_response_schema}
                 if expected_response_schema else {})
        prompt_packet = hdsg.build_prompt_packet(
            fact_packet, prompt_id="prompt_t", model_id="m", model_hash=hdsg.sha256_text("m"),
            quantisation=None, temperature=0.2, top_p=0.9, max_tokens=400,
            system_prompt="s", constraint_hash=hdsg.sha256_text("g"), **extra,
        )
        return composed, fact_packet, prompt_packet

    def _caption(self, value=1.74, caption="The way ahead narrows to 1.74 metres."):
        return {
            "schema_version": hdsg.CAPTION_SCHEMA,
            "caption": caption,
            "assertions": [{"fact_id": "sector:centre",
                            "measurement_id": "m:sector:centre:clearance",
                            "stated_value": value}],
            "visual_observations": [],
        }

    def test_a_typed_question_fact_packet_conforms(self):
        _, fact_packet, _ = self._event(input_method="TYPED_QUESTION", control_id="MORE_DETAIL")
        self.assertEqual(fact_packet["interaction"]["input_method"], "TYPED_QUESTION")
        self.assertConforms(fact_packet, "hdsg.fact_packet.v2")

    def test_a_prompt_packet_conforms_including_its_approved_templates(self):
        _, _, prompt_packet = self._event()
        self.assertTrue(any("approved_text_templates" in item
                            for item in prompt_packet["permitted_facts"]))
        self.assertConforms(prompt_packet, "hdsg.prompt_packet.v2")

    def test_the_prompt_packet_records_the_constraint_that_was_applied(self):
        # Pinned to the templated contract, the packet misreported which grammar constrained a
        # composed run, which is a provenance defect rather than a schema violation.
        _, _, templated = self._event()
        _, _, composed_packet = self._event(expected_response_schema=hdsg.CAPTION_SCHEMA)
        self.assertEqual(templated["expected_response_schema"], hdsg.CANDIDATE_SCHEMA)
        self.assertEqual(composed_packet["expected_response_schema"], hdsg.CAPTION_SCHEMA)
        self.assertEqual(composed_packet["generation"]["constraint_id"], hdsg.CAPTION_SCHEMA)
        self.assertConforms(composed_packet, "hdsg.prompt_packet.v2")

    def test_an_accepted_composed_release_conforms(self):
        composed, fact_packet, prompt_packet = self._event(
            expected_response_schema=hdsg.CAPTION_SCHEMA)
        candidate = self._caption()
        errors, scored = composed.validate_caption_candidate(
            candidate, prompt_packet, fact_packet, detector_classes=["person"]
        )
        self.assertEqual(errors, [])
        self.assertConforms(candidate, "hdsg.vlm_caption.v1")
        release = composed.build_composed_release(
            fact_packet, prompt_packet, release_id="release_t", candidate=candidate,
            scored_assertions=scored, failure_codes=errors,
        )
        self.assertEqual(release["verification"]["release_mode"], "VLM_ACCEPTED")
        self.assertConforms(release, "hdsg.release.v2")

    def test_a_release_rejected_for_a_wrong_value_conforms(self):
        """The new reason code must be admissible, or a detected hallucination is unrecordable."""
        composed, fact_packet, prompt_packet = self._event(
            expected_response_schema=hdsg.CAPTION_SCHEMA)
        candidate = self._caption(value=0.40, caption="There is 0.40 metres of space ahead.")
        errors, scored = composed.validate_caption_candidate(
            candidate, prompt_packet, fact_packet, detector_classes=["person"]
        )
        self.assertIn("RG_STATED_VALUE_MISMATCH", errors)
        release = composed.build_composed_release(
            fact_packet, prompt_packet, release_id="release_t", candidate=candidate,
            scored_assertions=scored, failure_codes=errors,
        )
        self.assertIn("RG_STATED_VALUE_MISMATCH", release["verification"]["reason_codes"])
        self.assertConforms(release, "hdsg.release.v2")

    def test_the_idle_and_pending_releases_conform(self):
        """Both states were introduced without reaching the enumeration they are recorded in."""
        _, fact_packet, prompt_packet = self._event()
        for label, kwargs in (("idle", {"no_intent": True}), ("pending", {"pending": True})):
            with self.subTest(state=label):
                release = hdsg.build_release(
                    fact_packet, prompt_packet, release_id="release_t",
                    candidate=None, failure_codes=[], **kwargs,
                )
                self.assertConforms(release, "hdsg.release.v2")

    def test_a_templated_release_still_conforms(self):
        """The comparison arm has to remain recordable under the same set."""
        _, fact_packet, prompt_packet = self._event()
        release = hdsg.build_release(
            fact_packet, prompt_packet, release_id="release_t",
            candidate=None, failure_codes=["RG_MODEL_UNAVAILABLE"],
        )
        self.assertConforms(release, "hdsg.release.v2")

    def test_every_reason_code_the_runtime_can_emit_is_admissible(self):
        frozen = set(self._schema("hdsg.release.v2")["definitions"]["reason_code"]["enum"])
        for code in hdsg.REASON_CODE_ORDER:
            self.assertIn(code, frozen, code)

    def test_every_valid_fixture_conforms(self):
        for path in sorted((FIXTURES / "valid").glob("*.json")):
            with self.subTest(fixture=path.name):
                self.assertConforms(json.loads(path.read_text(encoding="utf-8")), path.stem)

    def test_every_invalid_fixture_is_rejected(self):
        for path in sorted((FIXTURES / "invalid").glob("*.json")):
            with self.subTest(fixture=path.name):
                name = ".".join(path.name.split(".")[:3])
                errors = list(self.validator_class(self._schema(name)).iter_errors(
                    json.loads(path.read_text(encoding="utf-8"))))
                self.assertTrue(errors, f"{path.name} was accepted but is meant to be rejected")


if __name__ == "__main__":
    unittest.main()
