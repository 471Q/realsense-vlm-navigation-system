"""Conformance tests for the frozen schema set.

Two failures motivated these. The first is that the v1 manifest recorded a digest for the GBNF
grammar that did not match the file the runtime loads, and nothing detected it: the check existed
only as a PowerShell script that had to be remembered. The second is that the runtime validator
and the frozen JSON Schema agreed with each other only by inspection, so an edit to either could
silently part them.

The manifest check here runs on every test invocation, and the fixture binding drives the frozen
fixtures through the executing validator rather than through a separate schema engine.

Full JSON Schema validation of all five record types remains the responsibility of
`validate_schema_fixtures.ps1` under PowerShell 7, since `Test-Json` is the schema engine the
freeze was recorded against. That script and these tests check different things and neither
replaces the other.
"""

import hashlib
import json
from pathlib import Path
import unittest

from scripts import hdsg_composed as composed
from scripts import hdsg_runtime as hdsg


SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"
MANIFEST_PATH = SCHEMA_ROOT / "schema-manifest.v4.json"
FIXTURES = SCHEMA_ROOT / "fixtures"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class SchemaFreezeTests(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_digests_match_the_files_on_disk(self):
        """Fails if any frozen file is edited without the manifest being regenerated."""
        for entry in self.manifest["files"]:
            path = (SCHEMA_ROOT / entry["path"]).resolve()
            with self.subTest(path=entry["path"]):
                self.assertTrue(path.is_file(), f"frozen file is missing: {entry['path']}")
                self.assertEqual(
                    _sha256(path), entry["sha256"],
                    f"{entry['path']} does not match its recorded digest",
                )

    def test_manifest_covers_every_executing_grammar(self):
        """Each grammar the runtime loads must be the file the manifest hashed.

        The v1 manifest hashed an archived copy while the runtime loaded another, which is how the
        two drifted apart unnoticed. The v2 manifest then hashed one grammar while the runtime had
        acquired three, which left the routing and caption constraints free to drift the same way.
        """
        config = Path(__file__).resolve().parents[1] / "config"
        recorded = {
            (SCHEMA_ROOT / entry["path"]).resolve()
            for entry in self.manifest["files"] if entry["path"].endswith(".gbnf")
        }
        self.assertEqual(recorded, {path.resolve() for path in config.glob("*.gbnf")})

    def test_schema_set_version_is_recorded(self):
        self.assertEqual(self.manifest["schema_set_version"], "hdsg.schemas.v4")
        self.assertEqual(self.manifest["supersedes"], "hdsg.schemas.v3")

    def test_the_retired_templated_contract_is_absent_from_the_set(self):
        """No file in the set may describe a contract the runtime cannot produce.

        The templated candidate was removed on 21 August 2026. Its schema and grammar going with
        it is what keeps the set a description of the executing system rather than an accumulation
        of everything that was ever built.
        """
        config = Path(__file__).resolve().parents[1] / "config"
        self.assertFalse((SCHEMA_ROOT / "hdsg.vlm_candidate.v1.schema.json").exists())
        self.assertFalse((config / "hdsg.vlm_candidate.v1.gbnf").exists())
        self.assertNotIn("hdsg.vlm_candidate.v1",
                         {entry["record_schema"] for entry in self.manifest["files"]})
        self.assertFalse(hasattr(hdsg, "validate_candidate"))
        self.assertFalse(hasattr(hdsg, "parse_candidate"))
        self.assertFalse(hasattr(hdsg, "STATE_PREDICATES"))

    def test_the_retired_identifier_survives_in_the_enumerations(self):
        """Archived records carry it, so the frozen set must still validate them.

        Removing the value would make every release and prompt packet recorded before 21 August
        2026 non-conformant against the set that is meant to describe them, which would cost the
        archive its standing as evidence.
        """
        release = json.loads(
            (SCHEMA_ROOT / "hdsg.release.v2.schema.json").read_text(encoding="utf-8")
        )
        packet = json.loads(
            (SCHEMA_ROOT / "hdsg.prompt_packet.v2.schema.json").read_text(encoding="utf-8")
        )
        self.assertIn("hdsg.vlm_candidate.v1",
                      release["definitions"]["verification"]["properties"]
                             ["candidate_schema"]["enum"])
        self.assertIn("hdsg.vlm_candidate.v1", packet["properties"]
                      ["expected_response_schema"]["enum"])

    def test_the_runtime_identifiers_match_the_frozen_set(self):
        """The identifiers the runtime stamps on records must be the ones the manifest froze.

        A record stamped with an identifier absent from the set is not covered by the freeze, and
        nothing else detects it: the fixtures validate, the tests pass, and the telemetry a run
        produces is unverifiable against its own contract.
        """
        recorded = {entry["record_schema"] for entry in self.manifest["files"]}
        for identifier in (hdsg.FACT_PACKET_SCHEMA, hdsg.PROMPT_PACKET_SCHEMA,
                           hdsg.CAPTION_SCHEMA, hdsg.RELEASE_SCHEMA):
            self.assertIn(identifier, recorded, identifier)

    def test_the_new_reason_code_is_in_the_frozen_enumeration(self):
        """RG_STATED_VALUE_MISMATCH is the one code the composed design adds.

        Emitting a code the release schema rejects is how a composed run stopped conforming to its
        own contract under hdsg.schemas.v2.
        """
        release_schema = json.loads(
            (SCHEMA_ROOT / "hdsg.release.v2.schema.json").read_text(encoding="utf-8")
        )
        frozen = set(release_schema["definitions"]["reason_code"]["enum"])
        for code in hdsg.REASON_CODE_ORDER:
            self.assertIn(code, frozen, code)

    def _fixture(self, kind, name):
        return json.loads((FIXTURES / kind / name).read_text(encoding="utf-8"))

    def test_runtime_validator_accepts_the_valid_caption_fixture(self):
        """Binds the executing validator to the frozen fixture, rather than to a second engine."""
        errors, scored = composed.validate_caption_candidate(
            self._fixture("valid", "hdsg.vlm_caption.v1.json"),
            self._fixture("valid", "hdsg.prompt_packet.v2.json"),
            self._fixture("valid", "hdsg.fact_packet.v2.json"),
            detector_classes=["person"],
        )
        self.assertEqual(errors, [])
        self.assertTrue(scored, "an accepted caption declares at least one value")

    def test_runtime_validator_rejects_the_invalid_caption_fixture(self):
        errors, _ = composed.validate_caption_candidate(
            self._fixture("invalid", "hdsg.vlm_caption.v1.extra_action.json"),
            self._fixture("valid", "hdsg.prompt_packet.v2.json"),
            self._fixture("valid", "hdsg.fact_packet.v2.json"),
            detector_classes=["person"],
        )
        self.assertTrue(errors, "the fixture exists to be rejected")

    def test_valid_caption_fixture_renders_without_changing_the_action(self):
        """The frozen fixture must survive the whole release path, not only the validator."""
        fact_packet = self._fixture("valid", "hdsg.fact_packet.v2.json")
        prompt_packet = self._fixture("valid", "hdsg.prompt_packet.v2.json")
        candidate = self._fixture("valid", "hdsg.vlm_caption.v1.json")
        errors, scored = composed.validate_caption_candidate(
            candidate, prompt_packet, fact_packet, detector_classes=["person"]
        )
        release = composed.build_composed_release(
            fact_packet, prompt_packet, release_id="release_fixture", candidate=candidate,
            scored_assertions=scored, failure_codes=errors,
        )
        deterministic = fact_packet["deterministic"]
        self.assertEqual(release["verification"]["release_mode"], "VLM_ACCEPTED")
        self.assertEqual(
            release["authority"]["motion_decision"], deterministic["motion_decision"]
        )
        self.assertEqual(
            release["authority"]["selected_sector"], deterministic["selected_sector"]
        )

    def test_every_fixture_is_well_formed_json(self):
        found = sorted(FIXTURES.rglob("*.json"))
        self.assertEqual(len(found), 10, "expected five valid and five invalid fixtures")
        for path in found:
            with self.subTest(fixture=path.name):
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
