"""Conformance tests for the frozen schema set.

Two failures motivated these. The first is that the v1 manifest recorded a digest for the GBNF
grammar that did not match the file the runtime loads, and nothing detected it: the check existed
only as a PowerShell script that had to be remembered. The second is that the runtime validator
and the frozen JSON Schema agreed with each other only by inspection, so an edit to either could
silently part them.

The manifest check here runs on every test invocation, and the fixture binding drives the frozen
fixtures through the executing validator rather than through a separate schema engine.

Full JSON Schema validation of all six record types remains the responsibility of
`validate_schema_fixtures.ps1` under PowerShell 7, since `Test-Json` is the schema engine the
freeze was recorded against. That script and these tests check different things and neither
replaces the other.
"""

import hashlib
import json
from pathlib import Path
import unittest

from scripts import hdsg_runtime as hdsg


SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"
MANIFEST_PATH = SCHEMA_ROOT / "schema-manifest.v3.json"
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
        self.assertEqual(self.manifest["schema_set_version"], "hdsg.schemas.v3")
        self.assertEqual(self.manifest["supersedes"], "hdsg.schemas.v2")

    def test_the_runtime_identifiers_match_the_frozen_set(self):
        """The identifiers the runtime stamps on records must be the ones the manifest froze.

        A record stamped with an identifier absent from the set is not covered by the freeze, and
        nothing else detects it: the fixtures validate, the tests pass, and the telemetry a run
        produces is unverifiable against its own contract.
        """
        recorded = {entry["record_schema"] for entry in self.manifest["files"]}
        for identifier in (hdsg.FACT_PACKET_SCHEMA, hdsg.PROMPT_PACKET_SCHEMA,
                           hdsg.CANDIDATE_SCHEMA, hdsg.CAPTION_SCHEMA, hdsg.RELEASE_SCHEMA):
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

    def test_runtime_validator_accepts_the_valid_candidate_fixture(self):
        """Binds the executing validator to the frozen fixture, rather than to a second engine."""
        prompt_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.prompt_packet.v2.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (FIXTURES / "valid" / "hdsg.vlm_candidate.v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(hdsg.validate_candidate(candidate, prompt_packet), [])

    def test_runtime_validator_rejects_the_invalid_candidate_fixture(self):
        prompt_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.prompt_packet.v2.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (FIXTURES / "invalid" / "hdsg.vlm_candidate.v1.extra_action.json").read_text(
                encoding="utf-8"
            )
        )
        errors = hdsg.validate_candidate(candidate, prompt_packet)
        self.assertIn("RG_SCHEMA_FAILURE", errors)

    def test_valid_candidate_fixture_renders_without_changing_the_action(self):
        """The frozen fixture must survive the whole release path, not only the validator."""
        fact_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.fact_packet.v2.json").read_text(encoding="utf-8")
        )
        prompt_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.prompt_packet.v2.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (FIXTURES / "valid" / "hdsg.vlm_candidate.v1.json").read_text(encoding="utf-8")
        )
        release = hdsg.build_release(
            fact_packet, prompt_packet, release_id="release_fixture", candidate=candidate,
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
        self.assertEqual(len(found), 12, "expected six valid and six invalid fixtures")
        for path in found:
            with self.subTest(fixture=path.name):
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
