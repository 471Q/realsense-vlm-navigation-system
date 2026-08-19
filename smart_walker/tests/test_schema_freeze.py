"""Conformance tests for the frozen schema set.

Two failures motivated these. The first is that the v1 manifest recorded a digest for the GBNF
grammar that did not match the file the runtime loads, and nothing detected it: the check existed
only as a PowerShell script that had to be remembered. The second is that the runtime validator
and the frozen JSON Schema agreed with each other only by inspection, so an edit to either could
silently part them.

The manifest check here runs on every test invocation, and the fixture binding drives the frozen
fixtures through the executing validator rather than through a separate schema engine.

Full JSON Schema validation of all four record types remains the responsibility of
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
MANIFEST_PATH = SCHEMA_ROOT / "schema-manifest.v2.json"
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

    def test_manifest_covers_the_executing_grammar(self):
        """The recorded grammar must be the file the runtime actually loads.

        The v1 manifest hashed an archived copy while the runtime loaded another, which is how the
        two drifted apart unnoticed.
        """
        grammar_entries = [
            entry for entry in self.manifest["files"] if entry["path"].endswith(".gbnf")
        ]
        self.assertEqual(len(grammar_entries), 1)
        recorded = (SCHEMA_ROOT / grammar_entries[0]["path"]).resolve()
        runtime_default = (
            Path(__file__).resolve().parents[1] / "config" / "hdsg.vlm_candidate.v1.gbnf"
        ).resolve()
        self.assertEqual(recorded, runtime_default)

    def test_schema_set_version_is_recorded(self):
        self.assertEqual(self.manifest["schema_set_version"], "hdsg.schemas.v2")
        self.assertEqual(self.manifest["supersedes"], "hdsg.schemas.v1")

    def test_runtime_validator_accepts_the_valid_candidate_fixture(self):
        """Binds the executing validator to the frozen fixture, rather than to a second engine."""
        prompt_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.prompt_packet.v1.json").read_text(encoding="utf-8")
        )
        candidate = json.loads(
            (FIXTURES / "valid" / "hdsg.vlm_candidate.v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(hdsg.validate_candidate(candidate, prompt_packet), [])

    def test_runtime_validator_rejects_the_invalid_candidate_fixture(self):
        prompt_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.prompt_packet.v1.json").read_text(encoding="utf-8")
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
            (FIXTURES / "valid" / "hdsg.fact_packet.v1.json").read_text(encoding="utf-8")
        )
        prompt_packet = json.loads(
            (FIXTURES / "valid" / "hdsg.prompt_packet.v1.json").read_text(encoding="utf-8")
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
        self.assertEqual(len(found), 8, "expected four valid and four invalid fixtures")
        for path in found:
            with self.subTest(fixture=path.name):
                json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
