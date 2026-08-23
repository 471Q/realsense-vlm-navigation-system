"""The manifest against the files it freezes.

The manifest records a digest of every schema, grammar and configuration file in the frozen set, so
that a record in the archive can be checked against the exact contract in force when it was written.
Its value rests entirely on the digests being current.

Nothing checked them until 23 August 2026. They were recomputed by hand after each edit, and nothing
failed when a recompute was forgotten: the caption grammar disagreed with its recorded digest for two
days and was found only by comparing the files by eye. A single afternoon of auditing produced eight
manual reissues.

These tests make a stale manifest a failing suite rather than a discovery.
"""

from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout, redirect_stderr

from support import SCHEMAS  # noqa: F401
from scripts import generate_manifest


def run(*arguments) -> int:
    """Runs the generator quietly and returns its exit status."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        return generate_manifest.main(list(arguments))


class ManifestTests(unittest.TestCase):

    def test_every_recorded_digest_matches_its_file(self):
        self.assertEqual(0, run("--check"),
                         "the schema manifest is out of step with the files it describes. "
                         "Run: python scripts/generate_manifest.py")

    def test_there_is_exactly_one_manifest(self):
        """Two manifests means two answers to which contract is frozen."""
        self.assertEqual(1, len(list(SCHEMAS.glob("schema-manifest.v*.json"))))

    def test_the_manifest_describes_every_frozen_file(self):
        """A file in the set that the manifest does not name is a file whose changes are unrecorded.

        The four configuration files are named explicitly because they are not schemas and would not
        be caught by a glob over the schemas directory. The two grammars decide what the model can
        emit and the catalogue decides what it is asked, so a change to any of them changes the
        contract as surely as a change to a schema does.
        """
        manifest = json.loads(generate_manifest.manifest_path().read_text(encoding="utf-8"))
        described = {entry["path"].replace("\\", "/").rsplit("/", 1)[-1]
                     for entry in generate_manifest.described_files(manifest)}
        expected = {path.name for path in SCHEMAS.glob("hdsg.*.schema.json")} | {
            "hdsg.vlm_caption.v1.gbnf",
            "hdsg.question_route.v1.gbnf",
            "hdsg_request_catalogue.v1.json",
        }
        self.assertEqual(set(), expected - described,
                         "a frozen file is not described by the manifest")

    def test_every_described_file_exists(self):
        """A manifest naming a file that is not there records the digest of nothing. It named
        schema-manifest.v4.json in a validation script for a day after the set moved to v5."""
        manifest = json.loads(generate_manifest.manifest_path().read_text(encoding="utf-8"))
        missing = [entry["path"] for entry in generate_manifest.described_files(manifest)
                   if not (SCHEMAS / entry["path"]).exists()]
        self.assertEqual([], missing)

    def test_a_changed_file_is_detected(self):
        """The property the whole file rests on. Verified by changing a described file rather than
        by trusting that the comparison happens."""
        target = SCHEMAS / "hdsg.question_route.v1.schema.json"
        original = target.read_bytes()
        try:
            target.write_bytes(original + b"\n")
            self.assertEqual(1, run("--check"))
        finally:
            target.write_bytes(original)
        self.assertEqual(0, run("--check"))


if __name__ == "__main__":
    unittest.main()
