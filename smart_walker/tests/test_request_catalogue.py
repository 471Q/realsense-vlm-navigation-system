"""The startup check over the request catalogue.

The catalogue holds every instruction the runtime sends to the model. It is read once at startup and
then subscripted directly on the live paths, so a block that is absent is a crash part way through a
session rather than a degraded caption.

The check existed before 23 August 2026 but covered only the three request profiles, having been
written when those three were the whole file. Two blocks were added afterwards and neither addition
extended it, so a catalogue missing the question-answering block started, ran, and raised KeyError on
the first typed question with the camera open. These tests cover the whole file, so a block added
later fails here unless the check is extended with it.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from support import CONFIG  # noqa: F401
from scripts.realsense_vlm_on_change_qwen import (
    REQUIRED_CATALOGUE_BLOCKS,
    REQUIRED_CATALOGUE_REQUESTS,
    load_request_catalogue,
)

SHIPPED = CONFIG / "hdsg_request_catalogue.v1.json"


def shipped_catalogue() -> dict:
    return json.loads(SHIPPED.read_text(encoding="utf-8"))


def load(catalogue: dict):
    """Writes a catalogue to a temporary file and loads it, as the runtime does."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "catalogue.json"
        path.write_text(json.dumps(catalogue), encoding="utf-8")
        return load_request_catalogue(path)


class ShippedCatalogueTests(unittest.TestCase):
    """The file the runtime actually loads."""

    def test_the_shipped_catalogue_passes_its_own_check(self):
        self.assertEqual("hdsg.request_catalogue.v1",
                         load_request_catalogue(SHIPPED)["catalogue_version"])

    def test_the_check_covers_every_block_the_runtime_reads(self):
        """The rule that would have caught the gap.

        A block present in the shipped file but named nowhere in the check is a block that can go
        missing without the check noticing, which is exactly how the two later blocks slipped past.
        `_note` fields are excluded because nothing reads them.
        """
        checked = {"catalogue_version", "requests", "composed_system_prompt",
                   "composed_system_prompt_id"} | set(REQUIRED_CATALOGUE_BLOCKS)
        present = {key for key in shipped_catalogue() if not key.startswith("_")}
        self.assertEqual(set(), present - checked,
                         "a catalogue block is read by the runtime but not checked at startup")


class RejectionTests(unittest.TestCase):
    """What the check refuses. Each case started a run successfully before 23 August 2026 unless
    marked otherwise."""

    def test_a_wrong_version_is_refused(self):
        catalogue = shipped_catalogue()
        catalogue["catalogue_version"] = "hdsg.request_catalogue.v2"
        with self.assertRaises(ValueError):
            load(catalogue)

    def test_an_absent_request_profile_is_refused(self):
        """Refused before the change as well. Kept so the original guarantee cannot be lost while
        the check is extended."""
        for name in REQUIRED_CATALOGUE_REQUESTS:
            with self.subTest(name):
                catalogue = shipped_catalogue()
                del catalogue["requests"][name]
                with self.assertRaises(ValueError):
                    load(catalogue)

    def test_a_request_profile_missing_its_instruction_is_refused(self):
        """Present but hollow. The key existed, so the membership test passed and the run started."""
        catalogue = shipped_catalogue()
        catalogue["requests"]["MORE_DETAIL"]["fixed_instruction"] = ""
        with self.assertRaises(ValueError):
            load(catalogue)

    def test_an_absent_later_block_is_refused(self):
        """The defect itself. Without these blocks the run began and failed on the first question."""
        for block in REQUIRED_CATALOGUE_BLOCKS:
            with self.subTest(block):
                catalogue = shipped_catalogue()
                del catalogue[block]
                with self.assertRaises(ValueError):
                    load(catalogue)

    def test_a_later_block_missing_a_field_is_refused(self):
        for block, fields in REQUIRED_CATALOGUE_BLOCKS.items():
            for field in fields:
                with self.subTest(f"{block}.{field}"):
                    catalogue = shipped_catalogue()
                    del catalogue[block][field]
                    with self.assertRaises(ValueError):
                        load(catalogue)

    def test_an_empty_instruction_is_refused_as_an_absent_one(self):
        """An empty string is a present key and a silent failure: the model receives a prompt with
        no instruction in it and answers whatever it likes."""
        catalogue = shipped_catalogue()
        catalogue["question_answer"]["fixed_instruction"] = "   "
        with self.assertRaises(ValueError):
            load(catalogue)

    def test_an_absent_composed_system_prompt_is_refused(self):
        """Read immediately after the check with a bare subscript, so its absence was a KeyError at
        startup rather than a described refusal."""
        for field in ("composed_system_prompt", "composed_system_prompt_id"):
            with self.subTest(field):
                catalogue = shipped_catalogue()
                catalogue[field] = ""
                with self.assertRaises(ValueError):
                    load(catalogue)

    def test_the_error_names_the_block_at_fault(self):
        """A refusal that does not say which block is missing sends a maintainer through a 36-line
        file by eye."""
        catalogue = shipped_catalogue()
        del catalogue["unconstrained_diagnostic"]
        with self.assertRaises(ValueError) as caught:
            load(catalogue)
        self.assertIn("unconstrained_diagnostic", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
