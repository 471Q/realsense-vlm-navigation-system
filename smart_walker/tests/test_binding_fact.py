"""Which measured element the release says caused the decision.

An output that names a real but non-causal fact is faithful and still misleading. Reporting a wall
at 1.2 metres when the walker stopped for a chair at 0.5 metres is entailed by the Fact Packet,
passes every check the gate applies to a measurement, and sends the person into the chair. So the
release records which element caused the behaviour, not only whether each statement is true.

A second implementation of this idea lived in the canonical client until 25 August 2026,
`identify_binding_fact` and five supporting pieces, 190 lines that nothing called. Its docstring
described the metric in the present tense and named an `advisory_sources` record that exists nowhere
in the tree. These tests exercise the implementation that runs, and hold the deleted one deleted.
"""

from __future__ import annotations

import unittest

import support
from support import ROOT, detected_object, event, sectors  # noqa: F401
from scripts import hdsg_composed as composed  # noqa: E402
from scripts import realsense_vlm_on_change_qwen as client  # noqa: E402


def evidence(**kwargs):
    packet, prompt = event(**kwargs)
    release = composed.build_composed_release(
        packet, prompt, release_id="release_1", candidate=None,
        scored_assertions=[], failure_codes=["RG_MODEL_UNAVAILABLE"])
    return release["evidence"], release["content"]["caption_text"]


class TheCauseIsNamed(unittest.TestCase):

    def test_a_near_object_is_named_rather_than_the_sector_around_it(self):
        """The case the argument is built on: the chair at 0.50 m is the cause, and a caption
        naming only the more distant sector would be true and would send the person into it."""
        found, text = evidence(objects=detected_object(3, "chair", "CENTRE", 0.5),
                               object_advisory="STOP",
                               lane=sectors(left=3.13, centre=1.2, right=1.16))
        self.assertEqual(["object:3"], found["scene_binding_fact_ids"])
        self.assertIn("object:3", found["action_binding_fact_ids"])
        self.assertIn("chair", text)
        self.assertIn("0.50 metres", text)

    def test_a_sector_is_named_when_no_object_caused_it(self):
        found, text = evidence(objects=[], object_advisory="SAFE",
                               lane=sectors(left=3.13, centre=0.5, right=1.16,
                                            statuses=("CLEAR", "BLOCKED", "CONSTRAINED")))
        self.assertEqual(["sector:centre"], found["scene_binding_fact_ids"])
        self.assertIn("centre", text)

    def test_the_release_always_carries_both_lists(self):
        """The frozen schema requires them, so a scene that binds nothing carries empty lists rather
        than absent keys."""
        found, _ = evidence(intent="NONE")
        for key in ("action_binding_fact_ids", "scene_binding_fact_ids"):
            self.assertIsInstance(found[key], list, key)


class TheSupersededImplementationIsGone(unittest.TestCase):
    """190 lines in the canonical client that nothing called.

    Held deleted rather than merely deleted, because the reason it survived so long is that its
    docstring asserted it was in use. A second implementation reintroduced beside the first is the
    fault removed from the perception layer on 24 August 2026.
    """

    def test_no_symbol_of_it_remains(self):
        for name in ("identify_binding_fact", "more_severe", "ADVISORY_SEVERITY",
                     "_SEVERITY_TO_ADVISORY", "BINDING_TIE_MARGIN_M", "_LANE_NAMES",
                     "_LANE_STATUS_SEVERITY"):
            self.assertFalse(hasattr(client, name), f"{name} is back")

    def test_the_record_it_claimed_to_write_never_existed(self):
        """Its docstring said the rules that fired are recorded in `advisory_sources` and that
        scoring consults them. Neither the record nor the reader was ever written.

        Comment lines are excluded, the note left in place of the deleted block naming the record in
        order to say it never existed.
        """
        for path in (ROOT / "scripts").glob("*.py"):
            code = [line for line in path.read_text(encoding="utf-8").splitlines()
                    if not line.strip().startswith("#")]
            self.assertNotIn("advisory_sources", "\n".join(code), path.name)

    def test_the_file_points_at_the_implementation_that_runs(self):
        """A reader looking for the causal element must find it in one step."""
        source = (ROOT / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(encoding="utf-8")
        self.assertIn("hdsg_runtime.determine_authority", source)
        self.assertIn("evidence.action_binding_fact_ids", source)

    def test_the_argument_for_the_metric_survived_the_deletion(self):
        """The one paragraph of the deleted docstring that was about the measurement rather than
        about the code. Kept beside the code that does the work."""
        from scripts import hdsg_runtime as hdsg

        source = (ROOT / "scripts" / "hdsg_runtime.py").read_text(encoding="utf-8")
        self.assertIn("faithful and still misleading", source)
        self.assertIn("sends the person into the chair", source)
        self.assertTrue(hasattr(hdsg, "determine_authority"))


if __name__ == "__main__":
    unittest.main()
