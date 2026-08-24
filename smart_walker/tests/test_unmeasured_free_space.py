"""What the walker says when the depth strips return nothing usable.

Two measurements share the word. An object's distance comes from the depth inside its own bounding
box, where a chair's texture gives stereo something to work with. A sector clearance comes from the
depth across a third of the view, where a blank wall or a washed-out floor gives it nothing. The two
fail independently, so the walker can know where a chair is and not know how much room surrounds it.

Saying "no reliable measurement" beside "a chair is 1.20 metres away" reads as a contradiction, and
until 25 August 2026 the walker said exactly that. Three faults met in this state, and it is not a
rare one: 114 of the 1,480 frames of the archived run leave no strip measurable.
"""

from __future__ import annotations

import unittest

import support
from support import detected_object, event, sectors  # noqa: F401
from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_questions as questions  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

NO_STRIPS = sectors(left=None, centre=None, right=None,
                    statuses=("UNKNOWN", "UNKNOWN", "UNKNOWN"))
ONE_STRIP_FAILED = sectors(left=3.13, centre=None, right=1.16,
                           statuses=("CLEAR", "UNKNOWN", "CONSTRAINED"))
CHAIR = detected_object(3, "chair", "LEFT", 1.2)


def scene(lane, objects=()):
    packet, prompt = event(objects=objects, object_advisory="CAUTION", lane=lane)
    release = composed.build_composed_release(
        packet, prompt, release_id="release_1", candidate=None,
        scored_assertions=[], failure_codes=["RG_MODEL_UNAVAILABLE"])
    answer = questions.with_action_prefix(
        release, questions.deterministic_answer(packet, prompt["requirements"]))
    return packet, prompt, release["content"]["caption_text"], answer


class TheQuestionIsNotDeclinedWhileSomethingIsMeasured(unittest.TestCase):
    """The pre-check read the three strips alone.

    A scene with no usable strip and a chair at 1.20 m was declined, and the person was told there
    was no reliable measurement of the area while the packet held the chair's distance.
    """

    def answerable(self, lane, objects):
        packet, _, _, _ = scene(lane, objects)
        return questions.measurement_is_answerable(
            packet["sectors"], 100.0, 2000.0, packet["objects"])

    def test_a_measured_object_makes_a_scene_answerable(self):
        self.assertTrue(self.answerable(NO_STRIPS, CHAIR))

    def test_a_scene_with_nothing_measured_at_all_is_still_declined(self):
        self.assertFalse(self.answerable(NO_STRIPS, []))

    def test_a_measured_strip_is_still_enough_on_its_own(self):
        self.assertTrue(self.answerable(ONE_STRIP_FAILED, []))

    def test_a_stale_observation_is_declined_however_much_was_measured(self):
        """Freshness is a separate condition and the change must not weaken it."""
        packet, _, _, _ = scene(ONE_STRIP_FAILED, CHAIR)
        self.assertFalse(questions.measurement_is_answerable(
            packet["sectors"], 5000.0, 2000.0, packet["objects"]))

    def test_an_object_with_no_distance_does_not_count_as_measured(self):
        objects = hdsg.normalise_objects([{"id": 9, "raw_label": "chair", "canonical_class": "chair",
                                           "distance_m": None, "bearing": "LEFT", "conf": 0.9}])
        self.assertFalse(questions.measurement_is_answerable(
            {name: {"valid": False} for name in ("left", "centre", "right")},
            100.0, 2000.0, objects))


class AnUnmeasurableStripDoesNotDisplaceAMeasuredObject(unittest.TestCase):
    """The clause budget is four and the facts were added sectors first.

    With no usable strip, the `no_clear_sector` condition took one slot and the three sectors took
    the rest, so the prompt offered four facts, every one of them the absence of a measurement, and
    the chair was not among them. Skipping a sector that measured nothing settles nothing that is
    deferred at `HDSG_VERIFIED_GENERATION_POLICY.md` section 10.4: four is unchanged and so is the
    order facts are added in.
    """

    def permitted(self, lane, objects=()):
        _, prompt, _, _ = scene(lane, objects)
        return [item["fact_id"] for item in prompt["permitted_facts"]]

    def test_the_chair_is_offered_when_no_strip_could_be_measured(self):
        self.assertIn("object:3", self.permitted(NO_STRIPS, CHAIR))

    def test_an_unmeasurable_sector_earns_no_clause(self):
        self.assertEqual(["condition:no_clear_sector", "object:3"], self.permitted(NO_STRIPS, CHAIR))

    def test_a_measured_sector_still_earns_one(self):
        permitted = self.permitted(ONE_STRIP_FAILED, [])
        self.assertIn("sector:left", permitted)
        self.assertIn("sector:right", permitted)

    def test_an_unmeasurable_sector_still_appears_when_it_bound_the_decision(self):
        """The skip applies to the detail slots and not to the action binding. The intent here is
        forward and the centre strip failed, which is precisely why the walker is not going forward,
        so it has to be able to say so. What the skip removes is a second mention of the same
        sector, added again as detail, in a slot a measured fact could have used.
        """
        packet, prompt, _, _ = scene(ONE_STRIP_FAILED, [])
        self.assertIn("sector:centre", [item["fact_id"] for item in prompt["permitted_facts"]])
        self.assertIn("sector:centre",
                      packet["deterministic"]["action_binding"]["accepted_fact_ids"])
        detail = [item["requirement_id"] for item in prompt["requirements"]]
        self.assertNotIn("detail_sector_1", detail)

    def test_an_ordinary_scene_is_unchanged(self):
        self.assertEqual(["sector:centre", "sector:left", "sector:right", "object:3"],
                         self.permitted(sectors(left=3.13, centre=1.74, right=1.16),
                                        detected_object(3, "chair", "LEFT", 1.62)))


class TheSentenceSaysWhichMeasurementIsMissing(unittest.TestCase):
    """Wording chosen by Atiq on 25 August 2026: the object is known, the free space is not.

    "free space" is the person's word here and not the code's. `free_space` in `pipeline.yaml` is
    the unimplemented corridor-width measurement, a different thing, and no sentence here is a claim
    about it.
    """

    def test_the_answer_names_the_object_before_the_absence(self):
        _, _, _, answer = scene(NO_STRIPS, CHAIR)
        self.assertEqual(
            "Stop. A chair is detected on the left at 1.20 metres. "
            "I cannot measure the free space around me.", answer)

    def test_a_single_failed_strip_is_named_as_a_sector(self):
        """"the free space on the centre" does not read, so the sector is named as a sector, in the
        same shape as "The centre sector is clear for 1.74 metres" beside it."""
        _, _, _, answer = scene(ONE_STRIP_FAILED, [])
        self.assertIn("I cannot measure the free space in the centre sector.", answer)
        self.assertNotIn("on the centre", answer)

    def test_the_guidance_caption_uses_the_same_words(self):
        _, _, caption_text, _ = scene(NO_STRIPS, CHAIR)
        self.assertEqual("Stop. I cannot measure the free space around me.", caption_text)

    def test_the_decline_uses_the_same_words(self):
        self.assertIn("I cannot measure the free space around me",
                      questions.NO_MEASUREMENT_TEXT)

    def test_no_sentence_still_calls_it_an_unreliable_measurement(self):
        """The wording that read as a contradiction, held gone across the files that produced it.

        Only string literals the code can emit are examined. Scanning the whole source matched the
        docstrings that record the old wording in order to explain why it changed, which is the
        `.docx` substring trap in a Python file and has caught this audit twice already.
        """
        import ast

        for name in ("hdsg_runtime.py", "hdsg_questions.py"):
            tree = ast.parse((support.ROOT / "scripts" / name).read_text(encoding="utf-8"))
            documented = {id(node.body[0].value) for node in ast.walk(tree)
                          if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
                          and node.body and isinstance(node.body[0], ast.Expr)
                          and isinstance(node.body[0].value, ast.Constant)
                          and isinstance(node.body[0].value.value, str)}
            emitted = [node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Constant) and isinstance(node.value, str)
                       and id(node) not in documented]
            for phrase in ("no reliable measurement", "Reliable depth measurements",
                           "reliable measurement of the area"):
                for literal in emitted:
                    self.assertNotIn(phrase, literal, f"{name}: {phrase}")

    def test_the_sentence_has_one_home(self):
        self.assertEqual("I cannot measure the free space around me.", hdsg.no_free_space_text())
        self.assertEqual("I cannot measure the free space in the left sector.",
                         hdsg.no_free_space_text("left"))


if __name__ == "__main__":
    unittest.main()
