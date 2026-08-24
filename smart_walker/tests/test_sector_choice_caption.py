"""The sentence that asks the person to pick a side.

It names one sector as blocked and two as clear, and until 25 August 2026 it derived neither name
from the decision that produced it. The blocked sector was found by searching the action binding for
the first fact whose status was not CLEAR, without restricting the search to sector facts. An object
fact carries no status field, so `.get("status")` returned None, None is not "CLEAR", and the object
won. The name was then taken as the text after the colon, which for `object:3` is the detection's
number: "The 3 sector is blocked", spoken aloud, whenever a detected object rather than a depth strip
was what blocked the way. The pair of clear sectors was written out as "left and right" in the same
sentence, which is what the policy offers when the intent is forward.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401
from support import detected_object, event, sectors  # noqa: F401
from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

ALL_CLEAR = sectors(left=3.0, centre=3.0, right=3.0, statuses=("CLEAR", "CLEAR", "CLEAR"))
CENTRE_BLOCKED = sectors(left=3.0, centre=0.4, right=3.0,
                         statuses=("CLEAR", "BLOCKED", "CLEAR"))


def scene(lane, objects=(), intent="FORWARD"):
    packet, prompt = event(objects=objects, object_advisory="CAUTION", lane=lane, intent=intent)
    release = composed.build_composed_release(
        packet, prompt, release_id="release_1", candidate=None,
        scored_assertions=[], failure_codes=["RG_MODEL_UNAVAILABLE"])
    return packet, release["content"]["caption_text"]


class TheBlockedSectorIsNamedAsASector(unittest.TestCase):

    def test_an_object_blocking_the_centre_names_the_centre(self):
        packet, caption_text = scene(ALL_CLEAR, detected_object(3, "chair", "CENTRE", 0.40))
        self.assertEqual("AWAITING_SECTOR_CHOICE", packet["deterministic"]["interaction_state"])
        self.assertEqual(
            "Stop. The centre sector is blocked, while the left and right sectors are "
            "similarly clear. Select left or right.", caption_text)

    def test_the_detection_number_does_not_reach_the_sentence(self):
        """The number is what leaked. It is asserted directly because a tracker identifier that
        happens to be 1, 2 or 3 reads as a plausible sentence rather than an obvious defect."""
        for track_id in (3, 17):
            with self.subTest(track_id=track_id):
                _, caption_text = scene(ALL_CLEAR,
                                        detected_object(track_id, "chair", "CENTRE", 0.40))
                self.assertNotIn(f"The {track_id} sector", caption_text)
                self.assertIn("The centre sector is blocked", caption_text)

    def test_a_blocked_strip_still_names_the_same_sector(self):
        """The path that always worked, held. The object is what broke it, so a scene without one
        must reach the same sentence."""
        _, caption_text = scene(CENTRE_BLOCKED)
        self.assertIn("The centre sector is blocked", caption_text)

    def test_the_name_comes_from_the_binding_rather_than_a_default(self):
        """`_blocked_sector_name` fell back to "sector:centre" when its search found nothing, so a
        sentence that reads correctly is not on its own evidence that the derivation ran."""
        packet, _ = scene(ALL_CLEAR, detected_object(3, "chair", "CENTRE", 0.40))
        binding = packet["deterministic"]["action_binding"]["accepted_fact_ids"]
        self.assertEqual("object:3", binding[0])
        self.assertEqual("centre", hdsg._blocked_sector_name(packet, binding))
        self.assertEqual("left", hdsg._blocked_sector_name(packet, ["sector:left"]))


class TheClearPairComesFromWhatIsOffered(unittest.TestCase):
    """"left and right" was written into the sentence. It is correct for a forward intent, which is
    the only intent observed to reach this state, so this is a defect of derivation rather than one
    with a scene that exhibits it today. The sentence must not be able to disagree with the two
    buttons the browser page unhides from the same list.
    """

    def test_the_pair_matches_the_offered_options(self):
        packet, caption_text = scene(ALL_CLEAR, detected_object(3, "chair", "CENTRE", 0.40))
        options = [name.lower() for name in packet["deterministic"]["selection_options"]]
        self.assertEqual(["left", "right"], options)
        self.assertIn(f"while the {' and '.join(options)} sectors are similarly clear",
                      caption_text)

    def test_a_different_pair_would_be_named(self):
        """Asserted against `_fallback_reason` with the options replaced, since the policy offers
        only left and right today and a scene cannot produce the other pair."""
        packet, _ = scene(ALL_CLEAR, detected_object(3, "chair", "CENTRE", 0.40))
        packet["deterministic"]["selection_options"] = ["LEFT", "CENTRE"]
        text, _, _, _ = hdsg._fallback_reason(packet)
        self.assertIn("while the left and centre sectors are similarly clear", text)


class ASectorAnObjectStandsInIsNotCalledClear(unittest.TestCase):
    """The spoken answer contradicted itself in the same breath.

    A strip clearance and an object distance are measured differently, so the centre strip can read
    3.00 metres while a chair stands 0.40 metres into it. Both sentences are true and the packet
    carries the strip status, which is CLEAR. The detail slot rendered it, and the clause landed
    last, immediately before the instruction:

        A chair is detected in the centre at 0.40 metres. The left sector is clear for 3.00 metres.
        The right sector is clear for 3.00 metres. The centre sector is clear for 3.00 metres.
        Select left or right.

    The clause is dropped rather than reworded. The chair is already named with its own distance in
    the action binding, so no measurement is lost, and the alternative is publishing an
    object-adjusted status into a fact packet that is frozen against a schema.
    """

    def answer(self, lane, objects=()):
        packet, prompt = event(objects=objects, object_advisory="CAUTION", lane=lane)
        release = composed.build_composed_release(
            packet, prompt, release_id="release_1", candidate=None,
            scored_assertions=[], failure_codes=["RG_MODEL_UNAVAILABLE"])
        from scripts import hdsg_questions as questions
        return prompt, questions.with_action_prefix(
            release, questions.deterministic_answer(packet, prompt["requirements"]))

    def test_the_blocked_sector_earns_no_detail_clause(self):
        prompt, answer = self.answer(ALL_CLEAR, detected_object(3, "chair", "CENTRE", 0.40))
        self.assertEqual(["action_reason"],
                         [item["requirement_id"] for item in prompt["requirements"]])
        self.assertNotIn("The centre sector is clear", answer)

    def test_the_object_and_the_open_sides_are_still_named(self):
        """The clause is dropped, not the scene. What the person needs is the chair's distance and
        which way is open."""
        _, answer = self.answer(ALL_CLEAR, detected_object(3, "chair", "CENTRE", 0.40))
        self.assertEqual(
            "Stop. A chair is detected in the centre at 0.40 metres. "
            "The left sector is clear for 3.00 metres. "
            "The right sector is clear for 3.00 metres. Select left or right.", answer)

    def test_a_clear_sector_with_nothing_in_it_still_earns_its_clause(self):
        """The condition is that the decision excluded the sector, not that a strip read CLEAR."""
        _, answer = self.answer(ALL_CLEAR)
        self.assertIn("The centre sector is clear for 3.00 metres.", answer)

    def test_a_constrained_strip_is_unaffected(self):
        """It is excluded from `clear_sectors` too, and its own status is not CLEAR, so it is
        described by its measurement rather than dropped."""
        lane = sectors(left=3.0, centre=1.2, right=3.0,
                       statuses=("CLEAR", "CONSTRAINED", "CLEAR"))
        _, answer = self.answer(lane)
        self.assertIn("The centre sector has limited clearance at 1.20 metres.", answer)


if __name__ == "__main__":
    unittest.main()
