"""Whether a moving object reaches the prompt at all.

A fact left out of the prompt is not merely unmentioned. `permitted_facts` is built from the same
list, so the gate refuses a caption that names anything absent from it with
`RG_FACT_REFERENCE_INVALID`. Two faults met here until 25 August 2026 and both ended the same way: a
person walking towards the user went unmentioned in the sentence asking the user to walk that way.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401
from support import caption, declares_object, event, gate, sectors  # noqa: F401
from scripts import hdsg_runtime as hdsg  # noqa: E402

ALL_CLEAR = sectors(left=3.0, centre=3.0, right=3.0, statuses=("CLEAR", "CLEAR", "CLEAR"))


def moving(track_id, label, bearing, distance_m):
    from support import detected_object
    items = detected_object(track_id, label, bearing, distance_m)
    items = items if isinstance(items, list) else [items]
    for item in items:
        item["motion_state"] = "MOVING"
    return items


def still(track_id, label, bearing, distance_m):
    from support import detected_object
    items = detected_object(track_id, label, bearing, distance_m)
    return items if isinstance(items, list) else [items]


def prompt_for(objects, response_mode="AUTOMATIC", lane=ALL_CLEAR):
    packet, prompt = event(objects=objects, lane=lane, response_mode=response_mode)
    return packet, prompt


def alert(prompt):
    return next((item for item in prompt["requirements"]
                 if item["requirement_id"] == "moving_object_alert"), None)


class AMovingObjectIsAlwaysPermitted(unittest.TestCase):
    """The alert was skipped whenever the budget was already full.

    In the sector-choice state the action binding takes all three of the automatic profile's
    clauses, so `minimum_required_clauses < max_reasons` was false and the alert was dropped, taking
    the object out of `permitted_facts` with it. Settled by Atiq on 25 August 2026: a moving object
    is always permitted, at the cost of a clause. This changes the budget deferred at
    HDSG_VERIFIED_GENERATION_POLICY.md section 10.4.
    """

    def setUp(self):
        # A chair blocking the way ahead with both sides open reaches AWAITING_SECTOR_CHOICE, whose
        # action binding is three facts under ALL_OF and therefore fills the automatic budget.
        self.objects = moving(3, "chair", "CENTRE", 0.40) + moving(5, "person", "LEFT", 2.50)

    def test_the_scene_really_does_fill_the_budget(self):
        """Without this the test below could pass on a scene that never had the problem."""
        packet, prompt = prompt_for(self.objects)
        self.assertEqual("AWAITING_SECTOR_CHOICE", packet["deterministic"]["interaction_state"])
        action = next(item for item in prompt["requirements"]
                      if item["requirement_id"] == "action_reason")
        self.assertEqual("ALL_OF", action["match"])
        self.assertEqual(3, len(action["fact_ids"]))
        self.assertEqual(3, hdsg.PROFILE_LIMITS["AUTOMATIC"][0])

    def test_the_moving_person_is_still_required(self):
        _, prompt = prompt_for(self.objects)
        self.assertEqual(["object:5"], alert(prompt)["fact_ids"])

    def test_the_moving_person_is_permitted(self):
        _, prompt = prompt_for(self.objects)
        self.assertIn("object:5", [item["fact_id"] for item in prompt["permitted_facts"]])

    def test_naming_the_moving_person_is_no_longer_refused(self):
        """The consequence that matters. The walker asks the user to choose left or right, and
        naming the person moving on the left was rejected as an invalid fact reference."""
        packet, prompt = prompt_for(self.objects)
        candidate = caption(
            "A chair is detected in the centre at 0.40 metres. "
            "A person is moving on the left at 2.50 metres. "
            "The right sector is clear for 3.00 metres.",
            assertions=[declares_object(3, 0.40), declares_object(5, 2.50)])
        self.assertNotIn("RG_FACT_REFERENCE_INVALID", gate(candidate, packet, prompt)[0])


class TheAlertListsOnlyWhatNothingElseRequires(unittest.TestCase):
    """The list held every moving object under ANY_OF, including ones the action binding already
    required, so it was satisfiable by repeating a fact the caption had to name anyway.

    Naming any one member satisfies ANY_OF. With a moving chair binding the decision and a moving
    person beside it, naming the chair satisfied the alert and the person was never mentioned.
    """

    def test_an_object_the_action_binding_requires_is_not_relisted(self):
        _, prompt = prompt_for(moving(3, "chair", "CENTRE", 0.40) + moving(5, "person", "LEFT", 2.50))
        self.assertEqual(["object:5"], alert(prompt)["fact_ids"])

    def test_the_alert_is_absent_when_every_mover_is_required_elsewhere(self):
        """An empty requirement is satisfied by anything, so it is not added at all."""
        _, prompt = prompt_for(moving(3, "chair", "CENTRE", 0.40))
        self.assertIsNone(alert(prompt))
        self.assertIn("object:3", [item["fact_id"] for item in prompt["permitted_facts"]])

    def test_several_unbound_movers_are_all_listed(self):
        objects = (moving(3, "chair", "CENTRE", 0.40) + moving(5, "person", "LEFT", 2.50)
                   + moving(6, "person", "RIGHT", 2.60))
        _, prompt = prompt_for(objects)
        self.assertEqual(["object:5", "object:6"], alert(prompt)["fact_ids"])

    def test_a_stationary_object_never_reaches_the_alert(self):
        objects = moving(5, "person", "LEFT", 2.50) + still(6, "chair", "RIGHT", 2.60)
        _, prompt = prompt_for(objects)
        self.assertEqual(["object:5"], alert(prompt)["fact_ids"])

    def test_a_scene_with_nothing_moving_has_no_alert(self):
        _, prompt = prompt_for(still(6, "chair", "RIGHT", 2.60))
        self.assertIsNone(alert(prompt))


if __name__ == "__main__":
    unittest.main()
