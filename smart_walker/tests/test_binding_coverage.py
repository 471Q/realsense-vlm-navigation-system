"""The caption must state the measurement the decision rests on.

Until 25 August 2026 the gate guaranteed what a caption may not contain and nothing about what it
must. A walker stopping for a chair 0.40 metres ahead accepted and released "The right sector is
clear for 3.00 metres": true, permitted, correctly attributed, and silent about the reason for
stopping. Chapter 3's Property 3 is that the released explanation binds to the decision, and an
explanation that need not mention the binding fact does not support it.

Restored as `RG_REQUIRED_FACT_MISSING`, which was retired with the templated contract on 21 August
2026 on the ground that establishing prose coverage would need linguistic inference the architecture
keeps outside the safety boundary. Under the composed contract the model declares each measurement
it states against the fact identifier it belongs to, so the check compares identifiers.
"""

from __future__ import annotations

import unittest

import support  # noqa: F401
from support import (caption, declares, declares_object, detected_object, event, gate,  # noqa: F401
                     sectors)
from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

ALL_CLEAR = sectors(left=3.0, centre=3.0, right=3.0, statuses=("CLEAR", "CLEAR", "CLEAR"))
NO_STRIPS = sectors(left=None, centre=None, right=None,
                    statuses=("UNKNOWN", "UNKNOWN", "UNKNOWN"))


class TheCauseMustBeStated(unittest.TestCase):
    """The case that opened this: a chair 0.40 m ahead, both sides open, the walker asking the user
    to choose a side."""

    def setUp(self):
        self.packet, self.prompt = event(
            objects=detected_object(3, "chair", "CENTRE", 0.40), lane=ALL_CLEAR,
            response_mode="AUTOMATIC")

    def test_the_chair_is_the_cause(self):
        binding = self.packet["deterministic"]["action_binding"]
        self.assertEqual("object:3", binding["primary_fact_id"])

    def test_a_caption_about_a_different_part_of_the_scene_is_refused(self):
        codes, _ = gate(caption("The right sector is clear for 3.00 metres.",
                                [declares("right", 3.00)]), self.packet, self.prompt)
        self.assertIn("RG_REQUIRED_FACT_MISSING", codes)

    def test_the_failure_is_recorded_against_the_fact(self):
        """Chapter 5 must be able to say which element went unstated, not only that one did."""
        _, scored = gate(caption("The right sector is clear for 3.00 metres.",
                                 [declares("right", 3.00)]), self.packet, self.prompt)
        entry = next(item for item in scored if item.get("outcome") == "BINDING_FACT_NOT_STATED")
        self.assertEqual("object:3", entry["fact_id"])
        self.assertEqual("action_binding", entry["source"])

    def test_a_caption_naming_the_chair_passes_the_check(self):
        codes, _ = gate(caption("A chair is detected in the centre at 0.40 metres.",
                                [declares_object(3, 0.40)]), self.packet, self.prompt)
        self.assertNotIn("RG_REQUIRED_FACT_MISSING", codes)

    def test_the_refused_caption_falls_back_to_one_that_names_the_cause(self):
        """Rejection is not silence. The consequence of the check is that the person hears the
        walker's own sentence, and that sentence always names the cause."""
        codes, scored = gate(caption("The right sector is clear for 3.00 metres.",
                                     [declares("right", 3.00)]), self.packet, self.prompt)
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_1",
            candidate=caption("The right sector is clear for 3.00 metres.",
                              [declares("right", 3.00)]),
            scored_assertions=scored, failure_codes=codes)
        self.assertIn("chair", release["content"]["caption_text"])
        self.assertIn("0.40 metres", release["content"]["caption_text"])


class OnlyTheCauseIsRequired(unittest.TestCase):
    """`accepted_fact_ids` is ordered: the cause, then the context the decision was taken in.

    Requiring every measured entry refuses 47 further captions across this suite whose only fault is
    not repeating a direction the person has already heard. The action sentence is deterministic and
    always shown, and it reads "Change direction and continue towards the left." before the model's
    explanation begins.
    """

    def setUp(self):
        self.packet, self.prompt = event()

    def test_the_default_scene_binds_a_cause_and_a_destination(self):
        binding = self.packet["deterministic"]["action_binding"]
        self.assertEqual(["sector:centre", "sector:left"], binding["accepted_fact_ids"])
        self.assertEqual("sector:centre", binding["primary_fact_id"])

    def test_naming_the_cause_alone_is_enough(self):
        codes, _ = gate(caption("The centre narrows to 1.74 metres.",
                                [declares("centre", 1.74)]), self.packet, self.prompt)
        self.assertNotIn("RG_REQUIRED_FACT_MISSING", codes)

    def test_naming_the_destination_alone_is_not(self):
        codes, _ = gate(caption("The left side is clear for 3.13 metres.",
                                [declares("left", 3.13)]), self.packet, self.prompt)
        self.assertIn("RG_REQUIRED_FACT_MISSING", codes)

    def test_the_person_hears_the_destination_from_the_deterministic_layer(self):
        """Which is why the model is not required to repeat it."""
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_1", candidate=None,
            scored_assertions=[], failure_codes=["RG_MODEL_UNAVAILABLE"])
        self.assertIn("towards the left", release["content"]["action_text"])


class ADeclarationIsNotEnoughOnItsOwn(unittest.TestCase):
    """The specification compared declared identifiers only. A declaration whose value the prose
    never writes is metadata, and the property being restored is about what the person is told."""

    def test_declaring_the_cause_without_stating_it_is_refused(self):
        packet, prompt = event()
        codes, scored = gate(
            caption("The left side is clear for 3.13 metres.",
                    [declares("centre", 1.74), declares("left", 3.13)]), packet, prompt)
        stated = {entry.get("fact_id"): entry.get("stated_in_caption") for entry in scored
                  if "stated_in_caption" in entry}
        self.assertFalse(stated["sector:centre"], "the fixture must not state the centre's value")
        self.assertIn("RG_REQUIRED_FACT_MISSING", codes)

    def test_a_wrong_value_for_the_cause_does_not_count_as_stating_it(self):
        packet, prompt = event()
        codes, _ = gate(caption("The centre narrows to 1.70 metres.",
                                [declares("centre", 1.70)]), packet, prompt)
        self.assertIn("RG_STATED_VALUE_MISMATCH", codes)
        self.assertIn("RG_REQUIRED_FACT_MISSING", codes)


class AnUnmeasuredCauseIsNotDemanded(unittest.TestCase):
    """A declaration is made of a measurement, so a cause carrying none cannot be declared. The
    scene where every strip fails must not be made unreportable by this check."""

    def test_a_condition_carries_no_measurement_and_is_not_required(self):
        packet, prompt = event(lane=NO_STRIPS, response_mode="AUTOMATIC")
        binding = packet["deterministic"]["action_binding"]
        self.assertEqual("condition:no_clear_sector", binding["primary_fact_id"])
        self.assertEqual({}, composed.uncovered_binding_facts(prompt, packet, []))

    def test_a_sector_whose_strip_failed_is_not_required(self):
        """The walker redirects away from a centre it could not measure. The centre is the cause and
        carries no clearance, so there is nothing for the model to declare."""
        lane = sectors(left=3.13, centre=None, right=1.16,
                       statuses=("CLEAR", "UNKNOWN", "CONSTRAINED"))
        packet, prompt = event(lane=lane)
        binding = packet["deterministic"]["action_binding"]
        self.assertEqual("sector:centre", binding["primary_fact_id"])
        self.assertIsNone(packet["sectors"]["centre"]["clearance_m"])
        self.assertEqual({}, composed.uncovered_binding_facts(prompt, packet, []))

    def test_an_object_with_no_distance_never_becomes_the_cause(self):
        """The other half of the same guarantee, held from the decision side. `determine_authority`
        skips an object whose distance is not finite, so an unmeasured object cannot bind and the
        exclusion above is never asked to cover one."""
        objects = hdsg.normalise_objects([
            {"id": 9, "track_id": 9, "raw_label": "chair", "canonical_class": "chair",
             "bbox_xyxy": [0, 0, 10, 10], "distance_m": None, "conf": 0.9, "bearing": "CENTRE"}])
        packet, prompt = event(objects=objects, lane=ALL_CLEAR)
        binding = packet["deterministic"]["action_binding"]
        self.assertNotIn("object:9", binding["accepted_fact_ids"])
        self.assertEqual("sector:centre", binding["primary_fact_id"])

    def test_an_event_with_no_binding_at_all_is_not_required(self):
        packet, prompt = event(intent="NONE")
        self.assertEqual([], packet["deterministic"]["action_binding"]["accepted_fact_ids"])
        self.assertEqual({}, composed.uncovered_binding_facts(prompt, packet, []))


class TheCodeIsDeclaredWhereItMustBe(unittest.TestCase):

    def test_the_code_is_in_the_runtime_ordering(self):
        self.assertIn("RG_REQUIRED_FACT_MISSING", hdsg.REASON_CODE_ORDER)

    def test_the_code_sorts_ahead_of_the_value_failures(self):
        """A caption that never mentions the cause has the more basic fault, so it is the reason
        reported when it fails alongside a wrong value."""
        order = hdsg._reason_sort(["RG_STATED_VALUE_MISMATCH", "RG_REQUIRED_FACT_MISSING"])
        self.assertEqual("RG_REQUIRED_FACT_MISSING", order[0])

    def test_the_release_schema_carries_it(self):
        import json
        schema = json.loads((support.ROOT / "schemas" / "hdsg.release.v2.schema.json")
                            .read_text(encoding="utf-8"))
        codes = schema["definitions"]["reason_code"]["enum"]
        self.assertIn("RG_REQUIRED_FACT_MISSING", codes)


if __name__ == "__main__":
    unittest.main()


class AttributionRecordsAndDoesNotRefuse(unittest.TestCase):
    """Decided by Atiq on 25 August 2026, argued at Chapter3_And_5_Revision_Notes.md section 2.11.

    Every other check in the gate decides its question exactly, from declared identifiers and
    measured values. Attribution reads the prose: each number goes to the nearest fact the caption
    names within its own sentence, measured in characters. That is the inference
    `hdsg.vlm_caption.v1.gbnf` states the gate does not perform, and a heuristic over English cannot
    be completed by adding shapes to it, so it finds the departures whose phrasing falls inside the
    rule and misses the rest. A refusal rate built on it mixes a real property with an accident of
    phrasing. The same decision as the absent-class check, at section 10.15 of
    HDSG_VERIFIED_GENERATION_POLICY.md.

    Placed in this file because the two decisions are opposite halves of one question: what the gate
    may enforce is what it can decide exactly, and naming the causal fact is decidable from
    identifiers while placing a number beside it is not.
    """

    def setUp(self):
        self.packet, self.prompt = event()

    def run_gate(self, text, assertions):
        return gate(caption(text, assertions), self.packet, self.prompt)

    def test_a_swapped_pair_is_released(self):
        """The cost of the decision, asserted rather than left implicit. This caption is false in
        both clauses and the person now hears it."""
        codes, _ = self.run_gate(
            "The centre narrows to 3.13 metres and the left is clear for 1.74 metres.",
            [declares("centre", 1.74), declares("left", 3.13)])
        self.assertNotIn("RG_SUBJECT_MISMATCH", codes)

    def test_the_swap_is_still_recorded(self):
        """Releasing it is not the same as missing it. Chapter 5 reports how often this happened."""
        _, scored = self.run_gate(
            "The centre narrows to 3.13 metres and the left is clear for 1.74 metres.",
            [declares("centre", 1.74), declares("left", 3.13)])
        outcomes = [item.get("outcome") for item in scored]
        self.assertIn("ATTRIBUTION_FORM_NOT_FOLLOWED", outcomes)

    def test_the_two_outcomes_stay_apart(self):
        """The reason the check is worth keeping at all. One is false, the other is only unwanted
        word order, and a single figure adding them answers no question."""
        _, misattributed = self.run_gate("The left is clear for 1.74 metres.",
                                         [declares("centre", 1.74)])
        _, form = self.run_gate("The centre, wider than the left, is 1.74 metres.",
                                [declares("centre", 1.74)])
        self.assertEqual({"MISATTRIBUTED": 1},
                         dict(composed.assertion_summary(misattributed)["attribution_reasons"]))
        self.assertEqual({"FORM_NOT_FOLLOWED": 1},
                         dict(composed.assertion_summary(form)["attribution_reasons"]))

    def test_no_caption_is_refused_for_attribution_alone(self):
        """`RG_SUBJECT_MISMATCH` is emitted by nothing. Held against the source rather than by
        enumerating captions, because absence cannot be shown by example."""
        source = (support.ROOT / "scripts" / "hdsg_composed.py").read_text(encoding="utf-8")
        emitted = [line for line in source.splitlines()
                   if "RG_SUBJECT_MISMATCH" in line and "errors.append" in line]
        self.assertEqual([], emitted)

    def test_the_code_stays_in_the_enumeration(self):
        """The archived runs written between 22 and 25 August 2026 contain it, and a frozen schema
        that cannot validate its own archive is of no use as evidence."""
        import json
        schema = json.loads((support.ROOT / "schemas" / "hdsg.release.v2.schema.json")
                            .read_text(encoding="utf-8"))
        self.assertIn("RG_SUBJECT_MISMATCH", schema["definitions"]["reason_code"]["enum"])
        self.assertIn("RG_SUBJECT_MISMATCH", hdsg.REASON_CODE_ORDER)

    def test_the_exactly_decidable_checks_still_refuse(self):
        """The decision is about one check and must not have loosened the others."""
        codes, _ = self.run_gate("The centre narrows to 1.70 metres.", [declares("centre", 1.70)])
        self.assertIn("RG_STATED_VALUE_MISMATCH", codes)
