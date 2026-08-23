"""The release, checked against every state the runtime can publish one in.

The release is the only record user-visible text may be taken from, so a release that validates is
the guarantee that what reached the screen is describable. The sweep builds one from every
combination of response mode, intent, sector layout, scene, and the two overriding flags, and
validates all of them.

Six failed on 24 August 2026, all one fault: an idle release carrying a sector choice. See
`test_runtime_authority.test_no_intent_withholds_the_decision_rather_than_assuming_forward` for the
half of that fault the person could see.
"""

from __future__ import annotations

import json
import re
import unittest

from support import (SCHEMAS, caption, declares, detected_object, event, gate,  # noqa: F401
                     observation, sectors)
from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

LANES = {
    "all clear": sectors(3.2, 3.2, 3.2, ("CLEAR",) * 3),
    "centre blocked": sectors(3.2, 0.5, 0.4, ("CLEAR", "CONSTRAINED", "CONSTRAINED")),
    "centre blocked, sides tied": sectors(3.2, 0.4, 3.2, ("CLEAR", "BLOCKED", "CLEAR")),
    "nothing passable": sectors(0.4, 0.4, 0.4, ("BLOCKED",) * 3),
}


class ReleaseSweepTests(unittest.TestCase):

    def setUp(self):
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")
        self.validator = jsonschema.Draft7Validator(json.loads(
            (SCHEMAS / "hdsg.release.v2.schema.json").read_text(encoding="utf-8")))

    def releases(self):
        for mode in ("AUTOMATIC", "MORE_DETAIL", "REASSESSMENT"):
            for intent in ("FORWARD", "LEFT", "RIGHT", "BACKWARD"):
                for lane_name, lane in LANES.items():
                    for scene, objects in (("empty", ()),
                                           ("one object close", detected_object(distance_m=0.6))):
                        for pending in (False, True):
                            for no_intent in (False, True):
                                packet, prompt = event(intent=intent, lane=lane, objects=objects,
                                                       response_mode=mode)
                                name = (f"{mode}, {intent}, {lane_name}, {scene}"
                                        f"{', pending' if pending else ''}"
                                        f"{', idle' if no_intent else ''}")
                                yield name, hdsg.build_release(
                                    packet, prompt, release_id="release_1",
                                    pending=pending, no_intent=no_intent)

    def test_every_release_the_runtime_can_publish_is_accepted(self):
        for name, release in self.releases():
            with self.subTest(name):
                self.assertEqual(
                    [], [error.message for error in self.validator.iter_errors(release)])

    def test_an_idle_release_offers_no_choice(self):
        """The state and the option list are one statement recorded twice. An idle release listing
        two sectors says both that nothing was asked and that a choice is pending."""
        for name, release in self.releases():
            if release["authority"]["interaction_state"] != "IDLE_NO_INTENT":
                continue
            with self.subTest(name):
                self.assertEqual([], release["authority"]["selection_options"])

    def test_the_shown_caption_is_exactly_its_recorded_parts(self):
        """The claim the release makes: nothing reaches the screen that is not in a recorded field.

        `caption_text` is what the person reads and the other four fields are what the evidence
        binds to, so a word appearing in the first and in none of the others would be text with no
        recorded provenance. Checked by reconstruction rather than by substring, which would pass on
        a caption that had gained a clause.
        """
        for name, release in self.releases():
            with self.subTest(name):
                content = release["content"]
                parts = [content["action_text"], content["reason_text"]]
                if content["interaction_text"]:
                    parts.append(content["interaction_text"])
                parts.extend(content["additional_detail_texts"])
                rebuilt = " ".join(part.strip() for part in parts if part and part.strip())
                self.assertEqual(rebuilt, content["caption_text"])

    def test_text_is_absent_only_in_the_states_that_have_none(self):
        """`action_text` and `reason_text` lost their minimum lengths so the idle and pending states
        could be expressed. The states that may leave them empty are named here, so a later change
        that empties one in an active release is a failure rather than an accepted blank."""
        for name, release in self.releases():
            with self.subTest(name):
                state = release["authority"]["interaction_state"]
                if not release["content"]["action_text"]:
                    self.assertEqual("IDLE_NO_INTENT", state)
                if not release["content"]["reason_text"]:
                    self.assertIn(state, {"IDLE_NO_INTENT", "GENERATION_PENDING"})

    def test_the_sweep_reaches_the_states_it_claims_to(self):
        """A guard. The sweep proves nothing about a state it never builds."""
        produced = {release["authority"]["interaction_state"] for _, release in self.releases()}
        self.assertEqual(
            {"GUIDANCE_ACTIVE", "AWAITING_SECTOR_CHOICE", "REORIENTATION_REQUIRED",
             "IDLE_NO_INTENT", "GENERATION_PENDING"}, produced)


class AcceptedCaptionContentTests(unittest.TestCase):
    """The accepted path, where the reason text is the model's own and not the runtime's.

    The deterministic sweep above never reaches it: `build_release` writes the fallback sentence,
    and only `build_composed_release` substitutes a caption the gate has passed.
    """

    def setUp(self):
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")
        self.validator = jsonschema.Draft7Validator(json.loads(
            (SCHEMAS / "hdsg.release.v2.schema.json").read_text(encoding="utf-8")))
        self.packet, self.prompt = event(response_mode="MORE_DETAIL")

    def accepted(self, visuals=()):
        candidate = caption("The centre is clear for 1.74 metres.",
                            [declares("centre", 1.74)], visuals)
        codes, scored = gate(candidate, self.packet, self.prompt)
        self.assertEqual([], codes, "the fixture must be a caption the gate accepts")
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_1", candidate=candidate,
            scored_assertions=scored, failure_codes=codes)
        return candidate, release

    def test_the_reason_text_is_the_model_text_unaltered(self):
        """The reason text is evidence of what the model wrote, so it is stored as written. The
        shown caption is sentence-cased and punctuated on the way to the screen, which is why the
        two can differ by a capital letter and a full stop and both are correct."""
        candidate, release = self.accepted()
        self.assertEqual(candidate["caption"], release["content"]["reason_text"])
        self.assertEqual("VLM_ACCEPTED", release["verification"]["release_mode"])

    def test_a_visual_observation_becomes_one_detail_line(self):
        visuals = [observation("visual:1", "doorway", "LEFT"),
                   observation("visual:2", "wardrobe", "RIGHT")]
        _, release = self.accepted(visuals)
        self.assertEqual(["Possible doorway is visible in the left.",
                          "Possible wardrobe is visible in the right."],
                         release["content"]["additional_detail_texts"])
        self.assertEqual([], [e.message for e in self.validator.iter_errors(release)])

    def test_more_visual_observations_than_the_profile_allows_are_never_rendered(self):
        """The schema admits four detail lines and the profile allows two, so the headroom is only
        safe while the gate refuses the excess. It refuses the whole caption, and the release falls
        back, which is why no release can carry three."""
        visuals = [observation(f"visual:{index}", f"thing{index}", "LEFT")
                   for index in (1, 2, 3, 4)]
        candidate = caption("The centre is clear for 1.74 metres.",
                            [declares("centre", 1.74)], visuals)
        codes, scored = gate(candidate, self.packet, self.prompt)
        self.assertIn("RG_PROFILE_LIMIT_EXCEEDED", codes)
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_1", candidate=candidate,
            scored_assertions=scored, failure_codes=codes)
        self.assertEqual("DETERMINISTIC_FALLBACK", release["verification"]["release_mode"])
        self.assertEqual([], release["content"]["additional_detail_texts"])


class EvidenceTests(unittest.TestCase):
    """What the released text rests on, checked against the packet it came from.

    Evidence is the half of the release that makes a caption checkable after the fact. A number on
    screen with no substitution behind it, or a substitution naming a measurement the packet does
    not hold, both leave a sentence that cannot be verified against anything.
    """

    def releases(self):
        lanes = dict(LANES)
        # Replaces the all-blocked layout with one whose three clearances differ, so that the widest
        # of them is a specific sector rather than a tie.
        lanes["nothing passable"] = sectors(0.40, 0.50, 0.45, ("BLOCKED",) * 3)
        for intent in ("FORWARD", "LEFT", "RIGHT", "BACKWARD"):
            for lane_name, lane in lanes.items():
                for scene, objects in (("empty", ()),
                                       ("one object close", detected_object(distance_m=0.6))):
                    packet, prompt = event(intent=intent, lane=lane, objects=objects)
                    yield (f"{intent}, {lane_name}, {scene}", packet,
                           hdsg.build_release(packet, prompt, release_id="release_1"))

    def test_every_substitution_resolves_to_the_value_it_states(self):
        """A substitution is an identifier and a rendered number. If the identifier cannot be
        resolved in the packet the pair asserts nothing, and if it resolves to a different number
        the release contradicts its own measurement."""
        checked = 0
        for name, packet, release in self.releases():
            for substitution in release["evidence"]["measurement_substitutions"]:
                with self.subTest(f"{name}: {substitution['measurement_id']}"):
                    value = composed.measured_value(packet, substitution["measurement_id"])
                    self.assertIsNotNone(value, "the identifier names nothing in the packet")
                    self.assertEqual(hdsg._format_measurement(value),
                                     substitution["formatted_value"])
                    checked += 1
        self.assertGreater(checked, 0, "the sweep produced no substitutions to check")

    def test_every_number_shown_has_a_substitution_behind_it(self):
        """The other direction. The first test cannot see a number that was printed and never
        recorded, which is the case that leaves a figure on screen with no provenance."""
        for name, _packet, release in self.releases():
            with self.subTest(name):
                recorded = " ".join(item["formatted_value"]
                                    for item in release["evidence"]["measurement_substitutions"])
                for number in re.findall(r"[0-9]+[.][0-9]+", release["content"]["caption_text"]):
                    self.assertIn(number, recorded)

    def test_the_widest_sector_is_named_by_its_own_measurement(self):
        """When nothing is passable the release states the greatest clearance measured. It recorded
        that against `m:sector:best:clearance` until 24 August 2026, and no sector is called best,
        so the identifier resolved to nothing. It also split one measurement across two names in the
        contribution comparison, which matches identifiers between conditions and can only ever see
        the model declare an identifier it was offered.
        """
        packet, prompt = event(intent="FORWARD",
                               lane=sectors(0.40, 0.50, 0.45, ("BLOCKED",) * 3))
        release = hdsg.build_release(packet, prompt, release_id="release_1")
        self.assertEqual([{"measurement_id": "m:sector:centre:clearance",
                           "formatted_value": "0.50 metres"}],
                         release["evidence"]["measurement_substitutions"])

    def test_a_tie_for_widest_settles_on_the_sector_order(self):
        """Two sectors measuring the same leaves the choice to dictionary order otherwise, so the
        same scene could name a different sector between runs."""
        packet, prompt = event(intent="FORWARD",
                               lane=sectors(0.50, 0.40, 0.50, ("BLOCKED",) * 3))
        release = hdsg.build_release(packet, prompt, release_id="release_1")
        self.assertEqual("m:sector:left:clearance",
                         release["evidence"]["measurement_substitutions"][0]["measurement_id"])

    def test_a_binding_is_empty_only_where_no_intent_was_expressed(self):
        for name, _packet, release in self.releases():
            with self.subTest(name):
                if not release["evidence"]["action_binding_fact_ids"]:
                    self.assertEqual("IDLE_NO_INTENT",
                                     release["authority"]["interaction_state"])


class ReasonCodeTests(unittest.TestCase):
    """The rejection reasons, and the order that decides which of them is reported as the reason.

    `primary_reason_code` is the first of the sorted codes, and it is what an evaluation counts when
    it breaks rejections down by cause. A code missing from the ordering therefore does not merely
    sort oddly; it is never counted.
    """

    def setUp(self):
        self.declared = set(json.loads(
            (SCHEMAS / "hdsg.release.v2.schema.json").read_text(encoding="utf-8")
        )["definitions"]["reason_code"]["enum"])

    def emitted(self):
        """Every code appearing as a literal anywhere the runtime could write one."""
        sources = ("hdsg_runtime.py", "hdsg_composed.py", "hdsg_questions.py",
                   "realsense_vlm_on_change_qwen.py", "hdsg_contribution.py")
        found: set[str] = set()
        for name in sources:
            found.update(re.findall(
                r"RG_[A-Z_]+",
                (SCHEMAS.parent / "scripts" / name).read_text(encoding="utf-8")))
        return found

    def test_every_code_the_runtime_can_write_is_declared(self):
        self.assertEqual(set(), self.emitted() - self.declared)

    def test_the_enumeration_declares_nothing_the_runtime_cannot_write(self):
        """The reverse direction, which nothing checked. Six codes outlived the contract that
        raised them and were kept to validate archives deleted on 23 August 2026."""
        self.assertEqual(set(), self.declared - self.emitted())

    def test_every_failure_code_has_a_place_in_the_ordering(self):
        """An unlisted code sorts after every listed one, including the catch-all internal error,
        so it can never be the primary reason when anything else is present. RG_SUBJECT_MISMATCH
        was in exactly that position until 24 August 2026."""
        missing = self.declared - set(hdsg.REASON_CODE_ORDER) - {"RG_ACCEPTED"}
        self.assertEqual(set(), missing)

    def test_a_subject_mismatch_outranks_the_lesser_codes_it_appears_with(self):
        """The consequence, stated as behaviour rather than as list membership. A caption stating a
        number against the wrong subject is rejected for that, not for the bare number it also
        contains."""
        self.assertEqual(["RG_SUBJECT_MISMATCH", "RG_DIRECT_NUMBER_DETECTED"],
                         hdsg._reason_sort(["RG_DIRECT_NUMBER_DETECTED", "RG_SUBJECT_MISMATCH"]))
        self.assertEqual(["RG_SUBJECT_MISMATCH", "RG_INTERNAL_GATE_ERROR"],
                         hdsg._reason_sort(["RG_SUBJECT_MISMATCH", "RG_INTERNAL_GATE_ERROR"]))


if __name__ == "__main__":
    unittest.main()
