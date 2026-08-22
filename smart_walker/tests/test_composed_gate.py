"""The composed caption gate: what it refuses, and what it must not refuse.

Every case here was written against a defect found by reading `hdsg_composed.py` on 22 August 2026,
or against a false rejection found by putting ordinary phrasing through the gate. The reading passes
found six defects in code that had already been read end to end twice, which is the argument for
pinning each one: a check nothing exercises is a check that can be removed without anyone noticing,
and `tests/mutation_audit.py` measures exactly that.

Two obligations are equally weighted. A caption that misstates a measurement must be refused, and a
caption that states one correctly in ordinary English must not be. The second is not a convenience:
a gate that refuses natural phrasing inflates the rejection rate Chapter 5 reports with defects of
the gate rather than behaviour of the model, and the first version of the attribution check refused
four of ten naturally worded truthful captions before that was measured.
"""

from __future__ import annotations

import unittest

from support import DETECTOR_CLASSES, caption, declares, declares_object, detected_object, \
    event, gate, observation, sectors
from scripts import hdsg_composed as composed


# left 3.13, centre 1.74, right 1.16 are the support defaults, and every expectation below is
# written against them rather than against a value computed in the test.
LEFT, CENTRE, RIGHT = 3.13, 1.74, 1.16


class ExactValueTests(unittest.TestCase):
    """A measurement is stated as the deterministic renderer displays it, character for character."""

    def setUp(self):
        self.packet, self.prompt = event()

    def codes(self, text, assertions):
        return gate(caption(text, assertions), self.packet, self.prompt)[0]

    def test_the_displayed_form_is_accepted(self):
        self.assertEqual([], self.codes("The centre is clear for 1.74 metres.",
                                        [declares("centre", CENTRE)]))

    def test_extra_decimals_are_refused(self):
        """1.7449 rounds onto the measurement but is published verbatim, so the user sees precision
        the camera never had. Comparing parsed values rather than written tokens accepted it."""
        self.assertIn("RG_DIRECT_NUMBER_DETECTED",
                      self.codes("The centre is clear for 1.7449 metres.",
                                 [declares("centre", CENTRE)]))

    def test_a_truncated_value_is_refused(self):
        self.assertIn("RG_DIRECT_NUMBER_DETECTED",
                      self.codes("The centre is clear for 1.7 metres.",
                                 [declares("centre", CENTRE)]))

    def test_a_dropped_trailing_zero_is_refused(self):
        """The renderer displays 2.00. A caption saying "2 metres" makes one sensor reading reach
        the user in two forms depending on whether the gate accepted."""
        packet, prompt = event(lane=sectors(centre=2.0))
        codes, _ = gate(caption("The centre is clear for 2 metres.", [declares("centre", 2.0)]),
                        packet, prompt)
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", codes)

    def test_the_padded_form_is_accepted(self):
        packet, prompt = event(lane=sectors(centre=2.0))
        codes, _ = gate(caption("The centre is clear for 2.00 metres.", [declares("centre", 2.0)]),
                        packet, prompt)
        self.assertEqual([], codes)

    def test_a_word_number_is_refused(self):
        """No word is the displayed form of a measurement, and scanning only for digits left
        "roughly five metres" unchecked against a measured 1.74."""
        packet, prompt = event(lane=sectors(centre=2.0))
        codes, _ = gate(caption("The centre is clear for two metres.", [declares("centre", 2.0)]),
                        packet, prompt)
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", codes)

    def test_a_word_number_with_a_half_is_refused(self):
        self.assertIn("RG_DIRECT_NUMBER_DETECTED",
                      self.codes("The centre is clear for one and a half metres.",
                                 [declares("centre", CENTRE)]))

    def test_a_bare_half_is_refused(self):
        self.assertIn("RG_DIRECT_NUMBER_DETECTED",
                      self.codes("The centre is clear for half a metre.",
                                 [declares("centre", CENTRE)]))

    def test_a_word_without_a_unit_is_not_a_measurement(self):
        """"one of the chairs" states no distance. Treating every number word as a measurement
        would refuse ordinary English for no gain."""
        self.assertEqual([], self.codes("The centre is clear for 1.74 metres with no one ahead.",
                                        [declares("centre", CENTRE)]))


class UnitScreenTests(unittest.TestCase):
    """Every unit of length must carry a numeral in front of it.

    The numeral scan sees numerals only, so a distance written without one went through the gate
    undeclared and unrecorded. Detection had been a list of twenty number words followed by a list
    of unit spellings, and a list of spellings is always shorter than English: five phrasings passed
    untouched. Screening the unit catches the class instead of enumerating its members, so a
    phrasing nobody anticipated is refused rather than admitted.
    """

    def setUp(self):
        self.packet, self.prompt = event()

    def codes(self, text):
        return gate(caption(text, [declares("centre", CENTRE)]), self.packet, self.prompt)[0]

    def assertRefused(self, text):
        self.assertEqual(["RG_DIRECT_NUMBER_DETECTED"], self.codes(text), msg=text)

    def test_an_abbreviated_unit_without_a_numeral(self):
        """"m" was not in the unit list, so "two m" was not a distance as far as the gate knew."""
        self.assertRefused("The centre is clear for two m.")

    def test_an_article_standing_in_for_one(self):
        """"a" was not in the number word list."""
        self.assertRefused("The centre is clear for a metre.")

    def test_a_hyphenated_compound(self):
        """The lookahead demanded a space between the number and its unit."""
        self.assertRefused("There is a two-metre gap in the centre.")

    def test_an_imperial_unit(self):
        """The unit list was metric only, so an imperial distance was invisible to it."""
        self.assertRefused("The centre is clear for two feet.")

    def test_a_bare_half_phrased_differently(self):
        """One spelling of a bare half was anticipated and this was not it."""
        self.assertRefused("The centre is clear for half of a metre.")

    def test_a_quantity_word_carrying_no_number(self):
        self.assertRefused("The centre is clear for several centimetres.")

    def test_a_hedged_word_number(self):
        self.assertRefused("It is roughly five metres ahead.")

    def test_a_second_unit_without_its_own_numeral(self):
        """One correctly written distance does not license a second written without one."""
        self.assertRefused("The centre is clear for 1.74 m and a bit more metres.")

    def test_an_abbreviation_joined_to_its_numeral_is_accepted(self):
        self.assertEqual([], self.codes("The centre is clear for 1.74m."))

    def test_an_ordinary_word_containing_a_unit_is_not_a_unit(self):
        """"warm" ends in m and "into" contains in. Admitting the abbreviations only after a digit
        or a space is what keeps them out."""
        self.assertEqual([], composed.unquantified_units("A warm room, walking into the centre."))

    def test_prose_with_no_unit_at_all_is_accepted(self):
        self.assertEqual([], self.codes("The floor is level and the space ahead is quiet."))


class UndeclaredNumberTests(unittest.TestCase):
    """A number the declarations do not account for cannot reach the display."""

    def setUp(self):
        self.packet, self.prompt = event()

    def codes(self, text, assertions=()):
        return gate(caption(text, assertions), self.packet, self.prompt)[0]

    def test_an_invented_number_is_refused(self):
        self.assertIn("RG_DIRECT_NUMBER_DETECTED",
                      self.codes("The way ahead is clear for 9.99 metres."))

    def test_a_colon_does_not_shield_an_invented_number(self):
        """The exclusion for identifiers tested the character before a digit, so a colon disabled
        the check and an entirely fabricated distance reached the display."""
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", self.codes("Clearance:9.99 metres ahead."))

    def test_a_letter_does_not_shield_an_invented_number(self):
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", self.codes("The way ahead is about9.99 metres."))

    def test_an_observation_identifier_is_not_read_as_a_measurement(self):
        """A model copying visual:1 into the prose is a formatting slip, not a hallucination."""
        self.assertEqual([], self.codes("The centre is clear for 1.74 metres, see visual:1.",
                                        [declares("centre", CENTRE)]))

    def test_the_reported_tokens_match_the_code(self):
        """The helper an archive analysis calls and the check the gate runs are one implementation.
        Written twice, they would drift and a run would record a code the analysis disagreed with."""
        text = "The centre is 1.74 metres and the left is 9.99 metres."
        _, scored = gate(caption(text, [declares("centre", CENTRE)]), self.packet, self.prompt)
        self.assertEqual(["9.99"], composed.undeclared_numbers(text, scored))


class DeclarationScoringTests(unittest.TestCase):
    """What the scored assertions record, which is what Chapter 5 reports."""

    def setUp(self):
        self.packet, self.prompt = event()

    def score(self, text, assertions):
        return gate(caption(text, assertions), self.packet, self.prompt)

    def test_an_exact_declaration_agrees(self):
        codes, scored = self.score("The centre is clear for 1.74 metres.",
                                   [declares("centre", CENTRE)])
        self.assertEqual([], codes)
        self.assertEqual("AGREES", scored[0]["outcome"])
        self.assertTrue(scored[0]["exact"])
        self.assertEqual(0.0, scored[0]["absolute_error_m"])

    def test_a_declaration_below_display_precision_disagrees(self):
        """Rounding both sides scored 1.7449 against 1.74 as agreement while recording an error of
        0.005 on the same entry, so the agreement rate was inflated by every sub-centimetre
        departure and the two fields contradicted each other."""
        codes, scored = self.score("The centre is clear for 1.74 metres.",
                                   [declares("centre", 1.7449)])
        self.assertIn("RG_STATED_VALUE_MISMATCH", codes)
        self.assertEqual("DISAGREES", scored[0]["outcome"])
        self.assertFalse(scored[0]["exact"])

    def test_the_error_is_recorded_on_a_refused_declaration(self):
        """The size of a disagreement is the measurement this design exists to produce, and it is
        wanted for a refused candidate as much as an accepted one."""
        _, scored = self.score("The centre is clear for 1.74 metres.", [declares("centre", 2.50)])
        self.assertEqual(0.76, scored[0]["absolute_error_m"])

    def test_a_declaration_the_caption_never_states_is_flagged(self):
        """Declaring a fact the prose does not use widens the set of numbers the prose may contain
        without the model having written anything."""
        _, scored = self.score("The centre is clear for 1.74 metres.",
                               [declares("centre", CENTRE), declares("left", LEFT)])
        self.assertEqual([True, False], [item["stated_in_caption"] for item in scored])
        self.assertEqual(1, composed.assertion_summary(scored)["not_stated_in_caption"])

    def test_a_fact_outside_the_prompt_is_refused(self):
        codes, scored = self.score(
            "The centre is clear for 1.74 metres.",
            [{"fact_id": "sector:rear", "measurement_id": "m:sector:centre:clearance",
              "stated_value": CENTRE}])
        self.assertIn("RG_FACT_REFERENCE_INVALID", codes)
        self.assertEqual("FACT_NOT_PERMITTED", scored[0]["outcome"])

    def test_a_measurement_outside_the_prompt_is_refused(self):
        codes, scored = self.score(
            "The centre is clear for 1.74 metres.",
            [{"fact_id": "sector:centre", "measurement_id": "m:object:9:distance",
              "stated_value": CENTRE}])
        self.assertIn("RG_MEASUREMENT_REFERENCE_INVALID", codes)
        self.assertEqual("MEASUREMENT_NOT_PERMITTED", scored[0]["outcome"])

    def test_a_permitted_measurement_that_resolves_to_nothing_is_refused(self):
        """Defensive against a prompt packet and a fact packet that disagree. The prompt exposes a
        measurement only for a valid sector, so this cannot arise while the two are built from one
        observation; the guard exists for the case where they are not, and nothing exercised it
        until mutation testing showed the line could be deleted unnoticed."""
        packet, prompt = event()
        packet["sectors"]["centre"]["clearance_m"] = None
        codes, scored = gate(caption("The way ahead is open.", [declares("centre", CENTRE)]),
                             packet, prompt)
        self.assertIn("RG_MEASUREMENT_REFERENCE_INVALID", codes)
        self.assertEqual("NOT_MEASURED", scored[0]["outcome"])
        self.assertIsNone(scored[0]["measured_value"])

    def test_a_measurement_belonging_to_another_fact_is_refused(self):
        """The pairing is the whole of the attribution, and checking the permitted facts and the
        permitted measurements as two separate lists let a candidate pair any of one with any of
        the other. Declaring sector:left against m:sector:centre:clearance made 1.74 the left
        sector's measured value as far as every later check was concerned, so "The left side is
        clear for 1.74 metres" was released against a left sector measured at 3.13, with no reason
        code and the declaration scored as agreement. Nothing else in the gate detects it."""
        codes, scored = self.score(
            "The left side is clear for 1.74 metres.",
            [{"fact_id": "sector:left", "measurement_id": "m:sector:centre:clearance",
              "stated_value": CENTRE}])
        self.assertIn("RG_MEASUREMENT_REFERENCE_INVALID", codes)
        self.assertEqual("MEASUREMENT_NOT_FOR_FACT", scored[0]["outcome"])

    def test_an_object_taking_a_sector_measurement_is_refused(self):
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        codes, scored = gate(
            caption("A chair stands 1.74 metres away.",
                    [{"fact_id": "object:3", "measurement_id": "m:sector:centre:clearance",
                      "stated_value": CENTRE}]), packet, prompt)
        self.assertIn("RG_MEASUREMENT_REFERENCE_INVALID", codes)
        self.assertEqual("MEASUREMENT_NOT_FOR_FACT", scored[0]["outcome"])

    def test_a_refused_declaration_does_not_ground_its_number(self):
        """The grounded set was built from every identifier the candidate mentioned, so a
        declaration the gate was about to refuse still licensed its value to appear in the prose.
        The run then recorded the reference error without the undeclared number that came with it,
        and the two are different things a model did."""
        codes, _ = self.score(
            "The left side is clear for 1.74 metres.",
            [{"fact_id": "sector:left", "measurement_id": "m:sector:centre:clearance",
              "stated_value": CENTRE}])
        self.assertIn("RG_DIRECT_NUMBER_DETECTED", codes)

    def test_the_correct_pairing_is_accepted(self):
        self.assertEqual([], self.score("The left side is clear for 3.13 metres.",
                                        [declares("left", LEFT)])[0])

    def test_a_malformed_declaration_still_appears_in_the_count(self):
        """The count of what the model declared is the count of what it declared, including what
        the schema check refused."""
        codes, scored = self.score("The centre is clear for 1.74 metres.",
                                   [declares("centre", CENTRE), {"fact_id": "sector:left"}])
        self.assertIn("RG_SCHEMA_FAILURE", codes)
        self.assertEqual(2, composed.assertion_summary(scored)["declared"])
        self.assertEqual(1, composed.assertion_summary(scored)["comparable"])

    def test_a_boolean_is_not_a_stated_value(self):
        codes, scored = self.score(
            "The centre is clear for 1.74 metres.",
            [{"fact_id": "sector:centre", "measurement_id": "m:sector:centre:clearance",
              "stated_value": True}])
        self.assertIn("RG_SCHEMA_FAILURE", codes)
        self.assertEqual("MALFORMED", scored[0]["outcome"])


class AttributionTests(unittest.TestCase):
    """A number must be attached to the fact it was declared against.

    Every other check constrains a value or a name in isolation, so a caption declaring all three
    clearances correctly and writing each against the wrong sector passed all of them while being
    false in every clause.
    """

    def setUp(self):
        self.packet, self.prompt = event()
        self.all_three = [declares("left", LEFT), declares("centre", CENTRE),
                          declares("right", RIGHT)]

    def codes(self, text, assertions):
        return gate(caption(text, assertions), self.packet, self.prompt)[0]

    def assertAccepted(self, text, assertions):
        self.assertEqual([], self.codes(text, assertions), msg=text)

    def assertRefused(self, text, assertions):
        self.assertIn("RG_SUBJECT_MISMATCH", self.codes(text, assertions), msg=text)

    # The failures the check exists for.

    def test_every_value_moved_one_sector_along(self):
        self.assertRefused(
            "The left side measures 1.16 metres, the centre 3.13 metres and the right 1.74 metres.",
            self.all_three)

    def test_a_single_value_on_the_wrong_side(self):
        self.assertRefused("The left side measures 1.16 metres.", [declares("right", RIGHT)])

    def test_a_swap_behind_a_synonym(self):
        self.assertRefused("There is 1.16 metres of clear space ahead.", [declares("right", RIGHT)])

    def test_a_pairwise_swap(self):
        self.assertRefused("The left gives 1.16 metres and the right gives 3.13 metres.",
                           [declares("left", LEFT), declares("right", RIGHT)])

    def test_a_number_naming_no_fact_at_all(self):
        self.assertRefused("There is 1.74 metres of clear floor.", [declares("centre", CENTRE)])

    def test_an_object_given_another_fact_s_distance(self):
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        codes, _ = gate(caption("A chair is 1.74 metres away on the left.",
                                [declares_object(3, 1.62), declares("centre", CENTRE)]),
                        packet, prompt)
        self.assertIn("RG_SUBJECT_MISMATCH", codes)

    # Ordinary phrasing the check must not refuse. The first implementation cut the caption into
    # clauses on punctuation and coordinating words and refused four of these, because a
    # subordinator is exactly where a subject stops being repeated.

    def test_a_relative_clause_keeps_its_subject(self):
        self.assertAccepted("The centre, which is 1.74 metres, is the widest way through.",
                            [declares("centre", CENTRE)])

    def test_an_appositive_keeps_its_subject(self):
        self.assertAccepted("The left, at 3.13 metres, is open.", [declares("left", LEFT)])

    def test_right_in_front_names_the_centre_not_the_right(self):
        """The longest match wins, so the "right" inside "right in front" is not the right sector.
        Without that rule this truthful caption was refused as ambiguous."""
        self.assertAccepted("There is 1.74 metres of space right in front of the walker.",
                            [declares("centre", CENTRE)])

    def test_a_fact_named_in_the_next_sentence_does_not_compete(self):
        """"The left is open for 3.13 metres. The right is tighter at 1.16 metres." put 3.13
        exactly as far from "left" as from "right", so a truthful caption was refused because of a
        word in the sentence after it."""
        self.assertAccepted(
            "The left is open for 3.13 metres. The right is tighter at 1.16 metres.",
            [declares("left", LEFT), declares("right", RIGHT)])

    def test_a_comma_list_of_all_three(self):
        self.assertAccepted("Left 3.13 metres, centre 1.74 metres, right 1.16 metres.",
                            self.all_three)

    def test_a_prose_list_of_all_three(self):
        self.assertAccepted(
            "The left gives 3.13 metres, the centre 1.74 metres, and the right 1.16 metres.",
            self.all_three)

    def test_a_number_before_its_fact(self):
        self.assertAccepted("1.74 metres of clear floor lies in the centre.",
                            [declares("centre", CENTRE)])

    def test_a_parenthetical_measurement(self):
        self.assertAccepted("The left side (3.13 metres) is the most open.",
                            [declares("left", LEFT)])

    def test_a_semicolon_separates_two_facts(self):
        self.assertAccepted("The centre gives 1.74 metres; the right gives 1.16 metres.",
                            [declares("centre", CENTRE), declares("right", RIGHT)])

    def test_an_object_and_a_sector_together(self):
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        codes, _ = gate(caption("A chair is 1.62 metres away, and the centre is clear "
                                "for 1.74 metres.",
                                [declares_object(3, 1.62), declares("centre", CENTRE)]),
                        packet, prompt)
        self.assertEqual([], codes)

    # The required form is the prompt's: name the thing, then give its distance, with no other place
    # or object named in between. A caption that puts a bearing between an object and its distance
    # does not follow it, and is refused and recorded as such rather than rescued. Two rules were
    # written to admit those captions and both were wrong, which is why there are none now.

    def test_the_required_form_for_an_object_is_accepted(self):
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        codes, _ = gate(caption("A chair is 1.62 metres away on the left.",
                                [declares_object(3, 1.62)]), packet, prompt)
        self.assertEqual([], codes)

    def test_a_bearing_between_an_object_and_its_distance_is_refused(self):
        """True, and not in the form the prompt asks for. Recorded as FORM_NOT_FOLLOWED so the rate
        at which the model ignores the instruction can be reported apart from the rate at which it
        states something false."""
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        text = "The chair on the left is 1.62 metres away."
        codes, scored = gate(caption(text, [declares_object(3, 1.62)]), packet, prompt)
        self.assertIn("RG_SUBJECT_MISMATCH", codes)
        self.assertEqual(["FORM_NOT_FOLLOWED"],
                         [f["reason"] for f in composed.attribution_failures(text, scored, packet)])

    def test_a_fact_never_named_is_misattribution_rather_than_form(self):
        """The two findings are kept apart. This sentence never mentions the chair, so the caption
        states a measured value of something it does not name."""
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        text = "There is 1.62 metres to the left."
        codes, scored = gate(caption(text, [declares_object(3, 1.62)]), packet, prompt)
        self.assertIn("RG_SUBJECT_MISMATCH", codes)
        self.assertEqual(["MISATTRIBUTED"],
                         [f["reason"] for f in composed.attribution_failures(text, scored, packet)])

    def test_a_sector_named_through_a_bearing_phrase_is_accepted(self):
        """A bearing phrase is a perfectly good way to name a sector, and the nearest name here is
        the right one."""
        self.assertAccepted("There is 3.13 metres on the left.", [declares("left", LEFT)])

    def test_the_frozen_fixture_caption_is_accepted(self):
        """A bearing phrase is the nearest and correct name for 3.13 here, while "ahead" is 58
        characters away. A rule that demoted bearing phrases refused this, and no test caught it."""
        self.assertAccepted(
            "The way ahead narrows to 1.74 metres, with 3.13 metres of space to the left.",
            [declares("centre", CENTRE), declares("left", LEFT)])

    def test_a_sector_taking_an_object_distance_is_refused(self):
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        codes, _ = gate(caption("The left is 1.62 metres wide.", [declares_object(3, 1.62)]),
                        packet, prompt)
        self.assertIn("RG_SUBJECT_MISMATCH", codes)

    def test_a_sentence_ending_in_a_number_still_ends(self):
        """The full stop after 3.13 has a digit in front of it. Excluding a boundary on that basis
        made the whole caption one sentence, so every fact named anywhere in it competed."""
        self.assertAccepted("The left is 3.13. The centre is 1.74.",
                            [declares("left", LEFT), declares("centre", CENTRE)])

    def test_a_swap_across_a_sentence_ending_in_a_number(self):
        self.assertRefused("The left is 1.16. The centre is 1.74.",
                           [declares("right", RIGHT), declares("centre", CENTRE)])

    def test_two_objects_of_one_class_are_one_mention_of_both(self):
        """A label is shared, so "the chair" in a room with two chairs names both. Resolving that
        as an overlap kept whichever was found first and discarded the other, so a caption
        describing the second was refused for naming the first."""
        from scripts import hdsg_runtime as hdsg
        objects = hdsg.normalise_objects([
            {"id": 0, "track_id": 3, "raw_label": "chair", "canonical_class": "chair",
             "ontology_class": "furniture", "conf": 0.8, "bbox_xyxy": [10, 10, 90, 200],
             "bearing": "LEFT", "distance_m": 1.62,
             "distance_method": "D455F_LOWER_BBOX_MEDIAN"},
            {"id": 1, "track_id": 5, "raw_label": "chair", "canonical_class": "chair",
             "ontology_class": "furniture", "conf": 0.8, "bbox_xyxy": [10, 10, 90, 200],
             "bearing": "RIGHT", "distance_m": 2.40,
             "distance_method": "D455F_LOWER_BBOX_MEDIAN"}])
        packet = event(objects=objects)[0]
        mentions = composed._fact_mentions("A chair stands there.", packet)
        self.assertEqual(1, len(mentions))
        self.assertEqual({"object:3", "object:5"}, set(mentions[0].facts))

    def test_a_quoted_identifier_does_not_name_a_fact(self):
        """Quoting sector:centre is not describing a place, so it must not satisfy the check."""
        self.assertRefused("A clearance of 1.74 metres is recorded at sector:centre.",
                           [declares("centre", CENTRE)])


class ProhibitedContentTests(unittest.TestCase):
    """What a caption may not contain, whatever its numbers say."""

    def setUp(self):
        self.packet, self.prompt = event()

    def codes(self, text, assertions=()):
        return gate(caption(text, assertions), self.packet, self.prompt)[0]

    def test_an_instruction_is_refused(self):
        """The deterministic tuple is the guidance. Generated prose may describe but never direct."""
        self.assertIn("RG_ACTION_LANGUAGE_DETECTED", self.codes("Turn towards the wider side."))

    def test_commentary_about_the_system_is_refused(self):
        self.assertIn("RG_MODEL_COMMENTARY_DETECTED", self.codes("The image shows an open room."))

    def test_reading_out_a_sign_is_refused(self):
        self.assertIn("RG_VISIBLE_TEXT_CONTENT_DETECTED",
                      self.codes('A sign reads "exit" beside the opening.'))

    def test_a_possessive_is_not_visible_text(self):
        """A bare apostrophe was matched as a quotation mark, so ordinary prose was refused as
        reading out a sign. A templated caption never carried a possessive; a composed one does."""
        self.assertEqual([], self.codes("The room's centre is clear for 1.74 metres.",
                                        [declares("centre", CENTRE)]))

    def test_a_class_the_detector_did_not_report_is_refused(self):
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", self.codes("A chair stands in the doorway."))

    def test_the_plural_of_an_absent_class_is_refused(self):
        """A suffix rule misses "people", and person is the class a walker is most often wrong
        about: "two people" went unchecked while "a person" was refused."""
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", self.codes("Two people stand nearby."))

    def test_a_sibilant_plural_is_refused(self):
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", self.codes("Several buses are parked here."))

    def test_an_f_stem_plural_is_refused(self):
        """The suffix rules produce "shelfs" and leave "shelves" unguarded. The Open Images
        vocabulary adopted on 22 August 2026 contains Shelf, Scarf, Man, Woman, Goose, Deer, Potato
        and Tomato, none of which the rules pluralise correctly, so each is tabled explicitly."""
        packet, prompt = event()
        codes, _ = gate(caption("Shelves line the wall."), packet, prompt,
                        classes=("Shelf", "Chair"))
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", codes)

    def test_the_plural_of_man_is_refused(self):
        packet, prompt = event()
        codes, _ = gate(caption("Two men are waiting."), packet, prompt, classes=("Man", "Chair"))
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", codes)

    def test_the_hazard_class_may_not_be_named_when_absent(self):
        """Stairs is the class the whole hazard argument rests on, and it became nameable only when
        the detector vocabulary changed to Open Images. COCO had no word for it."""
        packet, prompt = event()
        codes, _ = gate(caption("Stairs lead down ahead."), packet, prompt,
                        classes=("Stairs", "Chair"))
        self.assertIn("RG_OBJECT_REFERENCE_INVALID", codes)

    def test_a_reported_class_may_be_named(self):
        packet, prompt = event(objects=detected_object(track_id=3, label="chair",
                                                       bearing="LEFT", distance_m=1.62))
        codes, _ = gate(caption("A chair stands 1.62 metres away on the left.",
                                [declares_object(3, 1.62)]), packet, prompt)
        self.assertEqual([], codes)

    def test_a_word_outside_the_detector_vocabulary_is_permitted(self):
        """The check bounds what the model may assert about recognised objects; it does not police
        ordinary language."""
        self.assertEqual([], self.codes("The floor is level and the space is quiet."))

    def test_the_detector_vocabulary_has_no_default(self):
        """An empty vocabulary turns the object check into a no-operation, and a property holding
        by construction must not switch itself off because a caller left an argument out."""
        with self.assertRaises(TypeError):
            composed.validate_caption_candidate(
                caption("A chair stands nearby."), self.prompt, self.packet)


class VisualObservationTests(unittest.TestCase):
    """A visual label is rendered into the release as prose, so it is screened as prose."""

    def setUp(self):
        self.packet, self.prompt = event()

    def codes(self, *visuals):
        return gate(caption("The centre is clear for 1.74 metres.",
                            [declares("centre", CENTRE)], visuals),
                    self.packet, self.prompt)[0]

    def test_a_plain_label_is_accepted(self):
        self.assertEqual([], self.codes(observation("visual:1", "doorway", "LEFT")))

    def test_a_label_naming_an_absent_class_is_refused(self):
        """Compared by equality, a forbidden class inside a longer label passed unnoticed while the
        same class alone was refused."""
        self.assertIn("RG_OBJECT_REFERENCE_INVALID",
                      self.codes(observation("visual:1", "glass doors", "LEFT")))

    def test_a_label_carrying_an_instruction_is_refused(self):
        self.assertIn("RG_UNAPPROVED_LANGUAGE_DETECTED",
                      self.codes(observation("visual:1", "follow the wall", "LEFT")))

    def test_a_label_carrying_a_distance_is_refused(self):
        self.assertIn("RG_UNAPPROVED_LANGUAGE_DETECTED",
                      self.codes(observation("visual:1", "gap two metres wide", "LEFT")))

    def test_a_duplicate_identifier_is_refused(self):
        self.assertIn("RG_SCHEMA_FAILURE",
                      self.codes(observation("visual:1", "doorway", "LEFT"),
                                 observation("visual:1", "step", "RIGHT")))

    def test_a_zero_identifier_is_refused(self):
        self.assertIn("RG_SCHEMA_FAILURE", self.codes(observation("visual:0", "doorway", "LEFT")))

    def test_an_unknown_bearing_is_refused(self):
        self.assertIn("RG_SCHEMA_FAILURE", self.codes(observation("visual:1", "doorway", "REAR")))

    def test_more_observations_than_the_profile_allows(self):
        self.assertIn("RG_PROFILE_LIMIT_EXCEEDED",
                      self.codes(observation("visual:1", "doorway", "LEFT"),
                                 observation("visual:2", "step", "RIGHT"),
                                 observation("visual:3", "ramp", "CENTRE")))

    def test_observations_where_the_profile_forbids_them(self):
        packet, prompt = event(response_mode="AUTOMATIC")
        codes, _ = gate(caption("The centre is clear for 1.74 metres.",
                                [declares("centre", CENTRE)],
                                [observation("visual:1", "doorway", "LEFT")]),
                        packet, prompt)
        self.assertIn("RG_PROFILE_LIMIT_EXCEEDED", codes)

    def test_the_permission_flag_is_enforced_on_its_own(self):
        """The only profile that forbids observations also caps them at zero, so the test above is
        satisfied by the count guard and leaves the permission guard unexercised. Mutation testing
        found it: deleting the permission check changed nothing. The constraints are set directly
        here so the flag is the only thing refusing the caption."""
        packet, prompt = event()
        prompt["response_constraints"] = {**prompt["response_constraints"],
                                          "max_visual_observations": 2,
                                          "visual_only_observations_allowed": False}
        codes, _ = gate(caption("The centre is clear for 1.74 metres.",
                                [declares("centre", CENTRE)],
                                [observation("visual:1", "doorway", "LEFT")]),
                        packet, prompt)
        self.assertEqual(["RG_PROFILE_LIMIT_EXCEEDED"], codes)

    def test_a_visual_observation_missing_a_key_is_refused(self):
        codes, _ = gate(caption("The centre is clear for 1.74 metres.",
                                [declares("centre", CENTRE)],
                                [{"candidate_observation_id": "visual:1",
                                  "proposed_label": "doorway"}]),
                        self.packet, self.prompt)
        self.assertEqual(["RG_SCHEMA_FAILURE"], codes)

    def test_a_label_outside_the_permitted_characters_is_refused(self):
        """The instruction and distance tests above both use labels that satisfy the character
        pattern and are caught further down the chain, so neither exercises the pattern itself. A
        capital letter reaches only this guard."""
        codes, _ = gate(caption("The centre is clear for 1.74 metres.",
                                [declares("centre", CENTRE)],
                                [observation("visual:1", "Stairs", "LEFT")]),
                        self.packet, self.prompt)
        self.assertEqual(["RG_UNAPPROVED_LANGUAGE_DETECTED"], codes)


class SchemaShapeTests(unittest.TestCase):
    """The reply must be the record the contract describes, without repair."""

    def setUp(self):
        self.packet, self.prompt = event()

    def test_a_non_json_reply_does_not_parse(self):
        self.assertEqual((None, ["RG_PARSE_FAILURE"]),
                         composed.parse_caption_candidate("not json at all"))

    def test_a_json_array_is_not_a_candidate(self):
        self.assertEqual((None, ["RG_SCHEMA_FAILURE"]), composed.parse_caption_candidate("[1, 2]"))

    def test_an_extra_key_is_refused(self):
        record = caption("The centre is clear for 1.74 metres.", [declares("centre", CENTRE)])
        record["confidence"] = 0.9
        self.assertIn("RG_SCHEMA_FAILURE", gate(record, self.packet, self.prompt)[0])

    def test_the_wrong_schema_version_is_refused(self):
        record = caption("The centre is clear for 1.74 metres.", [declares("centre", CENTRE)])
        record["schema_version"] = "hdsg.vlm_candidate.v1"
        self.assertIn("RG_SCHEMA_FAILURE", gate(record, self.packet, self.prompt)[0])

    def test_an_empty_caption_is_refused(self):
        self.assertIn("RG_SCHEMA_FAILURE", gate(caption("   "), self.packet, self.prompt)[0])

    def test_a_caption_longer_than_the_profile_allows(self):
        long_text = "The centre is clear for 1.74 metres. " + ("Open floor lies around it. " * 20)
        self.assertIn("RG_PROFILE_LIMIT_EXCEEDED",
                      gate(caption(long_text, [declares("centre", CENTRE)]),
                           self.packet, self.prompt)[0])


class ReleaseTests(unittest.TestCase):
    """What an accepted caption becomes, and what it must not disturb."""

    def setUp(self):
        self.packet, self.prompt = event()
        self.candidate = caption("The centre is clear for 1.74 metres.",
                                 [declares("centre", CENTRE)])
        self.codes, self.scored = gate(self.candidate, self.packet, self.prompt)
        self.assertEqual([], self.codes)

    def release(self, **kwargs):
        return composed.build_composed_release(
            self.packet, self.prompt, release_id="release_1", candidate=self.candidate,
            scored_assertions=self.scored, **kwargs)

    def test_the_action_text_comes_from_the_deterministic_layer(self):
        """The action the user acts on is unaffected by what the model wrote."""
        from scripts import hdsg_runtime as hdsg
        fallback = hdsg.build_release(self.packet, self.prompt, release_id="release_0",
                                      failure_codes=["RG_MODEL_UNAVAILABLE"])
        self.assertEqual(fallback["content"]["action_text"],
                         self.release()["content"]["action_text"])

    def test_the_caption_becomes_the_reason_text(self):
        self.assertEqual("The centre is clear for 1.74 metres.",
                         self.release()["content"]["reason_text"])

    def test_the_scene_binding_names_the_declared_facts(self):
        """Inherited from the base release, the binding named the deterministic sentence that was
        composed and then discarded: a caption about the centre carried sector:right as evidence."""
        self.assertEqual(["sector:centre"], self.release()["evidence"]["scene_binding_fact_ids"])

    def test_a_repeated_measurement_is_recorded_once(self):
        """The release schema requires unique substitutions, and a caption may legitimately state
        one measurement twice."""
        record = caption("The centre is clear for 1.74 metres, a full 1.74 metres of floor.",
                         [declares("centre", CENTRE), declares("centre", CENTRE)])
        codes, scored = gate(record, self.packet, self.prompt)
        self.assertEqual([], codes)
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_2", candidate=record,
            scored_assertions=scored)
        self.assertEqual(1, len(release["evidence"]["measurement_substitutions"]))

    def test_a_refused_candidate_falls_back(self):
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_3", candidate=self.candidate,
            scored_assertions=self.scored, failure_codes=["RG_SUBJECT_MISMATCH"])
        self.assertEqual("DETERMINISTIC_FALLBACK", release["verification"]["release_mode"])
        # The fallback states the same measurement, which is the point of it. What must not survive
        # is the model's wording.
        self.assertNotEqual(self.candidate["caption"], release["content"]["reason_text"])
        self.assertEqual(["RG_SUBJECT_MISMATCH"], release["verification"]["reason_codes"])

    def test_a_singular_label_takes_a_singular_verb(self):
        """Testing only for a trailing "s" produced "Possible bus are visible in the left"."""
        record = caption("The centre is clear for 1.74 metres.", [declares("centre", CENTRE)],
                         [observation("visual:1", "glass", "LEFT")])
        codes, scored = gate(record, self.packet, self.prompt)
        self.assertEqual([], codes)
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_4", candidate=record,
            scored_assertions=scored)
        self.assertEqual(["Possible glass is visible in the left."],
                         release["content"]["additional_detail_texts"])

    def test_a_plural_label_takes_a_plural_verb(self):
        record = caption("The centre is clear for 1.74 metres.", [declares("centre", CENTRE)],
                         [observation("visual:1", "handrails", "LEFT")])
        codes, scored = gate(record, self.packet, self.prompt)
        self.assertEqual([], codes)
        release = composed.build_composed_release(
            self.packet, self.prompt, release_id="release_5", candidate=record,
            scored_assertions=scored)
        self.assertEqual(["Possible handrails are visible in the left."],
                         release["content"]["additional_detail_texts"])

    def test_the_scored_assertions_have_no_default(self):
        """A caller omitting them produced a release recording no scene binding and no measurement
        substitutions for a caption resting on both, and nothing failed."""
        with self.assertRaises(TypeError):
            composed.build_composed_release(
                self.packet, self.prompt, release_id="release_6", candidate=self.candidate)


class FrozenFixtureTests(unittest.TestCase):
    """The caption in the frozen schema fixture must still pass the gate.

    Nothing exercised this. A change to the attribution check refused the fixture, 125 tests passed,
    and only `schemas/generate_fixtures.py` noticed, because it refuses to write a fixture the gate
    would reject. That guard runs when somebody thinks to run it. This runs every time.

    The fixture is the record every schema conformance claim rests on, so a gate that would refuse
    it means the frozen set no longer describes what the runtime can produce.
    """

    def test_the_frozen_caption_passes_the_gate(self):
        import json
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        record = json.loads((root / "schemas" / "fixtures" / "valid"
                             / "hdsg.vlm_caption.v1.json").read_text(encoding="utf-8"))
        fact_packet = json.loads((root / "schemas" / "fixtures" / "valid"
                                  / "hdsg.fact_packet.v2.json").read_text(encoding="utf-8"))
        prompt_packet = json.loads((root / "schemas" / "fixtures" / "valid"
                                    / "hdsg.prompt_packet.v2.json").read_text(encoding="utf-8"))
        codes, scored = composed.validate_caption_candidate(
            record, prompt_packet, fact_packet, detector_classes=DETECTOR_CLASSES)
        self.assertEqual([], codes)
        self.assertTrue(scored)
        self.assertTrue(all(item["outcome"] == "AGREES" for item in scored))


class PromptTests(unittest.TestCase):
    """What the model is shown, which must be what the gate then holds it to."""

    def setUp(self):
        self.packet, self.prompt = event()

    def test_measurements_are_shown_in_the_form_the_gate_requires(self):
        """The gate compares the written token against the renderer's string, so the prompt has to
        present that same string or the model is being asked for one form and judged on another."""
        text = composed.build_composed_prompt(self.prompt, self.packet)
        self.assertIn("1.74 metres", text)
        self.assertIn("m:sector:centre:clearance", text)

    def test_a_whole_metre_is_shown_padded(self):
        packet, prompt = event(lane=sectors(centre=2.0))
        self.assertIn("2.00 metres", composed.build_composed_prompt(prompt, packet))


if __name__ == "__main__":
    unittest.main()
