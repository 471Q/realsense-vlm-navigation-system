"""Tests for the open question routing pipeline.

Covers the four stages of HDSG_OPEN_QUESTION_ROUTING_POLICY.md section 2: the measurement
pre-check, the Tier 0 keyword filter, the Tier 1 reply parser, and route-scoped fact selection.
"""

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import hdsg_questions as questions  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402


def make_sectors(left="CLEAR", centre="CLEAR", right="CLEAR", valid=True):
    values = {"CLEAR": 2.4, "CONSTRAINED": 1.4, "BLOCKED": 0.5, "UNKNOWN": None}
    return {
        name: {
            "fact_id": f"sector:{name}",
            "clearance_m": values[status] if valid and status != "UNKNOWN" else None,
            "valid": bool(valid and status != "UNKNOWN"),
            "invalid_reason_codes": [] if valid and status != "UNKNOWN" else ["DEPTH_SECTOR_INVALID"],
            "status": status,
        }
        for name, status in (("left", left), ("centre", centre), ("right", right))
    }


def make_packet(sectors=None, objects=(), action_ids=("sector:centre",),
                scene_ids=(), scene_required=False, selected="CENTRE", decision="PROCEED"):
    return {
        "identity": {"event_id": "evt_1", "observation_id": "obs_1"},
        "observation": {"rgb_ref": "memory://obs_1/rgb", "depth_ref": "memory://obs_1/depth"},
        "interaction": {"request_id": "MORE_DETAIL", "response_mode": "MORE_DETAIL"},
        "objects": list(objects),
        "sectors": sectors if sectors is not None else make_sectors(),
        "deterministic": {
            "motion_decision": decision,
            "selected_sector": selected,
            "scene_advisory": "SAFE",
            "interaction_state": "GUIDANCE_ACTIVE",
            "scene_fact_required": scene_required,
            "action_binding": {"accepted_fact_ids": list(action_ids)},
            "scene_binding": {"accepted_fact_ids": list(scene_ids)},
        },
    }


def make_object(fact_id="object:7", bearing="LEFT", hazard=False, label="chair"):
    return {
        "fact_id": fact_id,
        "raw_label": label,
        "canonical_label": label,
        "bearing": bearing,
        "distance_m": 1.8,
        "distance_valid": True,
        "distance_bin": "NEAR",
        "grounding": "YOLO_D455F",
        "is_hazard": hazard,
        "motion_state": "STATIONARY",
    }


class MeasurementPreCheckTests(unittest.TestCase):
    def test_fresh_valid_measurement_is_answerable(self):
        self.assertTrue(questions.measurement_is_answerable(make_sectors(), 1200.0, 5000.0))

    def test_no_valid_sector_is_not_answerable(self):
        sectors = make_sectors("UNKNOWN", "UNKNOWN", "UNKNOWN", valid=False)
        self.assertFalse(questions.measurement_is_answerable(sectors, 100.0, 5000.0))

    def test_stale_observation_is_not_answerable(self):
        self.assertFalse(questions.measurement_is_answerable(make_sectors(), 9000.0, 5000.0))

    def test_absent_age_is_not_answerable(self):
        # An unknown age cannot be shown to be fresh, and the pre-check must not assume it is.
        self.assertFalse(questions.measurement_is_answerable(make_sectors(), None, 5000.0))


class Tier0KeywordTests(unittest.TestCase):
    def test_bearing_keywords(self):
        self.assertEqual(questions.classify_keywords("what is on my left"), "LEFT")
        self.assertEqual(questions.classify_keywords("anything to the right?"), "RIGHT")
        self.assertEqual(questions.classify_keywords("what is straight ahead"), "CENTRE")

    def test_why_question_about_a_sector_routes_to_the_decision(self):
        # Section 3: order matters where keywords overlap. A why-question about a sector is still
        # a question about the decision.
        self.assertEqual(questions.classify_keywords("why is the left blocked"), "EXPLAIN_DECISION")

    def test_a_named_side_beats_a_hazard_word(self):
        # The bearing route carries the sector state and every object bearing it, hazards among
        # them. HAZARDS scopes to hazard objects anywhere, so answering this from it could
        # describe a spill on the right.
        self.assertEqual(questions.classify_keywords("is it safe on the left"), "LEFT")
        self.assertEqual(questions.classify_keywords("any danger on the right"), "RIGHT")

    def test_hazard_words_route_to_hazards_when_no_side_is_named(self):
        self.assertEqual(questions.classify_keywords("is it safe"), "HAZARDS")
        self.assertEqual(questions.classify_keywords("anything in my way"), "HAZARDS")

    def test_more_detail_phrasings_reach_the_scene_overview(self):
        for phrase in ("tell me more about the scene", "tell me more", "more detail",
                       "give me more information", "what's going on", "describe the scene"):
            self.assertEqual(questions.classify_keywords(phrase), "SCENE_OVERVIEW", phrase)

    def test_a_named_side_beats_a_more_detail_phrasing(self):
        self.assertEqual(questions.classify_keywords("tell me more about the left"), "LEFT")

    def test_plural_keywords_match(self):
        self.assertEqual(questions.classify_keywords("any obstacles"), "HAZARDS")
        self.assertEqual(questions.classify_keywords("any hazards"), "HAZARDS")

    def test_unmatched_question_falls_through(self):
        self.assertIsNone(questions.classify_keywords("how tall is the building"))

    def test_off_topic_questions_are_never_guessed_at(self):
        # Bare "what" is excluded from SCENE_OVERVIEW for these. Answering them with a description
        # of the room would be worse than sending them to the classifier to be declined.
        for phrase in ("what time is it", "what is your name", "tell me a joke",
                       "who won the football", "what's the weather"):
            self.assertIsNone(questions.classify_keywords(phrase), phrase)

    def test_object_questions_without_a_side_reach_the_classifier(self):
        # Section 4 folds object questions into the bearing routes, and resolving which bearing an
        # object is on is inference the keyword filter cannot perform.
        for phrase in ("how far is the chair", "what is that", "is there a door"):
            self.assertIsNone(questions.classify_keywords(phrase), phrase)

    def test_substring_does_not_match_a_whole_word_keyword(self):
        # "leftover" contains "left". Matching it as the LEFT bucket would misroute the question.
        self.assertIsNone(questions.classify_keywords("any leftover items"))

    def test_multi_word_keyword_matches(self):
        self.assertEqual(questions.classify_keywords("please look again"), "REASSESS")

    def test_every_bucket_names_a_real_route(self):
        for route, keywords in questions.TIER0_KEYWORDS:
            self.assertIn(route, questions.ROUTES)
            self.assertTrue(keywords)


class RouteParsingTests(unittest.TestCase):
    def test_valid_reply(self):
        raw = json.dumps({"schema_version": questions.ROUTE_SCHEMA, "route": "HAZARDS"})
        self.assertEqual(questions.parse_route(raw), ("HAZARDS", None))

    def test_unknown_route_is_rejected(self):
        raw = json.dumps({"schema_version": questions.ROUTE_SCHEMA, "route": "OBJECT_QUERY"})
        route, error = questions.parse_route(raw)
        self.assertIsNone(route)
        self.assertIn("OBJECT_QUERY", error)

    def test_wrong_schema_is_rejected(self):
        raw = json.dumps({"schema_version": "something.else", "route": "LEFT"})
        self.assertIsNone(questions.parse_route(raw)[0])

    def test_non_json_is_rejected(self):
        self.assertIsNone(questions.parse_route("LEFT")[0])


class RouteRequirementTests(unittest.TestCase):
    def test_bearing_route_scopes_to_its_sector(self):
        packet = make_packet()
        requirements = questions.route_requirements("LEFT", packet)
        self.assertEqual(requirements[0]["fact_ids"], ["sector:left"])

    def test_bearing_route_includes_objects_on_that_side_only(self):
        packet = make_packet(objects=[
            make_object("object:1", "LEFT"),
            make_object("object:2", "RIGHT"),
        ])
        fact_ids = [f for item in questions.route_requirements("LEFT", packet)
                    for f in item["fact_ids"]]
        self.assertIn("object:1", fact_ids)
        self.assertNotIn("object:2", fact_ids)

    def test_each_requirement_names_exactly_one_fact(self):
        # A requirement spanning several facts under ANY_OF is satisfied by naming only one of
        # them, which is the terseness the answer path exists to avoid.
        packet = make_packet(objects=[make_object("object:1", "LEFT"), make_object("object:2", "LEFT")])
        for item in questions.route_requirements("LEFT", packet):
            self.assertEqual(len(item["fact_ids"]), 1)

    def test_requirements_never_exceed_the_schema_cap(self):
        packet = make_packet(objects=[make_object(f"object:{n}", "LEFT") for n in range(8)])
        self.assertLessEqual(len(questions.route_requirements("LEFT", packet)), 3)

    def test_hazard_route_selects_hazard_objects(self):
        packet = make_packet(objects=[
            make_object("object:1", "LEFT", hazard=False),
            make_object("object:2", "RIGHT", hazard=True),
        ])
        fact_ids = [f for item in questions.route_requirements("HAZARDS", packet)
                    for f in item["fact_ids"]]
        self.assertIn("object:2", fact_ids)
        self.assertNotIn("object:1", fact_ids)

    def test_hazard_route_with_no_hazard_still_answers_from_a_sector(self):
        # "Is it safe" on a benign scene is answerable from the sector states. Declining would be
        # a worse answer than a truthful one.
        packet = make_packet(sectors=make_sectors("CLEAR", "BLOCKED", "CLEAR"))
        requirements = questions.route_requirements("HAZARDS", packet)
        self.assertEqual(requirements[0]["fact_ids"], ["sector:centre"])

    def test_explain_decision_uses_the_action_binding(self):
        packet = make_packet(action_ids=["sector:centre", "object:3"], decision="STOP")
        requirements = questions.route_requirements("EXPLAIN_DECISION", packet)
        self.assertEqual(requirements[0]["role"], "ACTION_BINDING")
        self.assertEqual(requirements[0]["match"], "ALL_OF")
        self.assertEqual(requirements[0]["fact_ids"], ["sector:centre", "object:3"])

    def test_explain_decision_adds_the_scene_binding_when_required(self):
        packet = make_packet(action_ids=["sector:centre"], scene_ids=["sector:left"],
                             scene_required=True)
        roles = [item["role"] for item in questions.route_requirements("EXPLAIN_DECISION", packet)]
        self.assertIn("SCENE_BINDING", roles)

    def test_explain_decision_on_an_unbound_proceed_falls_back_to_the_selected_sector(self):
        # PROCEED on a clear sector records no binding fact. The honest answer to "why" is that
        # sector's state rather than silence.
        packet = make_packet(action_ids=[], selected="RIGHT")
        requirements = questions.route_requirements("EXPLAIN_DECISION", packet)
        self.assertEqual(requirements[0]["fact_ids"], ["sector:right"])

    def test_terminal_routes_produce_no_requirements(self):
        packet = make_packet()
        for route in ("OUT_OF_SCOPE", "REASSESS"):
            self.assertEqual(questions.route_requirements(route, packet), [])


class PromptPacketIntegrationTests(unittest.TestCase):
    """The routed requirements must produce a packet the frozen v1 schema accepts."""

    def build(self, route, packet):
        return hdsg.build_prompt_packet(
            packet,
            prompt_id="prompt_1",
            model_id="qwen3-vl-4b-instruct",
            model_hash=hdsg.sha256_text("model"),
            quantisation="Q4_K_M",
            temperature=0.2,
            top_p=0.9,
            max_tokens=400,
            system_prompt="system",
            constraint_hash=hdsg.sha256_text("grammar"),
            prompt_profile_id=questions.question_profile_id(route),
            question_requirements=questions.route_requirements(route, packet),
        )

    def test_routed_packet_carries_only_the_routed_facts(self):
        packet = make_packet(objects=[make_object("object:1", "RIGHT")],
                             action_ids=["sector:centre"])
        built = self.build("LEFT", packet)
        permitted = [item["fact_id"] for item in built["permitted_facts"]]
        self.assertEqual(permitted, ["sector:left"])
        # The action binding is deliberately absent: an answer about the left sector should
        # describe the left sector, not the fact binding the current action.
        self.assertNotIn("sector:centre", permitted)

    def test_routed_profile_identifier_matches_the_schema_pattern(self):
        packet = make_packet()
        built = self.build("EXPLAIN_DECISION", packet)
        self.assertEqual(built["routing"]["prompt_profile_id"], "question_explain_decision.v1")
        self.assertRegex(built["routing"]["prompt_profile_id"], r"^[a-z][a-z0-9._-]{1,127}$")

    def test_routed_packet_keeps_the_frozen_request_enumerations(self):
        built = self.build("LEFT", make_packet())
        self.assertEqual(built["routing"]["request_id"], "MORE_DETAIL")
        self.assertEqual(built["routing"]["response_mode"], "MORE_DETAIL")

    def test_permitted_facts_carry_approved_templates(self):
        built = self.build("LEFT", make_packet())
        self.assertTrue(built["permitted_facts"][0]["approved_text_templates"])

    def test_automatic_construction_is_unchanged_when_no_route_is_given(self):
        packet = make_packet(action_ids=["sector:centre"])
        packet["interaction"]["request_id"] = "AUTO_GUIDANCE"
        packet["interaction"]["response_mode"] = "AUTOMATIC"
        built = hdsg.build_prompt_packet(
            packet,
            prompt_id="prompt_1",
            model_id="m", model_hash=hdsg.sha256_text("m"), quantisation=None,
            temperature=0.2, top_p=0.9, max_tokens=400,
            system_prompt="system", constraint_hash=hdsg.sha256_text("g"),
        )
        self.assertEqual(built["routing"]["prompt_profile_id"], "guidance_reason.v1")
        self.assertEqual(built["requirements"][0]["role"], "ACTION_BINDING")


class AnswerTextTests(unittest.TestCase):
    def test_answer_joins_the_reason_and_the_details(self):
        release = {"content": {
            "action_text": "Stop.",
            "reason_text": "The left sector is clear for 2.40 metres.",
            "interaction_text": None,
            "additional_detail_texts": ["A chair is detected on the left at 1.60 metres."],
            "caption_text": "Stop. The left sector is clear for 2.40 metres. A chair is detected on the left at 1.60 metres.",
        }}
        answer = questions.answer_text_from_release(release)
        self.assertIn("left sector is clear", answer)
        self.assertIn("chair is detected", answer)

    def test_answer_excludes_the_action_text(self):
        # "Stop." belongs on the guidance line. Prefixed to a reply about the left sector it would
        # read as an instruction the user did not ask for.
        release = {"content": {
            "action_text": "Stop.", "reason_text": "The left sector is clear for 2.40 metres.",
            "interaction_text": "Select left or right.", "additional_detail_texts": [],
            "caption_text": "Stop. The left sector is clear for 2.40 metres. Select left or right.",
        }}
        answer = questions.answer_text_from_release(release)
        self.assertNotIn("Stop.", answer)
        self.assertNotIn("Select left", answer)


class DeterministicAnswerTests(unittest.TestCase):
    """The fallback must answer the question that was asked, not the guidance question."""

    def test_bearing_fallback_describes_the_routed_sector(self):
        packet = make_packet(sectors=make_sectors("CONSTRAINED", "BLOCKED", "CLEAR"),
                             action_ids=["sector:centre"])
        answer = questions.deterministic_answer(
            "LEFT", packet, questions.route_requirements("LEFT", packet)
        )
        self.assertIn("left sector", answer)
        # The action binding is on the centre sector. Reporting it here would answer the guidance
        # question instead of the one the user asked, which is what the release builder's own
        # fallback does and why this renderer exists.
        self.assertNotIn("centre", answer)

    def test_object_is_named_with_its_measured_distance(self):
        packet = make_packet(objects=[make_object("object:9", "LEFT", label="chair")])
        answer = questions.deterministic_answer(
            "LEFT", packet, questions.route_requirements("LEFT", packet)
        )
        self.assertIn("chair", answer)
        self.assertIn("1.80 metres", answer)

    def test_moving_object_is_reported_as_moving(self):
        moving = make_object("object:9", "RIGHT", label="person")
        moving["motion_state"] = "MOVING"
        packet = make_packet(objects=[moving])
        answer = questions.deterministic_answer(
            "RIGHT", packet, questions.route_requirements("RIGHT", packet)
        )
        self.assertIn("is moving", answer)

    def test_unmeasured_sector_says_so_rather_than_inventing_a_state(self):
        packet = make_packet(sectors=make_sectors("UNKNOWN", "CLEAR", "CLEAR", valid=False))
        answer = questions.deterministic_answer(
            "LEFT", packet, questions.route_requirements("LEFT", packet)
        )
        self.assertIn("no reliable measurement", answer)

    def test_no_renderable_fact_returns_the_measurement_reply(self):
        packet = make_packet()
        self.assertEqual(questions.deterministic_answer("LEFT", packet, []),
                         "I do not have a reliable measurement of the left side right now. "
                         "Select Reassess for a fresh look.")


class TelemetryRecordTests(unittest.TestCase):
    def test_question_text_is_hashed_not_stored(self):
        record = questions.build_route_record("what is on my left", "LEFT", "TIER_0_KEYWORD", True)
        self.assertNotIn("left", json.dumps(record))
        self.assertTrue(record["question_text_sha256"].startswith("sha256:"))

    def test_resolved_by_is_recorded(self):
        record = questions.build_route_record("q", None, "MEASUREMENT_PRECHECK", False)
        self.assertEqual(record["resolved_by"], "MEASUREMENT_PRECHECK")
        self.assertFalse(record["reached_generation"])


class GrammarTests(unittest.TestCase):
    def test_grammar_lists_every_route_and_nothing_else(self):
        path = Path(__file__).resolve().parents[1] / "config" / "hdsg.question_route.v1.gbnf"
        text = path.read_text(encoding="utf-8")
        for route in questions.ROUTES:
            self.assertIn(f'"\\"{route}\\""', text)
        # The grammar is what bounds the classifier, so a route present in code but missing from
        # the grammar, or the reverse, is a drift the C-1 finding warns about.
        quoted = set(__import__("re").findall(r'"\\"([A-Z_]+)\\""', text))
        self.assertEqual(quoted, set(questions.ROUTES))


class QuestionNormalisationTests(unittest.TestCase):
    def test_whitespace_is_collapsed_and_length_capped(self):
        self.assertEqual(questions.normalise_question("  what   is\n left "), "what is left")
        self.assertEqual(len(questions.normalise_question("x" * 500)), questions.MAX_QUESTION_CHARS)


if __name__ == "__main__":
    unittest.main()
