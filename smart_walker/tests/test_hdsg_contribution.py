"""Tests for the deterministic-only ablation.

The condition answers what the generative layer adds over the deterministic layer alone, so the
tests concentrate on the comparison being exact: that the action text is excluded, that a candidate
reproducing the deterministic wording is reported as adding nothing, and that a clause the
deterministic layer cannot produce is counted apart from one that merely rewords a measured fact.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_contribution as contribution  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402


def make_sectors(left="CLEAR", centre="CONSTRAINED", right="BLOCKED"):
    values = {"CLEAR": 2.40, "CONSTRAINED": 1.40, "BLOCKED": 0.50, "UNKNOWN": None}
    return {
        name: {
            "fact_id": f"sector:{name}",
            "clearance_m": values[status],
            "valid": status != "UNKNOWN",
            "invalid_reason_codes": [] if status != "UNKNOWN" else ["DEPTH_SECTOR_INVALID"],
            "status": status,
        }
        for name, status in (("left", left), ("centre", centre), ("right", right))
    }


def make_event(intent="FORWARD", advisory="CAUTION", sectors=None, objects=(),
               response_mode="MORE_DETAIL"):
    """Builds a matched fact packet and prompt packet.

    `response_mode` matters to the comparison. The More detail profile requires a clause for every
    sector, including sectors the action-binding fallback does not render, so a candidate under it
    names facts the fallback omits. The automatic profile requires only the binding facts, which is
    the case where a difference in text can only be a difference in wording.
    """
    sectors = sectors if sectors is not None else make_sectors()
    request_id = "AUTO_GUIDANCE" if response_mode == "AUTOMATIC" else "MORE_DETAIL"
    authority = hdsg.determine_authority(
        intent, advisory, sectors, previous_selected_sector=None, sector_choice_tolerance_m=0.10
    )
    root = Path(__file__).resolve().parents[1]
    fact_packet = hdsg.build_fact_packet(
        run_id="run_t", event_id="evt_1", observation_id="obs_1", ticket_id=None,
        timestamp_ms=1.0, intent=intent, trigger_type="USER_REQUESTED",
        request_id=request_id, response_mode=response_mode, previous_signature=None,
        objects=list(objects), sectors=sectors, authority=authority, mirror_view=False,
        detector_model="yolov8n.pt", detector_confidence=0.35,
        pipeline_config_path=root / "config" / "pipeline.yaml",
        ontology_path=root / "config" / "ontology.yaml",
        clear_threshold_m=2.0, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.10,
        motion_tracker=hdsg.MotionTracker(),
    )
    prompt_packet = hdsg.build_prompt_packet(
        fact_packet, prompt_id="prompt_1", model_id="m", model_hash=hdsg.sha256_text("m"),
        quantisation=None, temperature=0.2, top_p=0.9, max_tokens=400,
        system_prompt="s", constraint_hash=hdsg.sha256_text("g"),
    )
    return fact_packet, prompt_packet


# The text the deterministic layer renders for the fixture scene, on both profiles. A caption
# reproducing it exactly is the case where the model added nothing at all.
FALLBACK_CAPTION = ("The centre sector has limited clearance at 1.40 metres. "
                    "The left sector is clear for 2.40 metres.")

# The same two facts in the model's own words. Under the retired templated design this could only
# be a synonym drawn from the runtime's own table; the model now writes it, and the measure has to
# report it as adding nothing either way.
REWORDED_CAPTION = ("The way ahead narrows to 1.40 metres, and there is 2.40 metres of space "
                    "to the left.")


def sector_assertion(name, value):
    return {"fact_id": f"sector:{name}",
            "measurement_id": f"m:sector:{name}:clearance",
            "stated_value": value}


BOTH_SECTORS = (sector_assertion("centre", 1.40), sector_assertion("left", 2.40))


def composed_release(fact_packet, prompt_packet, caption, assertions=BOTH_SECTORS,
                     visuals=(), release_id="release_1"):
    """Builds an accepted release from a composed caption, failing loudly if the gate refuses it.

    A test caption that silently failed the gate would fall back to the deterministic rendering and
    every assertion about contribution would then be made about text the model did not write.
    """
    candidate = {
        "schema_version": composed.CAPTION_SCHEMA,
        "caption": caption,
        "assertions": [dict(item) for item in assertions],
        "visual_observations": [dict(item) for item in visuals],
    }
    errors, scored = composed.validate_caption_candidate(
        candidate, prompt_packet, fact_packet, detector_classes=["chair", "person", "door"]
    )
    if errors:
        raise AssertionError(f"the test caption did not pass the gate: {errors}")
    return composed.build_composed_release(
        fact_packet, prompt_packet, release_id=release_id, candidate=candidate,
        scored_assertions=scored, failure_codes=[],
    )


class ReleaseBodyTests(unittest.TestCase):
    def test_the_action_sentence_is_excluded(self):
        # The action text is produced by the rule engine in both conditions, so including it would
        # report agreement the generative layer had no part in.
        release = {"content": {
            "action_text": "Stop.", "reason_text": "The centre sector is blocked at 0.50 metres.",
            "interaction_text": "Select left or right.", "additional_detail_texts": [],
            "caption_text": "Stop. The centre sector is blocked at 0.50 metres. Select left or right.",
        }}
        body = contribution.release_body(release)
        self.assertNotIn("Stop.", body)
        self.assertNotIn("Select left", body)
        self.assertIn("centre sector is blocked", body)

    def test_additional_details_are_included(self):
        release = {"content": {
            "action_text": "Stop.", "reason_text": "One.", "interaction_text": None,
            "additional_detail_texts": ["Two.", "Three."], "caption_text": "x",
        }}
        self.assertEqual(contribution.release_body(release), "One. Two. Three.")


class SplitClauseTests(unittest.TestCase):
    def test_a_decimal_measurement_is_not_split(self):
        clauses = contribution.split_clauses(
            "The left sector is clear for 2.40 metres. The centre sector is blocked at 0.50 metres."
        )
        self.assertEqual(len(clauses), 2)
        self.assertIn("2.40 metres", clauses[0])

    def test_empty_text_gives_no_clauses(self):
        self.assertEqual(contribution.split_clauses(""), [])


class ComparisonTests(unittest.TestCase):
    def test_a_reworded_caption_is_not_informative(self):
        """Free prose naming the same facts adds nothing, however different it reads.

        This caption shares not one sentence with the deterministic rendering and names exactly
        the facts that rendering already named. Counting it as contribution would measure the
        model's appetite for rephrasing rather than what the user learns, which is the
        measurement error this distinction exists to prevent.
        """
        fact_packet, prompt_packet = make_event(response_mode="AUTOMATIC")
        released = composed_release(fact_packet, prompt_packet, REWORDED_CAPTION)
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertEqual(result.gate_outcome, "ACCEPTED")
        self.assertFalse(result.identical)
        self.assertTrue(result.added_clauses)
        self.assertFalse(result.informative)
        self.assertTrue(result.reworded_only)
        self.assertEqual(result.added_facts, [])

    def test_a_caption_reproducing_the_deterministic_text_is_identical(self):
        # The boundary case of the measure. A model that happens to write the sentences the
        # fallback renderer writes contributed nothing, and the comparison must say so rather
        # than crediting it for arriving at the same place independently.
        fact_packet, prompt_packet = make_event(response_mode="AUTOMATIC")
        released = composed_release(fact_packet, prompt_packet, FALLBACK_CAPTION)
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertEqual(result.gate_outcome, "ACCEPTED")
        self.assertTrue(result.identical)
        self.assertFalse(result.informative)

    def test_naming_a_fact_the_fallback_omitted_is_informative(self):
        fact_packet, prompt_packet = make_event()
        released = composed_release(
            fact_packet, prompt_packet,
            "The way ahead narrows to 1.40 metres, with 2.40 metres of space to the left "
            "and only 0.50 metres to the right.",
            assertions=BOTH_SECTORS + (sector_assertion("right", 0.50),),
        )
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        # The More detail profile requires a clause for the right sector, which the action-binding
        # fallback does not render.
        self.assertTrue(result.informative)
        self.assertIn("m:sector:right:clearance", result.added_facts)
        self.assertFalse(result.reworded_only)

    def test_facts_named_is_read_from_the_evidence_not_the_prose(self):
        fact_packet, prompt_packet = make_event()
        baseline = contribution.deterministic_release(fact_packet, prompt_packet)
        named = contribution.facts_named(baseline)
        self.assertIn("m:sector:centre:clearance", named)
        self.assertIn("m:sector:left:clearance", named)

    def test_a_visual_observation_is_counted_as_a_visual_clause(self):
        fact_packet, prompt_packet = make_event()
        released = composed_release(
            fact_packet, prompt_packet, REWORDED_CAPTION,
            visuals=[{"candidate_observation_id": "visual:1",
                      "proposed_label": "doorway", "bearing": "LEFT"}],
        )
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertEqual(result.gate_outcome, "ACCEPTED")
        self.assertEqual(result.added_visual_clauses, 1)
        self.assertFalse(result.identical)

    def test_a_rejected_candidate_is_identical_to_the_deterministic_condition(self):
        # A rejected candidate already falls back to the deterministic rendering, so the ablation
        # must report no contribution for it. Anything else would credit the model for text the
        # deterministic layer produced.
        fact_packet, prompt_packet = make_event()
        released = hdsg.build_release(
            fact_packet, prompt_packet, release_id="release_1",
            failure_codes=["RG_CONSTRAINT_FAILURE"],
        )
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertTrue(result.identical)
        self.assertEqual(result.added_clauses, [])

    def test_the_deterministic_condition_needs_no_model(self):
        fact_packet, prompt_packet = make_event()
        baseline = contribution.deterministic_release(fact_packet, prompt_packet)
        self.assertEqual(baseline["verification"]["candidate_status"], "UNAVAILABLE")
        self.assertTrue(contribution.release_body(baseline))


class BeyondProfileTests(unittest.TestCase):
    """The stricter measure: what the model added over the same requirement set, without a model."""

    def test_the_measure_reads_declared_identifiers_not_opening_words(self):
        """A caption sharing no wording with the profile baseline still reaches beyond nothing.

        The measure previously keyed on a clause's first three words, which grouped the runtime's
        own synonyms together and nothing else. Free prose defeats that: "The way ahead narrows to
        1.40 metres" and "The centre sector has limited clearance at 1.40 metres" state one fact
        and share no opening, so every composed caption would have been reported as reaching beyond
        its profile. The declared identifiers give the attribution exactly.
        """
        fact_packet, prompt_packet = make_event(response_mode="AUTOMATIC")
        released = composed_release(fact_packet, prompt_packet, REWORDED_CAPTION)
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertEqual(result.gate_outcome, "ACCEPTED")
        self.assertTrue(result.reworded_only)
        self.assertFalse(result.beyond_profile)
        self.assertEqual(result.beyond_profile_facts, [])

    def test_a_fact_the_profile_selected_is_not_beyond_it(self):
        """Informative and beyond-profile are different questions and separate here.

        The More detail profile selects the right sector, which the action-binding fallback does not
        render. A caption naming it tells the user something the deterministic layer withheld, and
        it still adds nothing over the requirement set, because a renderer given the same set
        produces it with no model at all.
        """
        fact_packet, prompt_packet = make_event(response_mode="MORE_DETAIL")
        released = composed_release(
            fact_packet, prompt_packet,
            "The way ahead narrows to 1.40 metres, with 2.40 metres of space to the left "
            "and only 0.50 metres to the right.",
            assertions=BOTH_SECTORS + (sector_assertion("right", 0.50),),
        )
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertTrue(result.informative)
        self.assertFalse(result.beyond_profile)

    def test_a_visual_observation_is_the_one_thing_that_does(self):
        fact_packet, prompt_packet = make_event()
        released = composed_release(
            fact_packet, prompt_packet, REWORDED_CAPTION,
            visuals=[{"candidate_observation_id": "visual:1",
                      "proposed_label": "doorway", "bearing": "LEFT"}],
        )
        result = contribution.compare_event(fact_packet, prompt_packet, released)
        self.assertTrue(result.beyond_profile)
        self.assertEqual(result.beyond_profile_facts, ["visual:1"])

    def test_the_profile_baseline_renders_every_required_fact(self):
        fact_packet, prompt_packet = make_event(response_mode="MORE_DETAIL")
        text = contribution.profile_text(fact_packet, prompt_packet)
        for name in ("centre", "left", "right"):
            self.assertIn(f"{name} sector", text)

    def test_the_profile_baseline_can_exceed_the_fallback(self):
        # The two baselines answer different questions, and on the More detail profile they differ.
        fact_packet, prompt_packet = make_event(response_mode="MORE_DETAIL")
        fallback = contribution.release_body(
            contribution.deterministic_release(fact_packet, prompt_packet))
        self.assertNotIn("right sector", fallback)
        self.assertIn("right sector", contribution.profile_text(fact_packet, prompt_packet))


class VisualClausePatternTests(unittest.TestCase):
    def test_the_pattern_matches_the_rendered_form(self):
        self.assertTrue(contribution.VISUAL_CLAUSE_RE.match(
            "Possible doorway is visible in the left."))
        self.assertTrue(contribution.VISUAL_CLAUSE_RE.match(
            "Possible steps are visible in the right."))

    def test_the_pattern_does_not_match_a_fact_clause(self):
        self.assertIsNone(contribution.VISUAL_CLAUSE_RE.match(
            "The left sector is clear for 2.40 metres."))


class SummaryTests(unittest.TestCase):
    def make(self, gate, added=(), facts=(), visual=0):
        return contribution.EventContribution(
            event_id="evt", gate_outcome=gate, released_text="r", deterministic_text="d",
            added_clauses=list(added), added_facts=list(facts), added_visual_clauses=visual,
        )

    def test_the_headline_rate_is_over_accepted_events_only(self):
        # A rejected candidate is identical by construction. Including rejections in the
        # denominator would make the architecture look more generative the more often it rejected.
        items = [
            self.make("ACCEPTED"),
            self.make("ACCEPTED", added=["Possible doorway is visible in the left."], visual=1),
            self.make("REJECTED_FALLBACK"),
            self.make("REJECTED_FALLBACK"),
        ]
        summary = contribution.summarise(items)
        self.assertEqual(summary["accepted_events"], 2)
        self.assertEqual(summary["accepted_informative"], 1)
        self.assertEqual(summary["accepted_informative_rate"], 0.5)

    def test_a_reworded_event_is_not_counted_as_informative(self):
        items = [self.make("ACCEPTED", added=["The left sector remains clear for 2.40 metres."])]
        summary = contribution.summarise(items)
        self.assertEqual(summary["accepted_informative"], 0)
        self.assertEqual(summary["accepted_reworded_only"], 1)
        self.assertEqual(summary["accepted_identical_to_deterministic"], 0)

    def test_the_three_accepted_categories_are_exhaustive(self):
        items = [
            self.make("ACCEPTED"),
            self.make("ACCEPTED", added=["reworded."]),
            self.make("ACCEPTED", added=["new."], facts=["m:sector:right:clearance"]),
        ]
        summary = contribution.summarise(items)
        self.assertEqual(
            summary["accepted_identical_to_deterministic"]
            + summary["accepted_reworded_only"]
            + summary["accepted_informative"],
            summary["accepted_events"],
        )

    def test_visual_and_fact_additions_are_counted_apart(self):
        items = [self.make("ACCEPTED",
                           added=["Possible doorway is visible in the left.",
                                  "The right sector is blocked at 0.50 metres."],
                           facts=["m:sector:right:clearance"], visual=1)]
        summary = contribution.summarise(items)
        self.assertEqual(summary["added_visual_clauses"], 1)
        self.assertEqual(summary["added_facts_total"], 1)

    def test_an_empty_population_reports_no_rate(self):
        summary = contribution.summarise([])
        self.assertEqual(summary["events"], 0)
        self.assertIsNone(summary["accepted_informative_rate"])


class ArchiveTests(unittest.TestCase):
    def write_archive(self, path, records):
        with Path(path).open("w", encoding="utf-8") as stream:
            for kind, record in records:
                stream.write(json.dumps({"record_type": kind, "record": record}) + "\n")

    def test_a_run_is_compared_end_to_end(self):
        fact_packet, prompt_packet = make_event()
        released = composed_release(fact_packet, prompt_packet, REWORDED_CAPTION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_archive(path, [
                ("full_fact_packet", fact_packet),
                ("restricted_prompt_packet", prompt_packet),
                ("authoritative_release", released),
            ])
            results = contribution.compare_run(path)
        self.assertEqual(results["compared"], 1)
        self.assertEqual(results["unpaired"], 0)
        self.assertEqual(results["summary"]["accepted_events"], 1)

    def test_a_pending_release_is_skipped(self):
        # An interim release carries the placeholder rather than candidate text. Comparing it would
        # report an absence the generative layer is not responsible for.
        fact_packet, prompt_packet = make_event()
        pending = hdsg.build_release(
            fact_packet, prompt_packet, release_id="release_0",
            failure_codes=[], pending=True,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_archive(path, [
                ("full_fact_packet", fact_packet),
                ("restricted_prompt_packet", prompt_packet),
                ("authoritative_release", pending),
            ])
            results = contribution.compare_run(path)
        self.assertEqual(results["released_events"], 0)
        self.assertEqual(results["compared"], 0)

    def test_a_release_without_its_evidence_is_counted_not_compared(self):
        fact_packet, prompt_packet = make_event()
        released = hdsg.build_release(
            fact_packet, prompt_packet, release_id="release_1",
            failure_codes=["RG_CONSTRAINT_FAILURE"],
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_archive(path, [("authoritative_release", released)])
            results = contribution.compare_run(path)
        self.assertEqual(results["unpaired"], 1)
        self.assertEqual(results["compared"], 0)

    def test_results_are_written_with_a_final_summary(self):
        fact_packet, prompt_packet = make_event()
        released = composed_release(fact_packet, prompt_packet, REWORDED_CAPTION)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.jsonl"
            self.write_archive(path, [
                ("full_fact_packet", fact_packet),
                ("restricted_prompt_packet", prompt_packet),
                ("authoritative_release", released),
            ])
            out = Path(directory) / "out.jsonl"
            contribution.write_results(contribution.compare_run(path), out)
            lines = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(lines[-1]["record_type"], "contribution_summary")
        self.assertEqual(lines[0]["record_type"], "contribution_event")


if __name__ == "__main__":
    unittest.main()
