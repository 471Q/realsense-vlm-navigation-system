"""The detector's labels against the safety buckets.

The ontology decides two things. It decides which detections become obstacles at all, and it decides
which of them stop the walker at 2.00 m rather than 0.70 m. Nothing else in the pipeline reads a
detector label, so an error here is invisible everywhere else and shows up only as a walker that
stops for a shirt or fails to stop for a staircase.

Rewritten 23 August 2026, when the file stopped being maintained by hand and became the output of
`scripts/generate_ontology.py` reading `model.names`. The tests before that date asserted the
contents of a hand-written mapping: that a chair was in `furniture`, that `Stairs` reached `hazard`,
that a `Cart` rolled. None of those propositions survives, because there are no longer any groups
beyond `hazard` and no entries for words the loaded weight cannot emit.

What replaces them is narrower and stronger. A hand-written mapping needs tests that each entry is
correct. A generated one needs tests that it agrees with the weight, which is a property of the
whole file and cannot be satisfied by an entry that happens to be right.
"""

from __future__ import annotations

import io
import unittest
from contextlib import redirect_stdout

from support import CONFIG  # noqa: F401
from scripts.realsense_shared_control import OntologyMapper
from scripts import generate_ontology

DETECTOR_WEIGHTS = "yolov8n.pt"


def detector_class_names():
    """The class list the shipped weights actually carry.

    Read from the weights rather than transcribed, because a transcription cannot go out of date
    visibly. Skipped where ultralytics or the weights file is unavailable, so the suite still runs
    on a machine with no model cache.
    """
    try:
        from ultralytics import YOLO
    except Exception as error:  # pragma: no cover, depends on the environment
        raise unittest.SkipTest(f"ultralytics unavailable: {error}")
    try:
        names = YOLO(DETECTOR_WEIGHTS).names
    except Exception as error:  # pragma: no cover, depends on the environment
        raise unittest.SkipTest(f"{DETECTOR_WEIGHTS} unavailable: {error}")
    return [str(names[index]) for index in sorted(names)]


class GeneratedFileTests(unittest.TestCase):
    """The file on disk against the weight it claims to describe."""

    def test_the_file_on_disk_is_what_the_generator_writes(self):
        """The whole point of generating the file. An edit made by hand, or a weight changed without
        regenerating, fails here rather than at the next capture session.

        This is the test that could not exist while the file was maintained by hand, and its absence
        is how three hundred Open Images entries survived the revert to COCO.
        """
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            status = generate_ontology.main(
                ["--model", DETECTOR_WEIGHTS, "--out", str(CONFIG / "ontology.yaml"), "--check"])
        self.assertEqual(0, status,
                         "config/ontology.yaml is out of step with the detector weight. "
                         "Run: python scripts/generate_ontology.py")

    def test_every_class_the_detector_can_emit_is_mapped(self):
        """Four of COCO's eighty names reached a bucket under the hand-written file. Generation makes
        full coverage structural rather than something to be checked against a threshold."""
        mapper = OntologyMapper(CONFIG / "ontology.yaml")
        unmapped = sorted(name for name in detector_class_names()
                          if not mapper.is_not_obstacle(name)
                          and mapper.map_label(name).ontology_class == "unknown_obstacle")
        self.assertEqual([], unmapped)

    def test_the_file_carries_nothing_the_detector_cannot_emit(self):
        """The stale-entry rule, applied to the whole file rather than to `not_obstacles` alone.

        Every bucket name, every canonical name and every synonym key must trace back to a word the
        loaded weight can produce, allowing for the four renames. An entry that cannot fire is
        indistinguishable in the file from one that can, which is what made the Open Images residue
        invisible for a day.
        """
        emitted = {name.strip().lower() for name in detector_class_names()}
        renamed = set(generate_ontology.RENAMES.values())
        allowed = emitted | renamed | {"hazard"}
        stray = sorted(set(mapper_names(CONFIG)) - allowed)
        self.assertEqual([], stray)


def mapper_names(config_dir):
    """Every canonical and bucket name the ontology declares."""
    mapper = OntologyMapper(config_dir / "ontology.yaml")
    return set(mapper.ontology_buckets) | set(mapper.ontology_buckets.values())


class MapperTests(unittest.TestCase):
    """Label to bucket."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_a_canonical_name_reaches_its_own_bucket(self):
        """Under generation every class is its own group, so `ontology_class` states the class
        rather than a judgement about it."""
        mapped = self.mapper.map_label("chair")
        self.assertEqual(("chair", "chair"),
                         (mapped.canonical_class, mapped.ontology_class))

    def test_matching_ignores_case(self):
        self.assertEqual("chair", self.mapper.map_label("Chair").ontology_class)

    def test_matching_ignores_surrounding_space(self):
        self.assertEqual("chair", self.mapper.map_label("  chair  ").ontology_class)

    def test_the_mapper_reads_every_key_the_generator_writes_and_no_others(self):
        """A reader for a key the generator does not emit is a branch that cannot run.

        The mapper read a `prompts` list per bucket until 24 August 2026, giving a word its bucket
        without a canonical name. No generated bucket has ever carried one and none can, the
        generator emitting `canonical` alone and `--check` failing on a hand edit, so the branch was
        unreachable and the loader read a key no file holds.

        Asserted over the file rather than over the mapper, because the direction that matters is
        the generator adding a key the mapper then ignores in silence.
        """
        import yaml

        ontology = yaml.safe_load((CONFIG / "ontology.yaml").read_text(encoding="utf-8"))
        keys = {key for bucket in ontology["ontology"] for key in bucket}
        self.assertEqual({"name", "canonical"}, keys)
        self.assertEqual({"ontology", "synonyms_to_canonical", "not_obstacles"}, set(ontology))

    def test_an_unlisted_label_is_an_unknown_obstacle_and_not_a_guess(self):
        """Approximate matching mapped `stop sign` to `stairs_up`, so the walker stopped for a road
        sign and recorded a staircase. An unrecognised label is now what it is."""
        mapped = self.mapper.map_label("Croissant")
        self.assertEqual((None, "unknown_obstacle"),
                         (mapped.canonical_class, mapped.ontology_class))

    def test_an_empty_label_is_an_unknown_obstacle(self):
        self.assertEqual("unknown_obstacle", self.mapper.map_label("").ontology_class)

    def test_the_renames_that_earn_their_place_survive(self):
        """Four COCO words read better renamed, and only those four are renamed. The rest of COCO's
        vocabulary is ordinary English, and the renames written for Open Images made it vaguer:
        `vase` was presented as "plant", `laptop` and `keyboard` both as "computer"."""
        for name, canonical in (("dining table", "table"), ("tv", "television"),
                                ("refrigerator", "fridge"), ("cell phone", "telephone")):
            with self.subTest(name):
                mapped = self.mapper.map_label(name)
                self.assertEqual(canonical, mapped.canonical_class)
                self.assertEqual(canonical, mapped.ontology_class)

    def test_a_plain_detector_word_is_not_made_vaguer(self):
        """Everything outside those four keeps the detector's own word."""
        for name in ("vase", "laptop", "keyboard", "stop sign", "horse", "teddy bear", "dog"):
            with self.subTest(name):
                self.assertEqual(name, self.mapper.map_label(name).canonical_class)

    def test_every_synonym_resolves_to_a_canonical_name(self):
        """A synonym pointing at a word no bucket lists sends the label to `unknown_obstacle` by a
        longer route than leaving it alone, which is worse because the file reads as if it works."""
        unresolved = sorted(value for value in self.mapper.synonyms.values()
                            if value not in self.mapper.ontology_buckets)
        self.assertEqual([], unresolved)


class HazardBucketTests(unittest.TestCase):
    """The one bucket that changes what the walker does."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_the_hazard_bucket_is_empty_and_unreachable(self):
        """A known and accepted limitation, asserted so that it cannot change unnoticed.

        COCO has no word for a staircase, a ramp or a drop, so no detection ever receives the 2.00 m
        hazard stopping distance and every obstacle is treated alike at 0.70 m. The bucket is emitted
        empty rather than filled with words from a vocabulary that is not loaded, which is what the
        hand-written file did.

        This records a state, not an aspiration. How a fall is detected is item 5 of the deferred
        register and is settled from the laboratory captures.
        """
        reaching = [name for name in detector_class_names()
                    if self.mapper.map_label(name).ontology_class == "hazard"]
        self.assertEqual([], reaching)

    def test_a_stop_sign_is_not_a_hazard(self):
        """The specific failure that approximate matching produced."""
        self.assertNotEqual("hazard", self.mapper.map_label("stop sign").ontology_class)


class NotObstacleTests(unittest.TestCase):
    """Labels dropped before they become facts. The only way for the walker to ignore something the
    camera saw, so this list is held to a stricter standard than anything else in the file."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_a_part_of_a_person_is_not_a_separate_obstacle(self):
        """COCO reports `person` and `tie` as two detections of the same body at the same distance,
        so without this the description names one visitor twice. It is the only entry."""
        self.assertTrue(self.mapper.is_not_obstacle("tie"))

    def test_the_list_holds_only_what_the_running_detector_can_say(self):
        """The rule that would have caught the stale list.

        84 entries were added on 22 August 2026 for Open Images V7, the detector was reverted to
        COCO the same day, and 82 survived as rules against words the loaded detector cannot say.
        Generation now filters the list against `model.names`, so this asserts the filter rather
        than the file.
        """
        vocabulary = {name.strip().lower() for name in detector_class_names()}
        listed = {str(name).lower() for name in self.mapper.not_obstacles}
        self.assertEqual(set(), listed - vocabulary,
                         "every dropped label must be one the loaded detector can emit")

    def test_no_class_is_both_dropped_and_bucketed(self):
        """`is_not_obstacle` is consulted first, so a class in both places is silently dropped and
        its bucket entry never fires. The first generated file did exactly this to `tie`."""
        conflicting = sorted(name for name in detector_class_names()
                             if self.mapper.is_not_obstacle(name)
                             and self.mapper.map_label(name).ontology_class != "unknown_obstacle")
        self.assertEqual([], conflicting)

    def test_a_window_would_be_an_obstacle(self):
        """Recorded by decision, 23 August 2026, against a vocabulary that has no word for it.

        A window was dropped while Open Images was loaded, because every model tested reported
        `Window` in all three recorded stair frames more confidently than the staircase. But a window
        sits in a wall, there is no `wall` class for the wall to be reported as, and discarding the
        window reports nothing for a surface the walker can hit. COCO cannot emit the word at all, so
        this asserts only that the reversed decision was not carried into the generator.
        """
        self.assertNotIn("window", [w.lower() for w in generate_ontology.NOT_OBSTACLES])

    def test_a_chair_is_an_obstacle(self):
        self.assertFalse(self.mapper.is_not_obstacle("Chair"))

    def test_an_empty_label_is_not_dropped_here(self):
        """An empty label is handled by `map_label`, which returns `unknown_obstacle`. Dropping it
        here instead would discard a detection that has a distance and a bearing."""
        self.assertFalse(self.mapper.is_not_obstacle(""))


class GeneratorTests(unittest.TestCase):
    """The generator itself, against a class list it is handed rather than a weight."""

    def build(self, names):
        return generate_ontology.build_document("fake.pt", names, "2.00", "0.70")

    def test_a_rename_for_an_absent_class_is_dropped(self):
        """The self-pruning that stops the drift. `cell phone` is a COCO word; a weight without it
        must not carry a rename for it, because a rule that cannot fire is indistinguishable in the
        file from one that can."""
        document = self.build(["chair", "door"])
        self.assertNotIn("cell phone", document)

    def test_a_rename_for_a_present_class_is_written(self):
        document = self.build(["chair", "cell phone"])
        self.assertIn("cell phone: telephone", document)
        self.assertIn("name: telephone", document)

    def test_a_dropped_class_gets_no_bucket(self):
        document = self.build(["chair", "tie"])
        self.assertIn("- tie", document)
        self.assertNotIn("name: tie", document)

    def test_the_weight_and_the_count_are_recorded_in_the_file(self):
        """A generated file must say what it was generated from, or the check above is the only way
        to find out and it needs the weight to hand."""
        document = self.build(["chair", "door", "person"])
        self.assertIn("fake.pt", document)
        self.assertIn("Classes: 3", document)

    def test_the_hazard_bucket_is_always_present(self):
        """`OntologyMapper` builds `hazard_set` by looking for a bucket of that name. Emitting the
        bucket unconditionally keeps the name defined whatever vocabulary is loaded."""
        self.assertIn("name: hazard", self.build(["chair"]))


if __name__ == "__main__":
    unittest.main()
