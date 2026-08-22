"""The detector's labels against the safety buckets.

The ontology decides two things. It decides which detections become obstacles at all, and it decides
which of them stop the walker at 2.00 m rather than 0.70 m. Nothing else in the pipeline reads a
detector label, so an error here is invisible everywhere else and shows up only as a walker that
stops for a shirt or fails to stop for a staircase.

There were no tests over this file before 22 August 2026. On that date the approximate matching was
removed, which exposed that four of COCO's eighty class names reached a bucket and none reached
`hazard`, and the detector was changed to Open Images V7. Both changes are asserted here against the
real weights' class list rather than against a copy of it, so a future change of weights fails these
tests instead of silently emptying the buckets again.
"""

from __future__ import annotations

import unittest

from support import CONFIG  # noqa: F401
from scripts.realsense_shared_control import OntologyMapper

DETECTOR_WEIGHTS = "yolov8n-oiv7.pt"


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
        return list(YOLO(DETECTOR_WEIGHTS).names.values())
    except Exception as error:  # pragma: no cover, depends on the environment
        raise unittest.SkipTest(f"{DETECTOR_WEIGHTS} unavailable: {error}")


class MapperTests(unittest.TestCase):
    """Label to bucket, without reference to any particular detector."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_a_canonical_name_reaches_its_bucket(self):
        mapped = self.mapper.map_label("chair")
        self.assertEqual(("chair", "furniture"),
                         (mapped.canonical_class, mapped.ontology_class))

    def test_matching_ignores_case(self):
        """Open Images capitalises its class names and COCO does not."""
        self.assertEqual("furniture", self.mapper.map_label("Chair").ontology_class)

    def test_a_synonym_is_folded_onto_its_canonical_name(self):
        """The canonical name becomes the word the caption may use, so the detector's own phrasing
        must not reach the user verbatim."""
        mapped = self.mapper.map_label("Kitchen & dining room table")
        self.assertEqual(("table", "furniture"),
                         (mapped.canonical_class, mapped.ontology_class))

    def test_an_unlisted_label_is_an_unknown_obstacle_and_not_a_guess(self):
        """Approximate matching mapped `stop sign` to `stairs_up`, so the walker stopped for a road
        sign and recorded a staircase. An unrecognised label is now what it is."""
        mapped = self.mapper.map_label("Croissant")
        self.assertEqual((None, "unknown_obstacle"),
                         (mapped.canonical_class, mapped.ontology_class))

    def test_an_empty_label_is_an_unknown_obstacle(self):
        self.assertEqual("unknown_obstacle", self.mapper.map_label("").ontology_class)

    def test_a_prompt_word_yields_a_bucket_without_renaming_the_object(self):
        """A word in the open-vocabulary list is not a canonical name. Returning the bucket's first
        canonical instead turned every bench and desk into a chair."""
        mapped = self.mapper.map_label("mobility scooter")
        self.assertEqual((None, "mobility_aid"),
                         (mapped.canonical_class, mapped.ontology_class))


class HomeEnvironmentTests(unittest.TestCase):
    """The rewrite of 22 August 2026, from a hospital vocabulary to a domestic one."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_a_pet_is_named_rather_than_called_an_obstacle(self):
        """A pet is the thing most likely to be underfoot in a house. Under the hospital ontology a
        dog reached no group, so the walker stopped for it but could only call it an obstacle, and
        "a dog is ahead" tells somebody something that "an obstacle is ahead" does not."""
        for name, expected in (("Dog", "dog"), ("Cat", "cat"), ("Rabbit", "rabbit")):
            with self.subTest(name):
                mapped = self.mapper.map_label(name)
                self.assertEqual((expected, "animal"),
                                 (mapped.canonical_class, mapped.ontology_class))

    def test_a_bed_does_not_roll_in_a_house(self):
        """The hospital ontology filed a bed with the things that roll, which is true of a bed on
        castors being pushed down a ward and false of a bed in a bedroom."""
        self.assertEqual("furniture", self.mapper.map_label("Bed").ontology_class)

    def test_a_trolley_still_rolls(self):
        self.assertEqual(("trolley", "wheeled_object"),
                         tuple(self.mapper.map_label("Cart").__dict__.values()))

    def test_kitchen_machines_are_separated_from_furniture(self):
        """Naming the fridge places somebody in the kitchen in a way that "furniture" does not."""
        self.assertEqual(("fridge", "appliance"),
                         tuple(self.mapper.map_label("Refrigerator").__dict__.values()))

    def test_a_bag_left_out_is_not_filed_as_furniture(self):
        """Furniture is where it was yesterday and a bag in a hallway is not, so they are separated.
        The group is also the one a person can do something about."""
        self.assertEqual(("bag", "floor_object"),
                         tuple(self.mapper.map_label("Backpack").__dict__.values()))


class HazardBucketTests(unittest.TestCase):
    """The one bucket that changes what the walker does."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_stairs_reach_the_hazard_bucket(self):
        """`hazard` is the only bucket that stops the walker at 2.00 m instead of 0.70 m. Under the
        COCO vocabulary no label reached it, so the longer stopping distance had never fired."""
        self.assertEqual("hazard", self.mapper.map_label("Stairs").ontology_class)

    def test_a_ladder_is_not_a_hazard(self):
        """The medium Open Images model labels the recorded staircase `Ladder` at 0.30, which is an
        argument for putting it here and not a reason. Mapping a class to `hazard` because one model
        confused it once is the error that put a stop sign there."""
        self.assertEqual("furniture", self.mapper.map_label("Ladder").ontology_class)

    def test_a_stop_sign_is_not_a_hazard(self):
        """The specific failure that approximate matching produced."""
        self.assertNotEqual("hazard", self.mapper.map_label("Stop sign").ontology_class)


class NotObstacleTests(unittest.TestCase):
    """Labels dropped before they become facts."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")

    def test_a_window_is_not_an_obstacle(self):
        """Every Open Images model tested reports `Window` in all three recorded stair frames, more
        confidently than it reports the staircase, because the stairwell is lit from behind by one.
        A window is on a wall and cannot be walked into on the floor plane."""
        self.assertTrue(self.mapper.is_not_obstacle("Window"))

    def test_a_part_of_a_person_is_not_a_separate_obstacle(self):
        """Open Images labels body parts separately from `Person`, so one pedestrian yields several
        detections at the same distance and fills the caption with one person described five times.
        """
        self.assertTrue(self.mapper.is_not_obstacle("Human face"))

    def test_a_chair_is_an_obstacle(self):
        self.assertFalse(self.mapper.is_not_obstacle("Chair"))

    def test_stairs_are_an_obstacle(self):
        """The list is a way of dropping noise and must never drop the hazard class."""
        self.assertFalse(self.mapper.is_not_obstacle("Stairs"))

    def test_a_swimming_pool_is_a_hazard_and_is_not_dropped(self):
        """The edge of a pool is an unguarded fall of over a metre, which is what this bucket is
        for, and it is the only outdoor drop-off the detector can name. The first draft of the
        ontology dropped it, having grouped it with building fabric."""
        self.assertFalse(self.mapper.is_not_obstacle("Swimming pool"))
        self.assertEqual("hazard", self.mapper.map_label("Swimming pool").ontology_class)

    def test_things_standing_on_the_floor_are_not_dropped(self):
        """The list is for what a wheel cannot reach. A standard lamp stands on the floor, a curtain
        hangs to it, a full length mirror leans against a wall, and a signpost or a hydrant is
        planted in the pavement at the height that catches a walker frame. All twelve were in the
        list on the first draft, grouped as building fabric, which they are not."""
        for name in ("Lamp", "Mirror", "Curtain", "Fountain", "Billboard",
                     "Stop sign", "Traffic sign", "Traffic light", "Street light",
                     "Parking meter", "Fire hydrant", "Swimming pool"):
            with self.subTest(name):
                self.assertFalse(self.mapper.is_not_obstacle(name))

    def test_an_empty_label_is_not_dropped_here(self):
        """An empty label is handled by `map_label`, which returns `unknown_obstacle`. Dropping it
        here instead would discard a detection that has a distance and a bearing."""
        self.assertFalse(self.mapper.is_not_obstacle(""))


class DetectorVocabularyTests(unittest.TestCase):
    """The ontology against the class list the shipped weights carry."""

    def setUp(self):
        self.mapper = OntologyMapper(CONFIG / "ontology.yaml")
        self.names = detector_class_names()

    def test_the_hazard_bucket_is_reachable(self):
        """The measurement that caused the change of weights. Under COCO this returned nothing, so
        no detection could ever receive the 2.00 m hazard stopping distance and the bucket was dead
        code that the safety argument nonetheless relied on."""
        reaching = [name for name in self.names
                    if self.mapper.map_label(name).ontology_class == "hazard"]
        self.assertTrue(reaching, "no detector class reaches the hazard bucket")

    def test_the_buckets_are_not_nearly_empty(self):
        """Four of COCO's eighty names reached a bucket. The threshold is deliberately low: it is
        set to catch a vocabulary mismatch of that scale, not to fix a target."""
        mapped = [name for name in self.names
                  if self.mapper.map_label(name).ontology_class != "unknown_obstacle"]
        self.assertGreater(len(mapped), 20, f"only {len(mapped)} of {len(self.names)} names mapped")

    def test_a_person_is_an_agent(self):
        self.assertEqual("person", self.mapper.map_label("Person").ontology_class)

    def test_every_not_obstacle_entry_names_a_real_class(self):
        """An entry that matches nothing the detector emits is dead configuration, and the file
        gives no sign of it. Comparison is lowercased because the entries are written that way."""
        emitted = {name.strip().lower() for name in self.names}
        unmatched = sorted(self.mapper.not_obstacles - emitted)
        self.assertEqual([], unmatched)

    def test_every_synonym_names_a_real_class_or_a_canonical_name(self):
        """A synonym key that the detector never emits, and that is not itself a canonical name, is
        a mapping that can never fire."""
        emitted = {name.strip().lower() for name in self.names}
        known = emitted | set(self.mapper.ontology_buckets) | set(self.mapper.prompt_buckets)
        unmatched = sorted(key for key in self.mapper.synonyms if key not in known)
        self.assertEqual([], unmatched)

    def test_every_synonym_resolves_to_a_canonical_name(self):
        """A synonym pointing at a word no bucket lists sends the label to `unknown_obstacle` by a
        longer route than leaving it alone, which is worse because the file reads as if it works."""
        unresolved = sorted(value for value in self.mapper.synonyms.values()
                            if value not in self.mapper.ontology_buckets)
        self.assertEqual([], unresolved)

    def test_no_class_is_both_dropped_and_bucketed(self):
        """`is_not_obstacle` is consulted first, so a class in both places is silently dropped and
        its bucket entry never fires."""
        conflicting = sorted(name for name in self.names
                             if self.mapper.is_not_obstacle(name)
                             and self.mapper.map_label(name).ontology_class != "unknown_obstacle")
        self.assertEqual([], conflicting)


if __name__ == "__main__":
    unittest.main()
