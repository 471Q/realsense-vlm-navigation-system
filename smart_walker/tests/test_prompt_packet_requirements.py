"""What the packet demands of the caption, checked against what the builder can produce.

A requirement names the facts a clause must rest on, and the set of them is what the person is
promised the answer will cover. The frozen schema capped the set at three until 23 August 2026,
which is one fewer than the builder can produce: a More detail packet naming an action binding, a
scene binding and two further details is a shape the runtime reaches from an ordinary scene, and the
schema refused it. Nothing failed at the time because the runtime does not validate a prompt packet
as it writes one, so the refusal would only have appeared later, against the archive.

The sweep below is the check that matters. Rather than asserting the cap is four, it builds packets
across the scene shapes the runtime can be in and validates every one, so a change to the builder
that lifts the count is caught by the same test that fixed the cap.
"""

from __future__ import annotations

import ast
import copy
import json
import unittest

from support import SCHEMAS, detected_object, event, sectors  # noqa: F401


def image_transform():
    """`_image_transform` from the interactive script, lifted out of its module.

    Imported by parsing rather than by `import`, because the module it lives in opens a camera and
    pulls in the detector at import time. The function has no dependencies of its own, so compiling
    the one definition is enough and keeps the test running on a machine with no camera attached.
    """
    source = (SCHEMAS.parent / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(
        encoding="utf-8")
    definition = next(node for node in ast.parse(source).body
                      if isinstance(node, ast.FunctionDef) and node.name == "_image_transform")
    namespace: dict = {}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), "<lifted>", "exec"), namespace)
    return namespace["_image_transform"]

# Three sector layouts, chosen so that the centre is clear in one, blocked in another, and the
# choice between the two sides is contested in the third. Which of these holds decides whether a
# scene binding is required, and the scene binding is one of the two requirements the builder adds
# without checking the remaining clause budget.
LANES = {
    "centre clear": sectors(),
    "centre blocked, right open": sectors(0.4, 0.5, 3.2, ("CONSTRAINED", "CONSTRAINED", "CLEAR")),
    "centre blocked, left open": sectors(3.2, 0.5, 0.4, ("CLEAR", "CONSTRAINED", "CONSTRAINED")),
}

SCENES = {
    "nothing detected": (),
    "one object close": detected_object(distance_m=0.9),
    "two objects": (detected_object(distance_m=0.9)
                    + detected_object(track_id=7, label="person", bearing="RIGHT",
                                      distance_m=2.0)),
}


class RequirementSweepTests(unittest.TestCase):

    def setUp(self):
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")
        self.validator = jsonschema.Draft7Validator(json.loads(
            (SCHEMAS / "hdsg.prompt_packet.v2.schema.json").read_text(encoding="utf-8")))

    def packets(self):
        for mode in ("AUTOMATIC", "MORE_DETAIL", "REASSESSMENT"):
            for intent in ("FORWARD", "LEFT", "RIGHT"):
                for advisory in ("CLEAR", "CAUTION", "STOP"):
                    for lane_name, lane in LANES.items():
                        for scene_name, objects in SCENES.items():
                            name = f"{mode}, {intent}, {advisory}, {lane_name}, {scene_name}"
                            _, prompt = event(intent=intent, object_advisory=advisory, lane=lane,
                                              objects=objects, response_mode=mode)
                            yield name, prompt

    def test_every_packet_the_builder_produces_is_accepted(self):
        for name, prompt in self.packets():
            with self.subTest(name):
                errors = [error.message for error in self.validator.iter_errors(prompt)]
                self.assertEqual([], errors)

    def test_the_widest_scene_reaches_four_requirements(self):
        """The guard on the test above. Every packet validating proves nothing if none of them
        approaches the cap, and three of the four requirement kinds only appear together on a
        contested scene."""
        widest = max(len(prompt["requirements"]) for _, prompt in self.packets())
        self.assertEqual(4, widest)

    def test_a_redirect_demands_both_the_intended_and_the_chosen_sector(self):
        """The behaviour the two removed `role` values were drafted for.

        `REDIRECT_INTENDED` and `REDIRECT_SELECTED` were named in the design draft and never built,
        because the action binding covers both: on a redirect it carries the sector the person aimed
        at and the sector they are being sent to, under ALL_OF, so a clause naming only one of them
        does not satisfy it. The answer must therefore say why the intended way is refused and why
        the chosen one is better. Locked here because removing the two names on 23 August 2026 left
        nothing else recording that the case is handled.
        """
        _, prompt = event(intent="FORWARD", object_advisory="CLEAR",
                          lane=LANES["centre blocked, left open"], response_mode="AUTOMATIC")
        binding = next(item for item in prompt["requirements"]
                       if item["role"] == "ACTION_BINDING")
        self.assertEqual("ALL_OF", binding["match"])
        self.assertEqual(["sector:centre", "sector:left"], binding["fact_ids"])

    def test_no_requirement_is_empty(self):
        """A requirement with no facts demands a clause resting on nothing, which the gate cannot
        score and the reader cannot check."""
        for name, prompt in self.packets():
            with self.subTest(name):
                for requirement in prompt["requirements"]:
                    self.assertTrue(requirement["fact_ids"], requirement["requirement_id"])

    def test_every_required_fact_is_offered(self):
        """A requirement may only name a fact the packet also permits. Demanding a clause about a
        fact the model was never given is a failure the gate would record against the model."""
        for name, prompt in self.packets():
            with self.subTest(name):
                offered = {item["fact_id"] for item in prompt["permitted_facts"]}
                for requirement in prompt["requirements"]:
                    for fact_id in requirement["fact_ids"]:
                        self.assertIn(fact_id, offered, requirement["requirement_id"])


class ImageTransformTests(unittest.TestCase):
    """How the image sent to the model is recorded.

    The runtime overwrites the packet's transform with the command line settings after building it,
    at two sites, so what the schema sees is not what `build_prompt_packet` wrote. Both encodings
    the command line offers are checked here for that reason.
    """

    def setUp(self):
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")
        self.validator = jsonschema.Draft7Validator(json.loads(
            (SCHEMAS / "hdsg.prompt_packet.v2.schema.json").read_text(encoding="utf-8")))
        self.transform = image_transform()

    def test_both_encodings_are_recorded_in_a_form_the_schema_accepts(self):
        """A PNG attempt wrote `jpeg_quality: 70` until 23 August 2026, which the schema refuses:
        a PNG has no JPEG quality. PNG is no longer a command line choice but it remains the first
        thing tried when the model refuses the image, so the case is still reachable, and every
        packet in such a run would have failed validation with nothing saying so at the time.
        """
        _, prompt = event()
        for encoding, quality in (("jpeg", 70), ("png", None)):
            with self.subTest(encoding):
                written = self.transform(encoding, 70, 448)
                self.assertEqual(quality, written["jpeg_quality"])
                packet = copy.deepcopy(prompt)
                packet["image"]["transform"].update(written)
                self.assertEqual([], [e.message for e in self.validator.iter_errors(packet)])

    def test_a_smaller_fallback_image_is_recorded_at_its_own_size(self):
        """The last fallback is a 224 pixel PNG. The packet stated 448 whichever attempt answered,
        until the transform started being taken from the attempt itself."""
        written = self.transform("png", 0, 224)
        self.assertEqual(224, written["longest_side_px"])
        _, prompt = event()
        prompt["image"]["transform"].update(written)
        self.assertEqual([], [e.message for e in self.validator.iter_errors(prompt)])


if __name__ == "__main__":
    unittest.main()
