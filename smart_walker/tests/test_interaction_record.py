"""What the interaction block says the person did.

The block records the intent in force, what triggered the event, which request was raised, and how
the person raised it. Its value is that an archived run can be read back and understood, so a field
stating something that did not happen is worse than an absent field, because it will be believed.

Two defects, both found on 23 August 2026 and both of the same kind. A typed question recorded
`control_id: "MORE_DETAIL"`, naming an on-screen control that nobody pressed, on every question in
the archive. And every field was constrained on its own and none against the others, so six
combinations the system cannot produce all validated against the schema.
"""

from __future__ import annotations

import copy
import json
import unittest

from support import SCHEMAS, fact_packet  # noqa: F401

# The four shapes an event can legitimately take, as the four sites that build one produce them.
LEGITIMATE = {
    "automatic guidance": {"request_id": "AUTO_GUIDANCE", "response_mode": "AUTOMATIC",
                           "input_method": "SYSTEM", "control_id": None},
    "more detail by button": {"request_id": "MORE_DETAIL", "response_mode": "MORE_DETAIL",
                              "input_method": "ONSCREEN_CONTROL", "control_id": "MORE_DETAIL"},
    "reassess by keyboard": {"request_id": "REASSESS", "response_mode": "REASSESSMENT",
                             "input_method": "KEYBOARD_SHORTCUT", "control_id": "REASSESS"},
    "typed question": {"request_id": "MORE_DETAIL", "response_mode": "MORE_DETAIL",
                       "input_method": "TYPED_QUESTION", "control_id": None},
}

IMPOSSIBLE = {
    "a reassess request in automatic mode":
        {"request_id": "REASSESS", "response_mode": "AUTOMATIC"},
    "an automatic request in more detail mode":
        {"request_id": "AUTO_GUIDANCE", "response_mode": "MORE_DETAIL"},
    "an automatic event naming a pressed control":
        {"input_method": "SYSTEM", "control_id": "REASSESS"},
    "a typed question naming a pressed control":
        {"input_method": "TYPED_QUESTION", "control_id": "MORE_DETAIL"},
    "a pressed control with no name":
        {"input_method": "ONSCREEN_CONTROL", "control_id": None},
    "a control disagreeing with the request it raised":
        {"input_method": "ONSCREEN_CONTROL", "request_id": "MORE_DETAIL",
         "control_id": "REASSESS"},
}


class SchemaConstraintTests(unittest.TestCase):
    """The relationships between the fields, not the fields alone."""

    def setUp(self):
        try:
            import jsonschema
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")
        self.validator = jsonschema.Draft7Validator(
            json.loads((SCHEMAS / "hdsg.fact_packet.v2.schema.json").read_text(encoding="utf-8")))
        self.packet = fact_packet()

    def with_interaction(self, overrides):
        packet = copy.deepcopy(self.packet)
        packet["interaction"].update(overrides)
        return packet

    def test_every_impossible_combination_is_refused(self):
        for name, overrides in IMPOSSIBLE.items():
            with self.subTest(name):
                self.assertTrue(list(self.validator.iter_errors(self.with_interaction(overrides))),
                                f"the schema accepts a record describing {name}")

    def test_every_legitimate_combination_is_accepted(self):
        """The other half, and the one that matters more. A constraint that refuses a real event
        turns a correct record into a validation failure, which is worse than the defect it fixes."""
        for name, overrides in LEGITIMATE.items():
            with self.subTest(name):
                errors = list(self.validator.iter_errors(self.with_interaction(overrides)))
                self.assertEqual([], [error.message for error in errors])

    def test_a_replayed_source_mode_is_refused(self):
        """`REPLAY` was admitted and nothing produced it. `hdsg_replay` reads archived fact packets
        and writes none, deliberately: recomputing one would introduce detector nondeterminism
        between the condition being scored and the record it is scored against."""
        packet = copy.deepcopy(self.packet)
        packet["identity"]["source_mode"] = "REPLAY"
        self.assertTrue(list(self.validator.iter_errors(packet)))


class TypedQuestionTests(unittest.TestCase):
    """A question typed into the chat box, as the runtime records it."""

    def test_a_typed_question_names_no_control(self):
        """`control_id` names the on-screen control that was activated, and typing activates none.
        It said MORE_DETAIL until 23 August 2026, so every question event in the archive records a
        button nobody pressed. It was redundant as well: `request_id` already carries MORE_DETAIL,
        for the frozen-enumeration reason the request catalogue gives.

        Asserted against the source rather than by running the interactive loop, which needs a
        camera. The value is a literal at one site.
        """
        source = (SCHEMAS.parent / "scripts" / "realsense_vlm_on_change_qwen.py").read_text(
            encoding="utf-8")
        self.assertIn('input_method="TYPED_QUESTION", control_id=None,', source)
        self.assertNotIn('input_method="TYPED_QUESTION", control_id="MORE_DETAIL"', source)


if __name__ == "__main__":
    unittest.main()
