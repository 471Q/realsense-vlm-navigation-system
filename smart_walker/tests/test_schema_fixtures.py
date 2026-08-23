"""Every fixture against its schema.

Two properties, and they are different. A valid fixture must be accepted, which establishes that the
schema does not reject a record the runtime actually writes. An invalid fixture must be rejected,
which establishes that the schema still catches the one fault it was built around; a schema loosened
by accident goes on accepting everything valid and says nothing.

Ported from `schemas/validate_schema_fixtures.ps1` on 23 August 2026, which was deleted with it. That
script needed PowerShell 7 for `Test-Json -SchemaFile` and so never ran on this machine, read a
`schema-manifest.v4.json` that had not existed since the set moved to v5, and pointed at two test
files that no longer exist. It had stopped being able to run at all, and nothing noticed, which is
the argument for the checks living in the suite that runs on every change.

The valid fixtures are generated from the runtime by `schemas/generate_fixtures.py`, so accepting one
establishes something about the records the system emits rather than about a document written by
hand. The invalid fixtures stay hand-written: each exists to be rejected for one specific reason,
which is a property of the schema and not of the runtime.
"""

from __future__ import annotations

import json
import unittest

from support import SCHEMAS  # noqa: F401

VALID = SCHEMAS / "fixtures" / "valid"
INVALID = SCHEMAS / "fixtures" / "invalid"

# Where each invalid fixture is expected to fail, as a set of paths into the record. Two of the
# faults break a rule stated over a pair of fields and so land in both.
FAULT_LOCATIONS = {
    # A stationary object is not drawn, so a record claiming both contradicts itself.
    "hdsg.fact_packet.v2.stationary_box.json": {"objects/0/display_bounding_box"},
    # The automatic profile permits no visual observations.
    "hdsg.prompt_packet.v2.automatic_visuals.json": {
        "response_constraints/visual_only_observations_allowed",
        "response_constraints/max_visual_observations"},
    "hdsg.question_record.v1.unknown_stage.json": {"resolved_by"},
    "hdsg.question_route.v1.unknown_route.json": {"route"},
    # An accepted release cannot carry the reason a caption was refused.
    "hdsg.release.v2.accepted_fallback_code.json": {
        "verification/primary_reason_code", "verification/reason_codes/0"},
    # The caption carries what the model may write, and a motion decision is the runtime's.
    "hdsg.vlm_caption.v1.extra_action.json": {"(root)"},
}


def schema_for(fixture_name: str):
    """The schema a fixture belongs to, taken from its own name.

    A fixture is named `<record>.<version>[.<fault>].json`, so the first three dot separated parts
    name the record. Derived rather than listed in a table, because a table is a second place to
    forget when a fixture is added: the PowerShell script carried one and it had gone stale.
    """
    stem = fixture_name[:-len(".json")] if fixture_name.endswith(".json") else fixture_name
    record = ".".join(stem.split(".")[:3])
    return SCHEMAS / f"{record}.schema.json"


def validator(path):
    import jsonschema

    return jsonschema.Draft7Validator(json.loads(path.read_text(encoding="utf-8")))


class FixtureTests(unittest.TestCase):

    def setUp(self):
        try:
            import jsonschema  # noqa: F401
        except ImportError as error:  # pragma: no cover, depends on the environment
            raise unittest.SkipTest(f"jsonschema unavailable: {error}")

    def test_there_are_fixtures_to_check(self):
        """A guard on the two tests below. Both iterate a directory, so an empty directory or a
        changed layout would make them pass while checking nothing."""
        self.assertTrue(list(VALID.glob("*.json")))
        self.assertTrue(list(INVALID.glob("*.json")))

    def test_every_fixture_names_a_schema_that_exists(self):
        for fixture in sorted(list(VALID.glob("*.json")) + list(INVALID.glob("*.json"))):
            with self.subTest(fixture.name):
                self.assertTrue(schema_for(fixture.name).exists(),
                                f"{fixture.name} names no schema in {SCHEMAS}")

    def test_every_valid_fixture_is_accepted(self):
        for fixture in sorted(VALID.glob("*.json")):
            with self.subTest(fixture.name):
                errors = sorted(
                    validator(schema_for(fixture.name)).iter_errors(
                        json.loads(fixture.read_text(encoding="utf-8"))),
                    key=lambda error: list(error.absolute_path))
                self.assertEqual(
                    [], [f"{'/'.join(str(p) for p in e.absolute_path)}: {e.message}"
                         for e in errors])

    def test_every_invalid_fixture_is_rejected(self):
        """Each invalid fixture carries one fault. A schema that stops catching it has been loosened,
        and no valid fixture would reveal that."""
        for fixture in sorted(INVALID.glob("*.json")):
            with self.subTest(fixture.name):
                self.assertTrue(
                    validator(schema_for(fixture.name)).iter_errors(
                        json.loads(fixture.read_text(encoding="utf-8"))),
                    f"{fixture.name} was accepted and exists to be rejected")

    def test_every_invalid_fixture_is_rejected_for_its_own_fault(self):
        """The test above is satisfied by any failure at all, which is not what a fixture is for.

        Until 24 August 2026 four of the six were skeletons: the block under test carried the fault
        and every other block was an empty object, so they were refused for dozens of missing
        required properties. A schema loosened to admit the named fault would have gone on rejecting
        them, and the test above would have gone on passing while defending nothing.

        Each is now a valid record with one thing wrong with it, and the places it may be wrong are
        named here. A fault that stops being caught leaves the fixture accepted, and a fault that
        starts catching something else changes where the error lands.
        """
        for fixture in sorted(INVALID.glob("*.json")):
            with self.subTest(fixture.name):
                expected = FAULT_LOCATIONS[fixture.name]
                errors = validator(schema_for(fixture.name)).iter_errors(
                    json.loads(fixture.read_text(encoding="utf-8")))
                where = {"/".join(str(part) for part in error.absolute_path) or "(root)"
                         for error in errors}
                self.assertEqual(expected, where)

    def test_the_named_faults_cover_the_directory(self):
        """A guard on the table above. A fixture added without an entry would otherwise raise a
        KeyError inside a subTest and be reported as an error in one case rather than as a gap."""
        self.assertEqual(set(FAULT_LOCATIONS), {path.name for path in INVALID.glob("*.json")})


if __name__ == "__main__":
    unittest.main()
