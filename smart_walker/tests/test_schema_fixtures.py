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


if __name__ == "__main__":
    unittest.main()
