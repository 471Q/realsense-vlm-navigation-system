"""Generate the schema manifest's digests from the files it describes.

WHY THIS SCRIPT EXISTS

The manifest records a digest of every schema, grammar and configuration file in the frozen set, so
that a record in the archive can be checked against the exact contract in force when it was written.

Its digests were maintained by hand. Every edit to any described file meant remembering to recompute
one, and nothing failed if the recompute was forgotten. That is the arrangement that let the caption
grammar disagree with its recorded digest for two days, and on 23 August 2026 it produced eight
manual reissues in a single afternoon of auditing.

Nothing checked the manifest either. There was no test over it at all, so a stale digest was invisible
until somebody compared the files by hand.

This script computes every digest from the file it names, and `--check` fails when the manifest and
the files disagree. `tests/test_schema_manifest.py` runs that check, so the manifest cannot go stale
without the suite saying so, and keeping it current costs one command rather than attention.

The notes are not touched. They are the written record of why the set is what it is, and a generator
has nothing to say about them.

WHAT THIS DOES NOT DO

It does not issue a new schema set version. Version numbers are a claim about compatibility and are
a decision, not a computation.

USAGE

    python scripts/generate_manifest.py
    python scripts/generate_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

SCHEMAS = Path(__file__).resolve().parents[1] / "schemas"


def manifest_path() -> Path:
    """The manifest in the schemas directory.

    Found rather than named, so that issuing a new set version does not also mean editing this
    script. More than one is an error: two manifests means two answers to which contract is frozen.
    """
    candidates = sorted(SCHEMAS.glob("schema-manifest.v*.json"))
    if not candidates:
        raise SystemExit(f"no schema-manifest.v*.json in {SCHEMAS}")
    if len(candidates) > 1:
        raise SystemExit("more than one manifest: " + ", ".join(p.name for p in candidates))
    return candidates[0]


def described_files(node, out=None):
    """Every entry in the manifest that names a file and records a digest for it."""
    out = [] if out is None else out
    if isinstance(node, dict):
        if "path" in node and "sha256" in node:
            out.append(node)
        for value in node.values():
            described_files(value, out)
    elif isinstance(node, list):
        for value in node:
            described_files(value, out)
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if any recorded digest disagrees with its file")
    args = parser.parse_args(argv)

    path = manifest_path()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    entries = described_files(manifest)
    if not entries:
        raise SystemExit(f"{path.name} describes no files")

    missing, stale = [], []
    for entry in entries:
        target = (SCHEMAS / entry["path"]).resolve()
        if not target.exists():
            missing.append(entry["path"])
            continue
        actual = digest(target)
        if actual != entry["sha256"]:
            stale.append(entry["path"])
            entry["sha256"] = actual

    if missing:
        print(f"{path.name} names files that do not exist: " + ", ".join(missing), file=sys.stderr)
        return 1

    if args.check:
        if stale:
            print(f"{path.name} is out of step with " + ", ".join(stale)
                  + ".\nRegenerate it: python scripts/generate_manifest.py", file=sys.stderr)
            return 1
        print(f"{path.name} matches all {len(entries)} described files.")
        return 0

    if stale:
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Reissued {len(stale)} digest(s) in {path.name}: " + ", ".join(stale))
    else:
        print(f"{path.name} was already current across {len(entries)} files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
