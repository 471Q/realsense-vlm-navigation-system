"""Measures what the test suite detects, rather than how much of it there is.

    python tests/mutation_audit.py            # every target module
    python tests/mutation_audit.py composed   # one module, by substring

The suite that preceded this one passed while ten of the twenty-two guards in the composed gate and
seven of the eight threshold boundaries in the deterministic layer could be deleted or shifted
without a single failure. Every defect those checks existed to catch therefore had to be found by
reading the source. A green suite over code whose checks can be removed reports the absence of
tests, not the absence of defects, and nothing in a pass or a test count distinguishes the two.

This harness breaks one thing at a time and asks whether any test notices. A survivor is a line the
suite does not constrain. Two mutation classes are applied, chosen because they are the two the old
suite proved blind to:

    guard     a reason-code append is replaced by pass, deleting one check
    boundary  a comparison operator is loosened by one, shifting a threshold

The run is slow by nature, one full suite per mutation, so it is a command rather than a test. It is
the acceptance criterion for a change to a checked module: add a guard, add the test that fails when
that guard is removed, and prove it here.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Compiled bytecode must never be reused between mutations. CPython treats a cached .pyc as current
# when the source's modification time and size both match, and a mutation harness violates both
# assumptions at once: it rewrites one line many times a second, and two mutations of equal-length
# lines produce files of identical size. Two guards on this module differ only in which line the
# same statement sits on, so mutating either produced a byte-identical file size within the same
# clock second, and the second run executed the bytecode compiled for the first.
#
# That reported a guard as unconstrained when a test did constrain it. The reverse is the dangerous
# direction and the reason this is not merely untidy: a stale cache can equally report a mutation as
# caught when nothing caught it, which is a harness certifying protection that does not exist.
#
# Caching is therefore switched off in the child and every existing cache directory is removed
# before the run, so each mutation is compiled from the source on disk.
_CHILD_ENV = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}

GUARD_RE = re.compile(r"^\s*errors\.append\(\"RG_[A-Z_]+\"\)\s*$")

# Loosening a comparison by one is the smallest change that moves a threshold. A test that pins the
# behaviour either side of the boundary fails; a test that only exercises the middle does not.
BOUNDARY_SWAPS = (("<=", "<"), (">=", ">"), (" < ", " <= "), (" > ", " >= "))

# Lines carrying a measured quantity. Restricted so the run stays about the policy thresholds rather
# than every comparison in the file.
BOUNDARY_HINTS = ("threshold", "_m", "clearance", "tolerance", "distance", "SEVERITY", "severity")

TARGETS = ("scripts/hdsg_runtime.py", "scripts/hdsg_composed.py", "scripts/hdsg_questions.py")


def _mutations(lines: list[str]) -> list[tuple[int, str, str]]:
    """Returns (line index, description, replacement line) for every mutation of a module."""
    found: list[tuple[int, str, str]] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or not stripped:
            continue
        if GUARD_RE.match(line):
            indent = " " * (len(line) - len(line.lstrip()))
            found.append((index, "guard   " + stripped, indent + "pass  # mutated\n"))
            continue
        if not any(hint in line for hint in BOUNDARY_HINTS):
            continue
        for before, after in BOUNDARY_SWAPS:
            if before in line:
                found.append((index, f"boundary {before.strip()!r}->{after.strip()!r} {stripped}",
                              line.replace(before, after, 1)))
                break
    return found


def _clear_caches() -> None:
    for cache in ROOT.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)


def audit(relative: str) -> list[str]:
    _clear_caches()
    path = ROOT / relative
    original = path.read_text(encoding="utf-8")
    lines = original.splitlines(keepends=True)
    mutations = _mutations(lines)
    survivors: list[str] = []
    print(f"\n{relative}: {len(mutations)} mutations")
    try:
        for index, description, replacement in mutations:
            patched = list(lines)
            patched[index] = replacement
            path.write_text("".join(patched), encoding="utf-8")
            passed = subprocess.run(
                [sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests"],
                cwd=ROOT, capture_output=True, env=_CHILD_ENV,
            ).returncode == 0
            if passed:
                survivors.append(f"{relative}:{index + 1}  {description}")
                print(f"  SURVIVED  line {index + 1}: {description[:74]}")
    finally:
        # Restored even on interrupt. A harness that can leave the source mutated is worse than none.
        path.write_text(original, encoding="utf-8")
    print(f"  {len(survivors)} survived of {len(mutations)}")
    return survivors


def main() -> int:
    wanted = sys.argv[1] if len(sys.argv) > 1 else ""
    targets = [name for name in TARGETS if wanted in name]
    if not targets:
        print(f"no target matches {wanted!r}; known: {', '.join(TARGETS)}")
        return 2
    survivors: list[str] = []
    for name in targets:
        survivors.extend(audit(name))
    print()
    if survivors:
        print(f"{len(survivors)} unconstrained lines. Each is a check no test protects:")
        for item in survivors:
            print(f"  {item}")
        return 1
    print("every mutation was caught")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
