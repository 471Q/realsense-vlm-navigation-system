# HDSG executable schemas

The executable record contracts for the HDSG implementation. Schema set `hdsg.schemas.v2`,
recorded in `schema-manifest.v2.json`.

| Record | JSON Schema | Role |
|---|---|---|
| Full Fact Packet | `hdsg.fact_packet.v1.schema.json` | Complete authoritative pre-VLM evidence |
| Restricted Prompt Packet | `hdsg.prompt_packet.v1.schema.json` | Restricted and traceable VLM input |
| VLM Candidate | `hdsg.vlm_candidate.v1.schema.json` | Constrained reason-only model output |
| Authoritative Release Object | `hdsg.release.v1.schema.json` | Sole user-visible release boundary |

The llama.cpp GBNF grammar expresses the same outer candidate shape for the local model path. It
is **not** stored here: it lives at `../config/hdsg.vlm_candidate.v1.gbnf`, where the runtime
loads it, and the manifest records that path. Keeping one copy is deliberate. Under
`hdsg.schemas.v1` the grammar existed both here and in the runtime configuration, the two drifted
from the recorded digest, and nothing detected it.

The JSON Schema remains the authoritative structural contract. Profile limits and cross-record
checks are enforced again by the release gate, because a generation grammar cannot establish
identity, freshness, fact permission, placeholder resolution or semantic agreement.

Schemas use JSON Schema Draft 7 with self-contained local definitions. Stable schema identifiers
use the `urn:hdsg:schema:*` namespace, while runtime records carry the approved `hdsg.*.v1` values
in `schema_version`.

## Why the set is v2 and the records are still v1

No record contract changed. The four schemas are byte-identical to their `hdsg.schemas.v1` form
and keep their v1 record identifiers, which is why `hdsg_runtime.py`'s schema constants are
unchanged. The set version advanced because the grammar inside it changed after the v1 freeze
while the v1 manifest was never regenerated, and `HDSG_EXECUTABLE_SCHEMA_FREEZE.md` requires a new
set version rather than a corrected digest inside the old one.

## Checks

Two complementary checks, neither of which replaces the other.

```powershell
# Full JSON Schema validation of all eight fixtures. Requires PowerShell 7.
& .\schemas\validate_schema_fixtures.ps1
```

```powershell
# Manifest drift and the runtime validator binding. Runs anywhere, no dependencies.
python -m unittest tests.test_schema_freeze -v
```

The PowerShell script validates record shape against the schemas using `Test-Json`, the engine the
freeze was recorded against. The Python tests verify that every frozen file still matches its
recorded digest, that the recorded grammar is the one the runtime loads, and that the frozen
candidate fixtures produce the same accept or reject outcome through `hdsg_runtime.validate_candidate`
as they do through the schema engine. The second exists because agreement between the runtime
validator and the frozen schema was previously established only by inspection.

The valid fixtures describe one coherent event across all four records. The invalid fixtures cover
a stationary object exposing a user-facing box, an automatic prompt profile permitting visual-only
observations, a VLM Candidate adding an action field, and an accepted release carrying a rejection
code.

Pilot-calibrated values remain nullable where the approved design requires measurement before
freeze. Formal evaluation records must carry the frozen values rather than the design-stage nulls.
