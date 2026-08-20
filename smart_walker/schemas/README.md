# Frozen schema set: `hdsg.schemas.v3`

The six record contracts the architecture is enforced through, their fixtures, the three grammars
the runtime loads, and a manifest recording the SHA-256 of every one.

| Record | Schema | Grammar |
|---|---|---|
| Full Fact Packet | `hdsg.fact_packet.v2` | — |
| Restricted Prompt Packet | `hdsg.prompt_packet.v2` | — |
| VLM Candidate, templated | `hdsg.vlm_candidate.v1` | `hdsg.vlm_candidate.v1.gbnf` |
| VLM Caption, composed | `hdsg.vlm_caption.v1` | `hdsg.vlm_caption.v1.gbnf` |
| Question Route | `hdsg.question_route.v1` | `hdsg.question_route.v1.gbnf` |
| Authoritative Release Object | `hdsg.release.v2` | — |

## What v3 changed and why

Three record contracts advanced. The fact packet records a typed question as `TYPED_QUESTION`
rather than as the on-screen control whose profile it borrows. The prompt packet records which of
the two candidate contracts was actually constrained, since `expected_response_schema` and
`constraint_id` were pinned to the templated contract and therefore misreported every composed run.
The release admits a composed candidate and carries `RG_STATED_VALUE_MISMATCH`, the one reason code
the composed design adds.

Two records are new. `hdsg.vlm_caption.v1` is the composed contract: a caption in the model's own
words together with a declaration of every numeric value it stated and the fact each came from.
`hdsg.question_route.v1` is the Tier 1 classifier's output, which existed as a grammar with no
schema and was absent from the v2 manifest.

`hdsg.vlm_candidate.v1` is unchanged as a record and keeps its identifier. Its grammar changed:
`candidate_observation_id` and `proposed_label` were free strings while the validator required
`visual:N` and a lower-case label.

## Seven conformance breaks found while issuing v3

None was detected by the v2 freeze, and the reason is the same in every case: the fixtures were
written by hand to match the schemas. A hand-written fixture agrees with its schema by construction
and establishes nothing about the records the system produces.

1. The prompt packet carried `approved_text_templates` while the schema forbade extra properties.
   True of every packet since v1.
2. `RG_NO_INTENT_EXPRESSED` and `RG_GENERATION_PENDING` were absent from the reason code
   enumeration, so every idle and every interim release was non-conformant from 19 August 2026.
3. `IDLE_NO_INTENT` and `GENERATION_PENDING` were absent from the interaction state enumeration,
   for the same reason.
4. The idle release carried empty `action_text`, `reason_text` and `caption_text` where the schema
   required a minimum length. The state legitimately has no text to state.
5. The idle release carried an empty `action_binding_fact_ids` where the schema required an entry.
   The state legitimately has no binding fact.
6. The More detail profile's visual-observation maximum was raised from one to two on 20 August
   2026 and the schema still pinned it to one.
7. A permitted fact could carry `state: UNKNOWN`, produced whenever a distance bin is unavailable,
   which the state enumerations did not admit.

## What prevents a recurrence

`tests/test_schema_conformance.py` builds records through the runtime and validates those. It
covers the typed-question fact packet, both prompt packet contracts, an accepted composed release,
a release rejected for a wrong declared value, the idle and pending releases, and the templated
arm. It also asserts that every reason code the runtime can emit is admissible, and that the
profile limits agree with the schema's conditionals.

The valid fixtures for the fact packet, prompt packet, candidate and release are generated from the
runtime rather than written by hand, so they cannot drift from what the system emits. The invalid
fixtures remain hand-written, since each exists to be rejected for one specific reason.

`tests/test_schema_freeze.py` covers manifest drift and requires that every grammar in `config/`
is hashed. The v2 manifest hashed one grammar while the runtime had acquired three, which left the
routing and caption constraints free to drift from their validators undetected.

## Running the checks

The Python tests run anywhere and need `jsonschema` for the conformance file; without it those
tests skip rather than fail.

```
python -m unittest discover -s tests
```

`validate_schema_fixtures.ps1` drives the fixtures through `Test-Json` under PowerShell 7, which is
the engine the freeze was recorded against. It and the Python tests check different things and
neither replaces the other.
