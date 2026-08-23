# Frozen schema set: `hdsg.schemas.v5`

The five record contracts the architecture is enforced through, their fixtures, the two grammars
the runtime loads, the request catalogue holding every system prompt it sends, and a manifest
recording the SHA-256 of every one.

| Record | Schema | Grammar |
|---|---|---|
| Full Fact Packet | `hdsg.fact_packet.v2` | — |
| Restricted Prompt Packet | `hdsg.prompt_packet.v2` | — |
| VLM Caption | `hdsg.vlm_caption.v1` | `hdsg.vlm_caption.v1.gbnf` |
| Question Route | `hdsg.question_route.v1` | `hdsg.question_route.v1.gbnf` |
| Question Record | `hdsg.question_record.v1` | — |
| Authoritative Release Object | `hdsg.release.v2` | — |

## What v5 changed and why

The gate now reads the caption's prose for one purpose. A clause stating a number must name the
fact that number was declared against, so a caption cannot declare all three sector clearances
correctly and write each one against the wrong sector. That caption satisfied every other check
and was false in every clause. Attribution is still taken from the declaration, so the reading can
only refuse and never admit.

`RG_SUBJECT_MISMATCH` carries the failure. It was retired with the templated contract a day
earlier, where it meant a clause whose subject was not the fact the clause was chosen for. The
enumeration is unchanged; only that member's description is corrected.

v5 also settles how a measurement is compared. A caption states it as the deterministic renderer
displays it, character for character, so `1.7449 metres` and `two metres` are both refused against
a measurement displayed as `1.74 metres` and `2.00 metres` respectively. The full list is in the
manifest notes.

## What v4 changed and why

The templated candidate contract is removed. Under it the runtime built a list of controlled
sentences from a fixed table of predicates, one per fact and state, and the model selected among
them. The released wording was therefore the runtime's, and the model's contribution was the choice
between "has limited clearance" and "is constrained". A vision-language model reduced to selecting
an index is not the thing the thesis is about, and the arm cost a schema, a grammar, a validator and
two fixtures to keep in step with the contract that does execute.

`hdsg.vlm_candidate.v1` and its grammar are gone, along with `validate_candidate`,
`parse_candidate`, `prompt_packet_text`, the approved-clause builder and the predicate table.
`build_release` no longer takes a candidate: it renders the deterministic account for the idle
state, the pending state and every rejection, and `hdsg_composed.build_composed_release` calls it
for the authority block before substituting an accepted caption's content.

The generative layer is now the composed contract alone. The model writes the caption in its own
words with the measured values in the prose, declares each value it stated, and the gate compares
every declaration against the Fact Packet.

## What was retained for the archive, and why none of it is retained now

The set carried three retired things so that records archived before 21 August 2026 would still
validate, a frozen schema that cannot validate its own archive being of no use as evidence:

- `hdsg.vlm_candidate.v1` in `expected_response_schema`, `constraint_id` and `candidate_schema`;
- `approved_text_templates` on a permitted fact, as an optional property;
- six reason codes that screened a sentence the runtime had written:
  `RG_CLAUSE_FORMAT_INVALID`, `RG_REQUIRED_FACT_MISSING`, `RG_PLACEHOLDER_INVALID`,
  `RG_MOVEMENT_FACT_REQUIRED`, `RG_STATE_EXPRESSION_MISMATCH` and `RG_NEGATION_DETECTED`.

Those archives were deleted on 23 August 2026. The first two were removed from the set the same day
and the six codes on 24 August, the reason for keeping any of them having expired with the records
they were kept for. The one surviving run is from 22 August and contains no occurrence of any.

A seventh code, `RG_SUBJECT_MISMATCH`, was retired with the six and revived on 22 August 2026 for
the composed contract, where it means a clause states a number without naming the fact the number
was declared against. It stays.

`RG_REQUIRED_FACT_MISSING` has no composed equivalent by design. Establishing that free prose
covered a required fact would need the linguistic inference the architecture excludes from the
safety boundary, which is the same reason the model declares its values rather than the gate
parsing them out.

The freeze test binds the remaining direction: every code the runtime can emit must appear in the
enumeration.

## Two defects found while issuing v4

Both were carried over rather than introduced, and both were invisible for the same reason as the
provenance defects v3 fixed: a field populated with the wrong value rather than left absent.

1. The prompt packet recorded `max_tokens` from the templated budget while the composed call sent
   the caption budget, so every packet recorded 220 where 400 was sent. The third defect of that
   class, after the system prompt digest and the constraint digest.
2. `hdsg_contribution.facts_named` prefixed `visual:` onto identifiers that already carried it,
   yielding `visual:visual:1`. Harmless while the value served only as a set key.

One check would have been lost silently. The templated validator screened a visual observation's
label against the action, commentary and prohibited-instruction expressions and the composed
validator did not, so the label pattern, which admits spaces and therefore admits a phrase, would
have been the only thing standing between an instruction and the display. The screen moved into
`hdsg_composed` with the removal.

## One measure had to be rebuilt

`hdsg_contribution` decided whether a released clause reached beyond its requirement set by taking
each sentence's opening three words as its subject. That held only while both renderings came from
the same predicate table and placed the subject first. Free prose defeats it: "The way ahead narrows
to 1.40 metres" and "The centre sector has limited clearance at 1.40 metres" state one fact and
share no opening, so every composed caption would have been reported as reaching beyond its profile.
The measure now compares the release's declared identifiers against the identifiers the requirement
construction selected, which is exact and is the fix the heuristic's own documentation called for.

## What prevents a recurrence

`tests/test_schema_conformance.py` builds records through the runtime and validates those. It
covers the typed-question fact packet, the prompt packet, an accepted composed release, a release
rejected for a wrong declared value, and the idle and pending releases. It also asserts that every
reason code the runtime can emit is admissible, that the profile limits agree with the schema's
conditionals, and that the catalogue holds every system prompt the runtime sends.

The valid fixtures are generated by `generate_fixtures.py`, which builds one event through the
runtime and writes the fact packet, prompt packet, caption and release it produces. A hand-written
fixture agrees with its schema by construction and establishes nothing about the records the system
produces, which is how seven conformance breaks survived the v2 freeze, and how the prompt packet
fixture went on carrying `approved_text_templates` after the runtime stopped emitting it. The
invalid fixtures remain hand-written, since each exists to be rejected for one specific reason,
which is a property of the schema rather than of the runtime.

```
python schemas/generate_fixtures.py
```

`tests/test_schema_freeze.py` covers manifest drift, requires that every grammar in `config/` is
hashed, and asserts that no file in the set describes a contract the runtime cannot produce.

## Running the checks

The Python tests run anywhere and need `jsonschema` for the conformance file; without it those
tests skip rather than fail.

```
python -m unittest discover -s tests
```

`validate_schema_fixtures.ps1` drives the fixtures through `Test-Json` under PowerShell 7, which is
the engine the freeze was recorded against. It and the Python tests check different things and
neither replaces the other.
