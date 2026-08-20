"""Measures what the generative layer contributes over the deterministic layer alone.

`hdsg_baselines.py` defines three conditions, C0_VLM_ONLY, C1_GROUNDED_UNGATED and C2_FULL_HDSG.
All three include the generative component, so none of them answers the question an examiner asks
first: remove the vision-language model and what changes. This module supplies the missing arm.

    C3_DETERMINISTIC_ONLY   the complete architecture with no candidate accepted

The condition is an exact ablation rather than an approximation. Every archived event already
records its Full Fact Packet and its Restricted Prompt Packet, and `build_release` already produces
a deterministic rendering whenever a candidate is absent. Rebuilding each event with no candidate
therefore yields precisely the text the system would have shown with the model removed, computed
from the archive without a model, a recording or a rerun.

**What the comparison answers.** Chapter 2 section 2.5.6 requires the generative component to
compose language rather than select from responses written in advance.
`HDSG_VERIFIED_GENERATION_POLICY.md` section 2 showed that the templated design did not meet that
requirement, and it was removed on 21 August 2026 for that reason. The composed design meets it by
construction, since the model writes the sentences, so the open question is no longer whether the
model composes but whether what it composes tells the user anything the deterministic layer would
not have told them anyway. This module answers that over a whole run: how often an accepted caption
names a fact the fallback withheld, how often it only rephrases one the fallback already gave, and
how often it reaches past the requirement set entirely.

Nothing here reaches the walker display, and nothing here calls a model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Optional

try:
    from . import hdsg_runtime as hdsg
    from . import hdsg_questions as questions
except ImportError:  # invoked as a plain script rather than as part of the package
    import hdsg_runtime as hdsg  # type: ignore
    import hdsg_questions as questions  # type: ignore


CONDITION_C3 = "C3_DETERMINISTIC_ONLY"

# A visual observation is rendered by build_release as "Possible <label> is visible in the <bearing>."
# It is the one clause class the deterministic layer cannot produce at all, so it is counted apart
# from clauses that restate a measured fact in different words.
VISUAL_CLAUSE_RE = re.compile(r"^Possible .+ (?:is|are) visible in the (?:left|centre|right)\.$")

# Sentences end at a full stop followed by a space or the end of the text. Splitting this way rather
# than on any full stop keeps a decimal measurement such as "2.00 metres" in one piece.
SENTENCE_RE = re.compile(r"(?<=\.)\s+")


@dataclass
class EventContribution:
    """What the generative layer added to one released event.

    Two kinds of difference are distinguished, because they mean different things. A text that
    names a fact the deterministic rendering did not name carries information the deterministic
    layer withheld. A text that names the same facts in other words carries none, whatever the
    prose is like. Counting the second as contribution would report the generative layer as
    productive in proportion to how freely it rephrases, which is a measure of the model's variety
    rather than of what the user learns.

    Both measures read the release's declared identifiers rather than its prose, so neither varies
    with wording.
    """

    event_id: str
    gate_outcome: str
    released_text: str
    deterministic_text: str
    profile_text: str = ""
    added_clauses: list[str] = field(default_factory=list)
    dropped_clauses: list[str] = field(default_factory=list)
    added_facts: list[str] = field(default_factory=list)
    added_visual_clauses: int = 0
    beyond_profile_facts: list[str] = field(default_factory=list)

    @property
    def identical(self) -> bool:
        """True when the released text is exactly the deterministic text."""
        return not self.added_clauses and not self.dropped_clauses

    @property
    def informative(self) -> bool:
        """True when the release names a fact, or a visual observation, that the fallback did not."""
        return bool(self.added_facts) or self.added_visual_clauses > 0

    @property
    def reworded_only(self) -> bool:
        """True when the text differs but names nothing the deterministic rendering did not."""
        return not self.identical and not self.informative

    @property
    def beyond_profile(self) -> bool:
        """True when the release carries a clause the requirement construction did not call for.

        The stricter of the two measures, and the one that attributes contribution to the model
        rather than to the profile.
        """
        return bool(self.beyond_profile_facts)

    def as_record(self) -> dict:
        return {
            "event_id": self.event_id,
            "gate_outcome": self.gate_outcome,
            "identical": self.identical,
            "informative": self.informative,
            "reworded_only": self.reworded_only,
            "beyond_profile": self.beyond_profile,
            "added_clauses": list(self.added_clauses),
            "dropped_clauses": list(self.dropped_clauses),
            "added_facts": list(self.added_facts),
            "added_visual_clauses": self.added_visual_clauses,
            "beyond_profile_facts": list(self.beyond_profile_facts),
            "released_text": self.released_text,
            "deterministic_text": self.deterministic_text,
            "profile_text": self.profile_text,
        }


def release_body(release: Mapping[str, Any]) -> str:
    """Returns the explanatory text of a release, excluding the deterministic action sentence.

    The action text and the interaction prompt are produced by the rule engine in both conditions,
    so including them would report agreement the generative layer had no part in.
    """
    content = release["content"]
    parts = [str(content.get("reason_text") or "")]
    parts.extend(str(item) for item in content.get("additional_detail_texts") or [])
    return " ".join(part.strip() for part in parts if part and part.strip())


def split_clauses(text: str) -> list[str]:
    """Splits released text into sentences."""
    return [item.strip() for item in SENTENCE_RE.split(str(text).strip()) if item.strip()]


def profile_facts(prompt_packet: Mapping[str, Any]) -> set[str]:
    """Returns identifiers for the facts the requirement construction selected.

    The baseline for the stricter measure. A release naming only these added nothing the profile
    had not already chosen, however it worded them.

    This replaced a heuristic that took the opening three words of each sentence as its subject.
    That held only while both renderings were built from the same predicate table and placed the
    subject first, which stopped being true when the model began composing its own prose: "The way
    ahead narrows to 1.74 metres" and "The centre sector has limited clearance at 1.40 metres"
    describe one fact and share no opening. Attributing from the declared identifiers is exact, and
    it is the fix the heuristic's own documentation called for.
    """
    wanted: set[str] = set()
    for requirement in prompt_packet.get("requirements", []):
        wanted.update(str(item) for item in requirement.get("fact_ids", []))
    named: set[str] = set()
    for item in prompt_packet.get("permitted_facts", []):
        if str(item.get("fact_id")) not in wanted:
            continue
        measurement = item.get("measurement")
        if isinstance(measurement, Mapping) and measurement.get("measurement_id"):
            named.add(str(measurement["measurement_id"]))
    return named


def facts_named(release: Mapping[str, Any]) -> set[str]:
    """Returns identifiers for the facts a release actually described.

    Taken from the measurement substitutions the release records, which name every fact whose
    measurement was rendered into the text, together with any released visual observation. This is
    a structural record rather than a reading of the prose, so it does not vary with wording.

    A fact rendered without a measurement, such as a sector with no reliable clearance, leaves no
    substitution and is therefore invisible here. Such a fact is counted as named by neither
    condition, which understates rather than overstates the generative contribution.
    """
    evidence = release.get("evidence") or {}
    named = {
        str(item["measurement_id"])
        for item in evidence.get("measurement_substitutions") or []
        if item.get("measurement_id")
    }
    # The recorded identifiers already carry the visual: prefix, so they are taken as they are.
    # Adding another produced visual:visual:1, which was harmless while the value served only as a
    # set key and misleading the moment it was reported.
    named.update(str(item) for item in evidence.get("released_visual_observation_ids") or [])
    return named


def deterministic_release(fact_packet: Mapping[str, Any], prompt_packet: Mapping[str, Any],
                          release_id: str = "release_c3") -> dict:
    """Rebuilds one event with no candidate, giving the text the system shows without a model.

    `RG_MODEL_UNAVAILABLE` is used rather than a rejection code because the condition removes the
    model rather than rejecting its output, and the two record different candidate statuses.
    """
    return hdsg.build_release(
        fact_packet,
        prompt_packet,
        release_id=release_id,
        failure_codes=["RG_MODEL_UNAVAILABLE"],
    )


def profile_text(fact_packet: Mapping[str, Any], prompt_packet: Mapping[str, Any]) -> str:
    """Renders every fact the profile requires, deterministically and without a model.

    This is the baseline that isolates the model. `deterministic_release` renders the action
    binding only, because that is what `_fallback_reason` does, so a More detail candidate appears
    to add the sectors the More detail profile asked for. Those sectors were chosen by the
    requirement construction rather than by the model, and a renderer given the same requirement
    set produces them without a model at all. Comparing against this text therefore reports what
    the model contributed, while comparing against the fallback reports what the user would lose if
    the model were removed. Both are wanted, and they answer different questions.
    """
    fact_ids: list[str] = []
    for requirement in prompt_packet.get("requirements", []):
        fact_ids.extend(requirement.get("fact_ids", []))
    parts = [
        rendered for rendered in (
            questions.render_fact(fact_packet, fact_id)
            for fact_id in dict.fromkeys(fact_ids)
        ) if rendered
    ]
    return " ".join(parts)


def compare_event(fact_packet: Mapping[str, Any], prompt_packet: Mapping[str, Any],
                  released: Mapping[str, Any]) -> EventContribution:
    """Compares one archived release against its deterministic-only counterpart."""
    baseline = deterministic_release(fact_packet, prompt_packet)
    released_text = release_body(released)
    deterministic_text = release_body(baseline)
    profile_baseline = profile_text(fact_packet, prompt_packet)

    released_clauses = split_clauses(released_text)
    baseline_clauses = split_clauses(deterministic_text)
    baseline_set = set(baseline_clauses)
    released_set = set(released_clauses)

    added = [clause for clause in released_clauses if clause not in baseline_set]
    dropped = [clause for clause in baseline_clauses if clause not in released_set]
    visual = sum(1 for clause in added if VISUAL_CLAUSE_RE.match(clause))
    added_facts = sorted(facts_named(released) - facts_named(baseline))

    beyond_profile = sorted(facts_named(released) - profile_facts(prompt_packet))

    return EventContribution(
        event_id=released["identity"]["event_id"],
        gate_outcome=released["verification"]["gate_outcome"],
        released_text=released_text,
        deterministic_text=deterministic_text,
        profile_text=profile_baseline,
        added_clauses=added,
        dropped_clauses=dropped,
        added_facts=[item for item in added_facts if not item.startswith("visual:")],
        added_visual_clauses=visual,
        beyond_profile_facts=beyond_profile,
    )


def read_archive(telemetry_path: Path) -> tuple[dict, dict, list[dict]]:
    """Returns the fact packets, prompt packets and releases of a run, keyed by event.

    Interim releases carrying the pending placeholder are skipped, since they hold no candidate text
    and comparing them would report an absence the generative layer is not responsible for.
    """
    facts: dict[str, dict] = {}
    prompts: dict[str, dict] = {}
    releases: list[dict] = []
    for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        envelope = json.loads(line)
        record = envelope.get("record") or {}
        kind = envelope.get("record_type")
        if kind == "full_fact_packet":
            facts[record["identity"]["event_id"]] = record
        elif kind == "restricted_prompt_packet":
            prompts[record["routing"]["event_id"]] = record
        elif kind in {"authoritative_release", "question_release"}:
            codes = record.get("verification", {}).get("reason_codes") or []
            if "RG_GENERATION_PENDING" in codes or "RG_NO_INTENT_EXPRESSED" in codes:
                continue
            releases.append(record)
    return facts, prompts, releases


def compare_run(telemetry_path: Path) -> dict:
    """Compares every released event in a run against the deterministic-only condition."""
    facts, prompts, releases = read_archive(Path(telemetry_path))
    contributions: list[EventContribution] = []
    unpaired = 0
    for released in releases:
        event_id = released["identity"]["event_id"]
        fact_packet = facts.get(event_id)
        prompt_packet = prompts.get(event_id)
        if fact_packet is None or prompt_packet is None:
            # An event whose evidence is incomplete is counted rather than compared against a
            # substitute, since a rebuilt release from another event's facts is not this event.
            unpaired += 1
            continue
        contributions.append(compare_event(fact_packet, prompt_packet, released))
    return {
        "condition": CONDITION_C3,
        "released_events": len(releases),
        "compared": len(contributions),
        "unpaired": unpaired,
        "contributions": contributions,
        "summary": summarise(contributions),
    }


def summarise(contributions: Iterable[EventContribution]) -> dict:
    """Aggregates per-event contributions into the figures that answer the ablation question."""
    items = list(contributions)
    total = len(items)
    accepted = [item for item in items if item.gate_outcome == "ACCEPTED"]

    def rate(numerator: int, denominator: int) -> Optional[float]:
        return round(numerator / denominator, 3) if denominator else None

    accepted_informative = sum(1 for item in accepted if item.informative)
    accepted_identical = sum(1 for item in accepted if item.identical)
    accepted_reworded = sum(1 for item in accepted if item.reworded_only)
    accepted_beyond = sum(1 for item in accepted if item.beyond_profile)

    return {
        # The stricter measure, and the one that answers what the model contributed as opposed to
        # what the requirement construction contributed.
        "accepted_beyond_profile": accepted_beyond,
        "accepted_beyond_profile_rate": rate(accepted_beyond, len(accepted)),
        "beyond_profile_facts_total": sum(
            len(item.beyond_profile_facts) for item in items
        ),
        "events": total,
        "accepted_events": len(accepted),
        # The headline figure. An accepted candidate is informative only where it names a fact, or
        # a visual observation, that the deterministic rendering did not. Everything else is either
        # the identical text or the same facts in synonymous wording.
        "accepted_informative": accepted_informative,
        "accepted_informative_rate": rate(accepted_informative, len(accepted)),
        "accepted_identical_to_deterministic": accepted_identical,
        "accepted_reworded_only": accepted_reworded,
        "identical_to_deterministic": sum(1 for item in items if item.identical),
        "identical_rate": rate(sum(1 for item in items if item.identical), total),
        "added_facts_total": sum(len(item.added_facts) for item in items),
        "added_facts_per_accepted_event": (
            round(sum(len(item.added_facts) for item in accepted) / len(accepted), 3)
            if accepted else None
        ),
        "added_visual_clauses": sum(item.added_visual_clauses for item in items),
        "added_clauses_total": sum(len(item.added_clauses) for item in items),
        "dropped_clauses_total": sum(len(item.dropped_clauses) for item in items),
    }


def write_results(results: Mapping[str, Any], output_path: Path) -> Path:
    """Writes the per-event comparisons as JSONL, with the summary as the final record."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for item in results["contributions"]:
            stream.write(json.dumps(
                {"record_type": "contribution_event", "condition": results["condition"],
                 "record": item.as_record()},
                separators=(",", ":"), ensure_ascii=True,
            ) + "\n")
        stream.write(json.dumps(
            {"record_type": "contribution_summary", "record": {
                "condition": results["condition"],
                "released_events": results["released_events"],
                "compared": results["compared"],
                "unpaired": results["unpaired"],
                "summary": results["summary"],
            }},
            separators=(",", ":"), ensure_ascii=True,
        ) + "\n")
    return output_path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Measure what the generative layer adds over the deterministic layer alone"
    )
    parser.add_argument("--telemetry", type=Path, required=True, help="the run's JSONL archive")
    parser.add_argument("--out", type=Path, default=None,
                        help="where to write the per-event comparisons")
    parser.add_argument("--show", type=int, default=3,
                        help="print this many accepted events that named nothing new")
    args = parser.parse_args(argv)

    results = compare_run(args.telemetry)
    summary = results["summary"]
    print(f"[contribution] released events: {results['released_events']}, "
          f"compared: {results['compared']}, unpaired: {results['unpaired']}")
    print(f"[contribution] accepted candidates: {summary['accepted_events']}")
    print(f"[contribution] against the deterministic fallback, informative: "
          f"{summary['accepted_informative']}/{summary['accepted_events']} "
          f"({summary['accepted_informative_rate']})")
    print(f"[contribution] against the same requirement set, beyond it: "
          f"{summary['accepted_beyond_profile']}/{summary['accepted_events']} "
          f"({summary['accepted_beyond_profile_rate']})")
    print(f"[contribution]   of the remainder: "
          f"{summary['accepted_identical_to_deterministic']} identical to the deterministic text, "
          f"{summary['accepted_reworded_only']} the same facts in other words")
    print(f"[contribution] facts named that the deterministic layer did not: "
          f"{summary['added_facts_total']} "
          f"({summary['added_facts_per_accepted_event']} per accepted event)")
    print(f"[contribution] visual observations released: {summary['added_visual_clauses']}")

    shown = 0
    for item in results["contributions"]:
        if shown >= args.show or item.informative or item.gate_outcome != "ACCEPTED":
            continue
        state = "identical to" if item.identical else "the same facts as"
        print(f"\n[contribution] {item.event_id}: accepted, {state} the deterministic text")
        print(f"    {item.released_text}")
        shown += 1

    if args.out:
        print(f"\n[contribution] written to {write_results(results, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
