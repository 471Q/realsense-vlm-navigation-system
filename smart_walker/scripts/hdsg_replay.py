"""Replays a recorded run through the offline comparison conditions and scores it.

Chapter 5 section 5.2.2 replays each base scene under C0_VLM_ONLY, C1_GROUNDED_UNGATED and
C2_FULL_HDSG "using the same observation and intent". This module supplies that replay for the
two offline conditions.

**Why the Fact Packet comes from the archive rather than being recomputed.** Section 5.6.1 scores
the offline responses "against the same event's Full Fact Packet and deterministic oracle". The
Fact Packet recorded during the live run *is* that record: it holds the measurements, the intent,
the deterministic decision and the binding facts as they stood. Recomputing it from the recorded
frames would introduce detector nondeterminism between the condition being scored and the record
it is scored against, and would make a replay disagree with the run it is meant to reproduce.
The recording supplies the image the model sees; the archive supplies what the walker measured.

**C2 is not re-run here.** Its released output is already in the archive, produced by the live
release path under the frozen configuration. Re-running it would sample a fresh candidate from a
non-deterministic model and would not be the release the archive records.

Nothing in this module can reach the walker display, per section 5.2.3.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Optional

try:
    from . import hdsg_baselines as baselines
    from . import hdsg_recording as recording
except ImportError:  # invoked as a plain script rather than as part of the package
    import hdsg_baselines as baselines  # type: ignore
    import hdsg_recording as recording  # type: ignore


# A responder takes (prompt, colour frame) and returns the model's raw reply. Injected rather
# than constructed here so the replay can be exercised without a model server, and so a run can
# be repeated against a different endpoint without touching this module.
Responder = Callable[[str, Any], str]


@dataclass
class ReplayEvent:
    """One archived event paired with the frame that produced it."""

    fact_packet: dict
    colour: Any
    depth_m: Any
    observation_id: str


def read_events(telemetry_path: Path) -> list[dict]:
    """Returns the Full Fact Packets from a telemetry archive, in recorded order."""
    packets: list[dict] = []
    for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        envelope = json.loads(line)
        if envelope.get("record_type") == "full_fact_packet":
            packets.append(envelope["record"])
    return packets


def read_releases(telemetry_path: Path) -> dict[str, dict]:
    """Returns the archived C2 releases, keyed by event identifier.

    Interim releases are skipped: they carry the pending placeholder rather than the scored
    output, and section 5.6.1 scores the Authoritative Release Object.
    """
    releases: dict[str, dict] = {}
    for line in Path(telemetry_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        envelope = json.loads(line)
        if envelope.get("record_type") != "authoritative_release":
            continue
        record = envelope["record"]
        releases[record["identity"]["event_id"]] = record
    return releases


def pair_with_recording(packets: list[dict], recording_dir: Path) -> Iterator[ReplayEvent]:
    """Yields archived events that have a recorded frame, in order.

    An event whose frame is missing is skipped rather than replayed against a substitute, since
    a condition scored on a different observation is not a matched comparison.
    """
    reader = recording.ObservationReader(recording_dir)
    by_id = {entry["observation_id"]: entry for entry in reader.entries()}
    for packet in packets:
        observation_id = packet["identity"]["observation_id"]
        entry = by_id.get(observation_id)
        if entry is None:
            continue
        colour, depth_m = reader.load(entry)
        yield ReplayEvent(packet, colour, depth_m, observation_id)


def replay_event(event: ReplayEvent, condition: str, respond: Responder) -> dict:
    """Runs one offline condition over one archived event and scores the reply."""
    prompt = baselines.build_baseline_prompt(condition, event.fact_packet)
    try:
        raw = respond(prompt, event.colour)
    except Exception as error:  # a model or transport failure is an unscoreable event, not a zero
        scored = baselines.score_baseline_event(
            condition, None, event.fact_packet, parse_failure=f"responder failed: {error}"
        )
        scored["raw_response"] = None
        return scored

    response, failure = baselines.parse_baseline_response(raw)
    scored = baselines.score_baseline_event(
        condition, response, event.fact_packet, parse_failure=failure
    )
    # The raw reply is retained for the same reason the live path retains it: a rejected or
    # unreadable response is precisely the case where the text is the evidence.
    scored["raw_response"] = raw
    scored["observation_id"] = event.observation_id
    return scored


def replay_run(telemetry_path: Path, recording_dir: Path, respond: Responder,
               conditions: tuple[str, ...] = baselines.OFFLINE_CONDITIONS,
               limit: Optional[int] = None) -> dict:
    """Replays an archived run under the offline conditions and returns scored results.

    Returns the per-event scores and the per-condition summaries that populate Chapter 5
    Table 5-9 and Table 5-11.
    """
    for condition in conditions:
        if condition not in baselines.OFFLINE_CONDITIONS:
            raise ValueError(f"{condition} cannot be replayed offline")

    packets = read_events(Path(telemetry_path))
    events = list(pair_with_recording(packets, Path(recording_dir)))
    if limit is not None:
        events = events[:limit]

    scored: dict[str, list[dict]] = {condition: [] for condition in conditions}
    for event in events:
        for condition in conditions:
            scored[condition].append(replay_event(event, condition, respond))

    return {
        "archived_events": len(packets),
        "replayed_events": len(events),
        "skipped_without_frame": len(packets) - len(events),
        "conditions": list(conditions),
        "scored": scored,
        "summary": {
            condition: baselines.summarise_condition(results)
            for condition, results in scored.items()
        },
    }


def write_results(results: Mapping[str, Any], output_path: Path) -> Path:
    """Writes the scored population as JSONL, with the summaries as the final record."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        for condition, entries in results["scored"].items():
            for entry in entries:
                stream.write(json.dumps(
                    {"record_type": "baseline_event", "condition": condition, "record": entry},
                    separators=(",", ":"), ensure_ascii=True,
                ) + "\n")
        stream.write(json.dumps(
            {"record_type": "baseline_summary", "record": {
                "archived_events": results["archived_events"],
                "replayed_events": results["replayed_events"],
                "skipped_without_frame": results["skipped_without_frame"],
                "summary": results["summary"],
            }},
            separators=(",", ":"), ensure_ascii=True,
        ) + "\n")
    return output_path


def http_responder(endpoint: str, model: str, *, temperature: float = 0.2,
                   top_p: float = 0.9, max_tokens: int = 400,
                   image_size: int = 448, jpeg_quality: int = 70,
                   timeout_s: float = 60.0) -> Responder:
    """Builds a responder that calls an OpenAI-compatible endpoint.

    No grammar is attached. The offline conditions are unconstrained by definition, and
    constraining their output would make them a weaker form of C2 rather than a comparison
    against it.
    """
    import base64

    import cv2
    import requests

    session = requests.Session()

    def respond(prompt: str, colour: Any) -> str:
        height, width = colour.shape[:2]
        scale = float(image_size) / float(max(height, width))
        if scale < 1.0:
            colour = cv2.resize(
                colour, (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", colour, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
        if not ok:
            raise RuntimeError("could not encode the replay frame")
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": False,
        }
        reply = session.post(
            endpoint.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout_s
        )
        if reply.status_code >= 400:
            raise RuntimeError(f"replay request failed: {reply.status_code} {reply.text[:200]}")
        return reply.json()["choices"][0]["message"]["content"]

    return respond


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Replay a recorded HDSG run through the offline comparison conditions"
    )
    parser.add_argument("--telemetry", type=Path, required=True,
                        help="the run's JSONL archive")
    parser.add_argument("--recording", type=Path, required=True,
                        help="the run's RGB-D recording directory")
    parser.add_argument("--out", type=Path, required=True,
                        help="where to write the scored population")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default="qwen3-vl-4b-instruct")
    parser.add_argument("--conditions", nargs="+", default=list(baselines.OFFLINE_CONDITIONS),
                        choices=list(baselines.OFFLINE_CONDITIONS))
    parser.add_argument("--limit", type=int, default=None,
                        help="replay only the first N events, for a smoke run")
    args = parser.parse_args(argv)

    responder = http_responder(args.endpoint, args.model)
    results = replay_run(
        args.telemetry, args.recording, responder,
        conditions=tuple(args.conditions), limit=args.limit,
    )
    path = write_results(results, args.out)

    print(f"[replay] archived events: {results['archived_events']}")
    print(f"[replay] replayed: {results['replayed_events']}, "
          f"skipped without a frame: {results['skipped_without_frame']}")
    for condition, summary in results["summary"].items():
        binding = summary["binding_causal_reason"]
        agreement = summary["guidance_agreement"]
        print(f"[replay] {condition}: "
              f"binding reason {binding['n']}/{binding['d']}, "
              f"guidance agreement {agreement['n']}/{agreement['d']}, "
              f"parse failures {summary['parse_failures']}")
    print(f"[replay] written to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
