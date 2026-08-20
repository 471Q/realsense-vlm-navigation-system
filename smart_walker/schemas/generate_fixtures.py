"""Regenerates the valid fixtures from the runtime.

A fixture written by hand agrees with its schema by construction and establishes nothing about the
records the system emits. Seven conformance breaks survived the v2 freeze for that reason, and the
prompt packet fixture then carried `approved_text_templates` and named the templated contract for a
day after the runtime stopped producing either. Generating the fixtures removes the gap between what
the frozen set is checked against and what the system writes.

The invalid fixtures stay hand-written. Each exists to be rejected for one specific reason, which is
a property of the schema rather than of the runtime, and a generator cannot express it.

    python schemas/generate_fixtures.py

Identifiers and timestamps are fixed so that regenerating an unchanged runtime produces an unchanged
file and the manifest digests do not churn.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

VALID = Path(__file__).resolve().parent / "fixtures" / "valid"
CONFIG = ROOT / "config"

# One scene, carried through every record, so the fixtures form a consistent event rather than four
# unrelated documents. The centre sector is constrained and the right sector is blocked, which is
# the case that exercises sector selection rather than a plain proceed.
DEPTHS = [3.13, 1.74, 0.55]
STATUSES = ["CLEAR", "CONSTRAINED", "BLOCKED"]

CAPTION = "The way ahead narrows to 1.74 metres, with 3.13 metres of space to the left."


def build_event():
    sectors = hdsg.sectors_from_lane_state({"depths": DEPTHS, "status": STATUSES})
    authority = hdsg.determine_authority(
        "FORWARD", "CAUTION", sectors,
        previous_selected_sector=None, sector_choice_tolerance_m=0.10,
    )
    fact_packet = hdsg.build_fact_packet(
        run_id="run_fixture", event_id="evt_fixture", observation_id="obs_fixture",
        ticket_id=None, timestamp_ms=1.0, intent="FORWARD", trigger_type="USER_REQUESTED",
        request_id="MORE_DETAIL", response_mode="MORE_DETAIL", previous_signature=None,
        objects=[], sectors=sectors, authority=authority, mirror_view=False,
        detector_model="yolov8n.pt", detector_confidence=0.35,
        pipeline_config_path=CONFIG / "pipeline.yaml",
        ontology_path=CONFIG / "ontology.yaml",
        clear_threshold_m=2.0, blocked_threshold_m=0.7, sector_choice_tolerance_m=0.10,
        motion_tracker=hdsg.MotionTracker(),
    )
    catalogue = json.loads((CONFIG / "hdsg_request_catalogue.v1.json").read_text(encoding="utf-8"))
    prompt_packet = hdsg.build_prompt_packet(
        fact_packet, prompt_id="prompt_fixture", model_id="qwen3-vl-4b-instruct",
        model_hash=hdsg.sha256_text("model"), quantisation="Q4_K_M",
        temperature=0.2, top_p=0.9, max_tokens=400,
        system_prompt=catalogue["composed_system_prompt"],
        system_prompt_id=catalogue["composed_system_prompt_id"],
        constraint_hash=hdsg.sha256_file(CONFIG / "hdsg.vlm_caption.v1.gbnf"),
        expected_response_schema=hdsg.CAPTION_SCHEMA,
    )
    candidate = {
        "schema_version": hdsg.CAPTION_SCHEMA,
        "caption": CAPTION,
        "assertions": [
            {"fact_id": "sector:centre", "measurement_id": "m:sector:centre:clearance",
             "stated_value": 1.74},
            {"fact_id": "sector:left", "measurement_id": "m:sector:left:clearance",
             "stated_value": 3.13},
        ],
        "visual_observations": [],
    }
    errors, scored = composed.validate_caption_candidate(
        candidate, prompt_packet, fact_packet, detector_classes=["person", "chair"]
    )
    if errors:
        raise SystemExit(f"the fixture caption no longer passes the gate: {errors}")
    release = composed.build_composed_release(
        fact_packet, prompt_packet, release_id="release_fixture", candidate=candidate,
        scored_assertions=scored, failure_codes=[],
    )
    return fact_packet, prompt_packet, candidate, release


def freeze(record):
    """Replaces the wall-clock fields so an unchanged runtime regenerates an unchanged file."""
    frozen = json.loads(json.dumps(record))
    for section, key in (("identity", "captured_at_utc"), ("identity", "released_at_utc"),
                         ("routing", "issued_at_utc")):
        if isinstance(frozen.get(section), dict) and key in frozen[section]:
            frozen[section][key] = "2026-08-21T00:00:00Z"
    return frozen


def main():
    fact_packet, prompt_packet, candidate, release = build_event()
    written = []
    for name, record in (
        ("hdsg.fact_packet.v2.json", fact_packet),
        ("hdsg.prompt_packet.v2.json", prompt_packet),
        ("hdsg.vlm_caption.v1.json", candidate),
        ("hdsg.release.v2.json", release),
    ):
        path = VALID / name
        path.write_bytes((json.dumps(freeze(record), indent=2) + "\n").encode("utf-8"))
        written.append(name)
    print("regenerated: " + ", ".join(written))
    print("hdsg.question_route.v1.json is left alone; it is a classifier reply, not a runtime record.")
    print("Regenerate the manifest digests after this.")


if __name__ == "__main__":
    main()
