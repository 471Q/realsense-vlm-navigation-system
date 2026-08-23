"""Regenerates the valid fixtures from the runtime.

A fixture written by hand agrees with its schema by construction and establishes nothing about the
records the system emits. Seven conformance breaks survived the v2 freeze for that reason, and the
prompt packet fixture then carried `approved_text_templates` and named the templated contract for a
day after the runtime stopped producing either. Generating the fixtures removes the gap between what
the frozen set is checked against and what the system writes.

The invalid fixtures are generated too, since 24 August 2026, each by injecting one named fault into
a valid record. Written by hand they were skeletons: the block under test carried the fault and every
other block was an empty object, so four of the six were refused for dozens of missing required
properties and the named fault was one line in the noise. A schema loosened to admit that fault would
have gone on rejecting them, and the freeze test would have gone on passing. Deriving each from a
record that validates means the only reason it fails is the fault it is named for, which
`tests/test_schema_fixtures.py` now asserts individually.

    python schemas/generate_fixtures.py

Identifiers and timestamps are fixed so that regenerating an unchanged runtime produces an unchanged
file and the manifest digests do not churn.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_questions as questions  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

VALID = Path(__file__).resolve().parent / "fixtures" / "valid"
INVALID = Path(__file__).resolve().parent / "fixtures" / "invalid"
CONFIG = ROOT / "config"

# One scene, carried through every record, so the fixtures form a consistent event rather than four
# unrelated documents. The centre sector is constrained and the right sector is blocked, which is
# the case that exercises sector selection rather than a plain proceed.
DEPTHS = [3.13, 1.74, 0.55]
STATUSES = ["CLEAR", "CONSTRAINED", "BLOCKED"]

CAPTION = "The way ahead narrows to 1.74 metres, with 3.13 metres of space to the left."

# The commit a fixture was generated from is not a property of the record's shape, and letting the
# real one through makes every fixture carrying `code_version` change on every commit. Pinned for the
# same reason the identifiers and timestamps are pinned. A real record carries the commit that
# produced it; see `hdsg_runtime.code_version`.
FIXTURE_CODE_VERSION = {"commit": "0" * 40, "branch": "fixture", "dirty": False}


def pin_code_version(record):
    """Replaces any `code_version` in a generated record with the fixed one, wherever it sits."""
    if isinstance(record, dict):
        for key, value in record.items():
            if key == "code_version" and isinstance(value, dict):
                record[key] = dict(FIXTURE_CODE_VERSION)
            else:
                pin_code_version(value)
    elif isinstance(record, list):
        for item in record:
            pin_code_version(item)
    return record


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
        # Read from the runtime's constants rather than written out. This said 2.0 until 23 August
        # 2026 while the shipped clear distance was 1.8, so the frozen example of a well formed
        # record stated a threshold no run used, which is the defect the thresholds work removed
        # everywhere else.
        clear_threshold_m=hdsg.SECTOR_CLEAR_AT_OR_ABOVE_M,
        blocked_threshold_m=hdsg.OBJECT_STOP_BELOW_M,
        sector_choice_tolerance_m=0.10,
        motion_tracker=hdsg.MotionTracker(),
    )
    fact_packet["configuration"]["code_version"] = dict(FIXTURE_CODE_VERSION)
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


FROZEN_TIMESTAMP = "2026-08-21T00:00:00Z"


def freeze(value):
    """Replaces every wall-clock field so an unchanged runtime regenerates an unchanged file.

    The search is over any key ending in `_at_utc`, at any depth, rather than over a list of the
    places timestamps are known to appear. A named list missed `observation.captured_at_utc` and the
    fact packet fixture was consequently different on every run.
    """
    if isinstance(value, dict):
        return {key: FROZEN_TIMESTAMP if key.endswith("_at_utc") else freeze(item)
                for key, item in value.items()}
    if isinstance(value, list):
        return [freeze(item) for item in value]
    return value



def with_object(fact_packet):
    """The fixture scene holds no detections, and one fault is about an object, so it gets one."""
    packet = copy.deepcopy(fact_packet)
    packet["objects"] = hdsg.normalise_objects([{
        "id": 0, "track_id": 3, "raw_label": "chair", "canonical_class": "chair",
        "ontology_class": "chair", "conf": 0.8, "bbox_xyxy": [10, 10, 90, 200],
        "bearing": "LEFT", "distance_m": 1.62,
        "distance_method": "D455F_LOWER_BBOX_MEDIAN",
    }])
    return packet


def build_faults(fact_packet, prompt_packet, candidate, release):
    """One valid record per entry, with exactly one thing wrong with it.

    Each fault is a rule the schema states across fields, which is the kind a loosened schema stops
    enforcing without any single field looking wrong. The route reply is the exception: it is a
    two-field object, so a bad value in one field is the whole of it.
    """
    stationary = with_object(fact_packet)
    # A stationary object is not drawn, so claiming both at once contradicts the display rule.
    stationary["objects"][0]["motion_state"] = "STATIONARY"
    stationary["objects"][0]["display_bounding_box"] = True

    # The automatic profile permits no visual observations, so allowing them contradicts the mode.
    automatic_visuals = copy.deepcopy(prompt_packet)
    automatic_visuals["routing"]["request_id"] = "AUTO_GUIDANCE"
    automatic_visuals["routing"]["response_mode"] = "AUTOMATIC"
    automatic_visuals["routing"]["prompt_profile_id"] = "guidance_reason.v1"

    # A released caption cannot be both accepted and carry the reason it was refused for.
    accepted_fallback_code = copy.deepcopy(release)
    accepted_fallback_code["verification"]["primary_reason_code"] = "RG_ACTION_LANGUAGE_DETECTED"
    accepted_fallback_code["verification"]["reason_codes"] = ["RG_ACTION_LANGUAGE_DETECTED"]

    # The caption carries only what the model may write. A motion decision is the runtime's.
    extra_action = copy.deepcopy(candidate)
    extra_action["motion_decision"] = "PROCEED"

    unknown_stage = questions.build_route_record(
        "what is on my left?", "IN_SCOPE", "ADMISSION_CLASSIFIER", True)
    unknown_stage["resolved_by"] = "TIER_0_KEYWORD"

    unknown_route = {"schema_version": questions.ROUTE_SCHEMA, "route": "OBJECT_QUERY"}

    return (
        ("hdsg.fact_packet.v2.stationary_box.json", stationary),
        ("hdsg.prompt_packet.v2.automatic_visuals.json", automatic_visuals),
        ("hdsg.question_record.v1.unknown_stage.json", unknown_stage),
        ("hdsg.question_route.v1.unknown_route.json", unknown_route),
        ("hdsg.release.v2.accepted_fallback_code.json", accepted_fallback_code),
        ("hdsg.vlm_caption.v1.extra_action.json", extra_action),
    )


def main():
    fact_packet, prompt_packet, candidate, release = build_event()
    written = []
    for name, record in (
        ("hdsg.fact_packet.v2.json", fact_packet),
        ("hdsg.prompt_packet.v2.json", prompt_packet),
        ("hdsg.vlm_caption.v1.json", candidate),
        ("hdsg.release.v2.json", release),
        # Generated for the same reason as the rest. It was hand-written, and so still carried
        # question_chars for a moment after the runtime stopped writing it on 24 August 2026.
        ("hdsg.question_record.v1.json",
         questions.build_route_record("what is on my left?", "IN_SCOPE",
                                      "ADMISSION_CLASSIFIER", True)),
    ):
        path = VALID / name
        path.write_bytes(
            (json.dumps(pin_code_version(freeze(record)), indent=2) + "\n").encode("utf-8"))
        written.append(name)
    faults = []
    for name, record in build_faults(fact_packet, prompt_packet, candidate, release):
        (INVALID / name).write_bytes(
            (json.dumps(pin_code_version(freeze(record)), indent=2) + chr(10)).encode("utf-8"))
        faults.append(name)
    print("regenerated: " + ", ".join(written))
    print("regenerated with one fault each: " + ", ".join(faults))
    print("hdsg.question_route.v1.json is left alone; it is a classifier reply, not a runtime record.")
    print("Then: python scripts/generate_manifest.py")


if __name__ == "__main__":
    main()
