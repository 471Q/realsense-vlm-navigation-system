"""Builders shared by the test modules.

One place to construct a scene, so a test says what is measured and nothing else. The previous
suite repeated a forty-line packet builder in four files, and they drifted: two of them still passed
arguments the runtime had stopped accepting, which is invisible while the tests pass.

Nothing here asserts. A helper that asserts hides which test failed and why.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import hdsg_composed as composed  # noqa: E402
from scripts import hdsg_runtime as hdsg  # noqa: E402

CONFIG = ROOT / "config"
SCHEMAS = ROOT / "schemas"

# The detector's whole vocabulary is what the object check screens against, so a scene that reports
# no chair must still know "chair" is a class the detector could have reported. These are Open
# Images V7 names, as the detector now emits, rather than the COCO names used before 22 August 2026.
# COCO spellings, as the shipped detector emits. A few Open Images words are kept alongside them
# because the gate is case-insensitive and vocabulary-agnostic, and the tests that matter should
# keep working whichever weights are in use.
DETECTOR_CLASSES = ("chair", "person", "tv", "bed", "couch", "bicycle", "bottle", "bus",
                    "suitcase", "backpack", "potted plant", "dining table",
                    "Door", "Stairs", "Wheelchair", "Furniture")


def sectors(left=3.13, centre=1.74, right=1.16,
            statuses=("CLEAR", "CONSTRAINED", "CONSTRAINED")):
    """Builds the three-sector lane state. A depth of None is a sector with no reliable reading."""
    return hdsg.sectors_from_lane_state(
        {"depths": [left, centre, right], "status": list(statuses)}
    )


def detected_object(track_id=3, label="chair", bearing="LEFT", distance_m=1.62,
                    ontology_class="furniture"):
    """One detection, normalised as the runtime normalises it.

    `ontology_class` is passed rather than hidden behind a flag because "hazard" is what makes an
    object stop the walker at the longer hazard distance. `normalise_objects` derives `is_hazard`
    from it, so setting that key directly does nothing.

    It no longer governs motion. Requiring an ontology class of agent or rolling obstacle before an
    object could be called moving recorded everything else as STATIONARY with a confidence of 1.0,
    which asserted a certainty nothing had measured.
    """
    return hdsg.normalise_objects([{
        "id": 0, "track_id": track_id, "raw_label": label, "canonical_class": label,
        "ontology_class": ontology_class, "conf": 0.8, "bbox_xyxy": [10, 10, 90, 200],
        "bearing": bearing, "distance_m": distance_m,
        "distance_method": "D455F_LOWER_BBOX_MEDIAN",
    }])


def fact_packet(intent="FORWARD", object_advisory="CAUTION", lane=None, objects=(),
                response_mode="MORE_DETAIL", previous_selected_sector=None,
                trigger_type="USER_REQUESTED", input_method=None, control_id=None):
    lane = sectors() if lane is None else lane
    authority = hdsg.determine_authority(
        intent, object_advisory, lane, objects=list(objects),
        previous_selected_sector=previous_selected_sector,
        sector_choice_tolerance_m=0.10,
    )
    request_id = "AUTO_GUIDANCE" if response_mode == "AUTOMATIC" else "MORE_DETAIL"
    extra = {}
    if input_method:
        extra["input_method"] = input_method
    if control_id:
        extra["control_id"] = control_id
    return hdsg.build_fact_packet(
        run_id="run_t", event_id="evt_1", observation_id="obs_1", ticket_id=None,
        timestamp_ms=1.0, intent=intent, trigger_type=trigger_type,
        request_id=request_id, response_mode=response_mode, previous_signature=None,
        objects=list(objects), sectors=lane, authority=authority, mirror_view=False,
        detector_model="yolov8n.pt", detector_confidence=0.35,
        pipeline_config_path=CONFIG / "pipeline.yaml",
        ontology_path=CONFIG / "ontology.yaml",
        # Read from the runtime's constants rather than written out, so a test packet's record of
        # how it was configured agrees with the shipped configuration. It said 2.0 while the live
        # default was 1.8 until 23 August 2026.
        clear_threshold_m=hdsg.SECTOR_CLEAR_AT_OR_ABOVE_M,
        blocked_threshold_m=hdsg.OBJECT_STOP_BELOW_M, sector_choice_tolerance_m=0.10,
        motion_tracker=hdsg.MotionTracker(), **extra,
    )


def prompt_packet(packet, prompt_profile_id=None):
    extra = {}
    if prompt_profile_id is not None:
        extra["prompt_profile_id"] = prompt_profile_id
    return hdsg.build_prompt_packet(
        packet, prompt_id="prompt_1", model_id="qwen3-vl-4b-instruct",
        model_hash=hdsg.sha256_text("model"), quantisation="Q4_K_M",
        temperature=0.2, top_p=0.9, max_tokens=400,
        system_prompt="system", constraint_hash=hdsg.sha256_text("grammar"),
        expected_response_schema=hdsg.CAPTION_SCHEMA, **extra,
    )


def event(**kwargs):
    """The pair most tests need: a fact packet and the prompt packet built from it."""
    packet = fact_packet(**kwargs)
    return packet, prompt_packet(packet)


def caption(text, assertions=(), visuals=()):
    return {
        "schema_version": composed.CAPTION_SCHEMA,
        "caption": text,
        "assertions": [dict(item) for item in assertions],
        "visual_observations": [dict(item) for item in visuals],
    }


def declares(name, value):
    """A declaration of a sector clearance."""
    return {"fact_id": f"sector:{name}",
            "measurement_id": f"m:sector:{name}:clearance",
            "stated_value": value}


def declares_object(track_id, value):
    return {"fact_id": f"object:{track_id}",
            "measurement_id": f"m:object:{track_id}:distance",
            "stated_value": value}


def observation(identifier="visual:1", label="doorway", bearing="LEFT"):
    return {"candidate_observation_id": identifier, "proposed_label": label, "bearing": bearing}


def gate(candidate, packet, prompt, classes=DETECTOR_CLASSES):
    return composed.validate_caption_candidate(
        candidate, prompt, packet, detector_classes=classes
    )
