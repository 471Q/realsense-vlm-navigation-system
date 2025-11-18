from __future__ import annotations
from typing import Dict, Any


def _nearest_obstacle_m(objs):
    ds = [o.get("distance_m")
          for o in objs if isinstance(o.get("distance_m"), (int, float))]
    return min(ds) if ds else None


def _multi_near_count(objs):
    return sum(1 for o in objs if o.get("distance_bin") in ("very_close", "near"))


def compute_baseline_risk(facts: Dict[str, Any], cfg: Dict[str, Any]) -> dict:
    """Return {'risk': 'safe|caution|stop', 'rules_fired': [..]} based on config thresholds."""
    rf = []
    rules = cfg["risk_rules_baseline"]
    objs = facts.get("objects", [])
    nearest_m = _nearest_obstacle_m(objs)
    facts["explain"]["min_distance_m"] = nearest_m

    # STOP rules
    stop = rules["stop"]
    if nearest_m is not None and nearest_m < stop["nearest_obstacle_m_lt"]:
        rf.append("stop:nearest_obstacle")
    fs = facts.get("free_space", {})
    cmw = fs.get("corridor_min_width_m")
    if cmw is not None and cmw < stop["corridor_min_width_m_lt"]:
        rf.append("stop:corridor_narrow")
    hazards = facts.get("hazards", [])
    if nearest_m is not None and hazards and nearest_m <= stop["hazard_within_m_lte"]:
        rf.append("stop:hazard_nearby")
    if rf:
        return {"risk": "stop", "rules_fired": rf}

    # CAUTION rules
    caut = rules["caution"]
    if nearest_m is not None and nearest_m < caut["nearest_obstacle_m_lt"]:
        rf.append("caution:nearest_obstacle")
    if _multi_near_count(objs) >= caut["multi_near_objects_count_gte"]:
        rf.append("caution:multi_near")
    if caut.get("high_uncertainty") and facts.get("uncertainty", {}).get("low_light"):
        rf.append("caution:uncertainty")
    if rf:
        return {"risk": "caution", "rules_fired": rf}

    return {"risk": "safe", "rules_fired": []}
