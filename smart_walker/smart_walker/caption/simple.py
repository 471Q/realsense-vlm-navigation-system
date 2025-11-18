from __future__ import annotations
from typing import Dict, Any
import math

def _nearest(objs):
    ds = [(o["id"], o.get("raw_label"), o.get("distance_m"), o.get("bearing")) for o in objs if isinstance(o.get("distance_m"), (int, float))]
    if not ds: return None
    return min(ds, key=lambda x: x[2])

def make_caption(facts: Dict[str, Any]) -> str:
    risk = facts.get("risk", "safe")
    objs = facts.get("objects", [])
    haz_ids = set(facts.get("hazards", []))

    if not objs:
        return f"{risk.capitalize()}. No obstacles detected."

    parts = [risk.capitalize() + "."]

    # nearest obstacle
    n = _nearest(objs)
    if n:
        _, lbl, d, bearing = n
        if d is not None:
            parts.append(f"Nearest {lbl} {d:.1f} m {bearing}.")

    # hazards summary
    hz = [o for o in objs if o["id"] in haz_ids]
    if hz:
        # e.g., “Hazard: wet floor sign 1.2 m centre.”
        h = min(hz, key=lambda o: (o["distance_m"] if isinstance(o.get("distance_m"), (int,float)) else math.inf))
        dtxt = f"{h['distance_m']:.1f} m " if isinstance(h.get("distance_m"), (int,float)) else ""
        parts.append(f"Hazard: {h['raw_label']} {dtxt}{h['bearing']}.")

    # crowding cue
    near_count = sum(1 for o in objs if o.get("distance_bin") in ("very_close","near"))
    if near_count >= 2:
        parts.append("Multiple nearby obstacles.")

    return " ".join(parts)
