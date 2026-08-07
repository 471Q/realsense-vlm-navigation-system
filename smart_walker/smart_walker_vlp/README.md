# smart_walker_vlp

Minimal vision-language planning (VLP) layer for the smart walker project.

This module does **not** add any new neural models. Instead, it uses:

- Depth-based clearances (front/left/right/back) from your existing RealSense pipeline.
- Optional semantic flags (e.g. `person_in_front`, `crowded_front`) from Qwen / ontology.
- The user's immediate intent (e.g. `W/A/S/D` mapped to `forward/back/left/right`).

Given this, it produces a **Suggestion**:

- A primary action: `forward`, `backward`, `left`, `right`, `rotate_left`, `rotate_right`, or `stop`.
- A safety level: `safe`, `caution`, or `danger`.
- A natural-language message that you can speak or display to the user.

The core decision logic lives in `planner.py` and is intentionally simple
and transparent so you can describe and evaluate it in your thesis.

## Quick offline demo

From the project root (where `smart_walker_vlp` is a folder), run:

```powershell
python -m smart_walker_vlp.demo
```

You will see a few hard-coded scenarios (clear path ahead, blocked ahead,
only back is free, bad visibility) and the corresponding textual
suggestions.

## How to integrate

In your real-time RealSense + VLM scripts, you would:

1. Compute depth-based clearances (front/left/right/back) as you already do.
2. Optionally set semantic flags like `person_in_front` or `crowded_front`.
3. Map keyboard events or higher-level commands to a `UserIntent`.
4. Call `decide_suggestion(state, intent)` and use its output to:
   - Decide whether to allow the move or suggest an alternative.
   - Provide feedback via TTS or on-screen text.

This keeps your existing working code intact while adding a
well-defined VLP decision layer that you can extend later.
