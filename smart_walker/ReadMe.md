# Activate the venv and enter the application folder
& D:/Swinburne/PostGrad/PhD/Implementation/.venv/Scripts/Activate.ps1
Set-Location D:/Swinburne/PostGrad/PhD/Implementation/smart_walker

# Test the GPU is detected 

## Start the local LLM server (CUDA on RTX)

Use the launcher to start llama.cpp with GPU offload. You should see CUDA/cuBLAS in the logs and how many layers were offloaded to the GPU.

PowerShell (from the `smart_walker` folder):

```powershell
# Use script defaults, or pass -ModelPath to a specific GGUF
& .\scripts\start_llama_server.ps1 -NGL 28 -Ctx 2048 -Threads 8 -Port 8080

# Example with explicit model path
& .\scripts\start_llama_server.ps1 -ModelPath "models\llm\qwen2.5-7b-instruct\Qwen2.5-7B-Instruct-Q4_K_M.gguf" -NGL 28 -Ctx 2048 -Threads 8 -Port 8080
```

Verify RTX usage:


## True VLM (vision-language) live demo — retired, archived

Development history only. `scripts\vlm_realsense_live.py` was the stage 1 LLaVA-1.5 prototype
described in the thesis's Chapter 4 development-progression table; it is superseded by the
canonical Qwen3-VL client below and moved to `scripts\archive\` on 19 August 2026.

For a synchronized, ChatGPT-video-style loop, use a vision-capable model/server (e.g., LLaVA 1.5 7B in llama.cpp vision build or a cloud VLM) and run the RealSense VLM client below.

1) Start a VLM server (choose one):

- Local (llama.cpp vision): download a LLaVA GGUF + mmproj and start the CUDA server via `scripts\start_llama_server.ps1`, pointing `-ModelPath` to the vision GGUF. Note: requires a llama.cpp build that supports images.
- Cloud alternative: use an OpenAI-compatible endpoint (e.g., GPT-4o/mini) that accepts image parts in Chat Completions.

2) Run the RealSense VLM client:

```powershell
# Historical. From repo root; sends 1 FPS frames with image+text messages
python scripts\archive\vlm_realsense_live.py --endpoint http://127.0.0.1:8080 --model llava-1.5-7b --hz 1 --width 640 --height 480
```

Notes:
- On 6 GB VRAM, prefer quantized models (Q4_K_M) and modest context to improve stability/latency.
- If using cloud, set the endpoint accordingly and ensure your API key is configured per provider docs.

Starting a local LLaVA (vision) server example:

```powershell
# Place files under models\llm\llava-1.5-7b\
#  - llava-v1.5-7b-Q4_K_M.gguf (model)
#  - llava-v1.5-mmproj-f16.gguf  (projector)

& .\scripts\start_llama_server.ps1 `
	-ModelPath "models\llm\llava-1.5-7b\llava-v1.5-7b-Q4_K_M.gguf" `
	-MmprojPath "models\llm\llava-1.5-7b\llava-v1.5-mmproj-f16.gguf" `
	-NGL 24 -Ctx 2048 -Threads 8 -Port 8080 -MainGpu 0 -Verbose
```

# Test per frame detection and facts
python per_frame_facts.py

# Test RealSense Bag File (run from root folder)
python scripts/realsense_bag_player.py --bag data/outdoors.bag --realtime true --save_one_pair

# Test RealSense Depth Estimation
python scripts\rgbd_facts_from_pair.py --rgb "outputs\bag_test\color.png" --depth "outputs\bag_test\depth_mm.png" --preset realsense_mm

# Test Real Sense Device
python scripts/realsense_depth_test.py

# Test RGBD mapping (offline)
python scripts\rgbd_facts_with_mapping.py --rgb "outputs\bag_test\color.png" --depth "outputs\bag_test\depth_mm.png" --preset realsense_mm  

# Run RealSense live stream estimation + mapping
python scripts\realsense_per_frame_facts.py --model yolov8n.pt --imgsz 640 --conf 0.25

# Run RealSense live stream risk estimation
# Historical. Retired, moved to scripts\archive\ on 19 August 2026.
python scripts\archive\realsense_per_frame_with_risk.py --model yolov8n.pt --imgsz 640 --conf 0.25

# Run RealSense live stream risk estimation + direction input + simple caption
python scripts/only_realsense.py --model yolov8n.pt --imgsz 640 --conf 0.25 --half --json_hz 10 --json_pretty

# Run RealSense live stream risk estimation + direction input + LLM
# Start the LLM server first (separate terminal) qwen
# Example: & .\scripts\start_llama_server.ps1 -NGL 28 -Ctx 2048 -Threads 8 -Port 8080

python scripts\realsense_shared_control.py --model yolov8n.pt --imgsz 640 --conf 0.25 --half --json_hz 10 --json_pretty --imu --llm --llm_endpoint http://localhost:8080 --llm_hz 2


# llava-1.5-7b
## run the server
& .\scripts\start_llama_server.ps1 -ModelPath "models\llm\llava-1.5-7b\llava-v1.5-7b-Q4_K_M.gguf" -MmprojPath "models\llm\llava-1.5-7b\llava-v1.5-7b-mmproj-model-f16.gguf" -NGL 24 -Ctx 2048 -Threads 8 -Port 8080 -MainGpu 0 -Verbose

# run the RealSense VLM
# Historical. Stage 2 prototype, retired, moved to scripts\archive\ on 19 August 2026.
python scripts\archive\realsense_vlm_on_change.py `
  --endpoint http://127.0.0.1:8080 `
  --model llava-1.5-7b `
  --width 640 --height 480 --fps 15 `
  --image_size 256 --encode jpeg --jpeg_quality 60 `
  --det_model yolov8n.pt --imgsz 448 --conf 0.25 --half `
  --process_hz 2 `
  --min_interval_s 0.6 `
  --max_tokens 160 `
  --warmup `
  --show

  
## Qwen3-VL 4B (current local VLM)

Replaces Qwen2.5-VL 3B: better spatial grounding and OCR at similar latency.
Requires a llama.cpp build with Qwen3-VL mtmd support (b6800+; this repo ships b6937).

`-NGL 99` puts all 37 layers plus the vision encoder on the GPU, which fits the
3060's 6 GB with room for YOLOv8. Confirm the startup log shows
`offloaded 37/37 layers to GPU` and `clip_ctx: CLIP using CUDA0 backend`; if it
says CPU instead, see "CUDA runtime DLLs" below.

1) Start the local llama.cpp server with Qwen3‑VL 4B:

```powershell
& .\scripts\start_llama_server.ps1 `
  -ModelPath "models\llm\qwen3-vl-4b\hf_repo\Qwen3VL-4B-Instruct-Q4_K_M.gguf" `
  -MmprojPath "models\llm\qwen3-vl-4b\mmproj\mmproj-Qwen3VL-4B-Instruct-F16.gguf" `
  -NGL 99 -Ctx 2048 -Threads 8 -Port 8080 -MainGpu 0
```

### CUDA runtime DLLs

`ggml-cuda.dll` needs `cudart64_12.dll`, `cublas64_12.dll` and `cublasLt64_12.dll`
beside it in `tools\llama_cpp\`. Without them the DLL fails to load *silently* and
everything runs on CPU. They ship separately from the llama.cpp binaries, in
`cudart-llama-bin-win-cuda-12.4-x64.zip` on the matching release. Verify with:

```powershell
.\tools\llama_cpp\llama-server.exe --version   # must print "found 1 CUDA devices"
```

3) Run the canonical HDSG client with the Intel RealSense D455f:

```powershell
python scripts\realsense_vlm_on_change_qwen.py `
  --endpoint http://127.0.0.1:8080 `
  --model qwen3-vl-4b-instruct `
  --model_hash sha256:66358cb18bb6b3b1b6675aa412c7a88ef01d228f481184d13668e5201c730a0a `
  --width 640 --height 480 --fps 15 `
  --image_size 448 --encode jpeg --jpeg_quality 70 `
  --det_model yolov8n.pt --imgsz 448 --conf 0.25 --half `
  --process_hz 8 `
  --max_tokens 220 `
  --show `
  --debug_lanes
```

4) To record a run, add these two and nothing else:

```powershell
  --evaluate true `
  --eval_name kitchen_walkthrough
```

`--eval_name` is a short label used to group runs; change it per scenario. Output goes to
`smart_walker\logs\<eval_name>\`. `--record_rgbd` is on by default and writes the colour and depth
frame for every observation, so a recorded run can be replayed later through a different detector or
a different model without walking the route again. That is how the detector comparison in section
10.12 of `HDSG_VERIFIED_GENERATION_POLICY.md` was made, four minutes after the run that prompted it.

**`--det_model` must stay `yolov8n.pt`.** `yolov8n-oiv7.pt` was adopted on 22 August 2026 for its
wider vocabulary, which includes `Stairs`, and reverted the same day. Over 1412 recorded frames it
found none of the furniture COCO found, which was bed 453 times, tv 294, laptop 243 and chair 69, and
it returned nothing at all on 926 frames against 349. A caption cannot describe what the detector
does not report. The default is now correct; the hazard is an old command line that passes the other
weights explicitly and silently overrides it.

The recorded hash belongs to the Qwen3-VL GGUF named in the server command above.
Recalculate and replace it if that model file changes. A run can start without
`--model_hash`, but its telemetry then records an unverified-model digest and is
unsuitable for the formal model-comparison results.

### The interface

`--show` starts the web interface and prints its address, by default
<http://127.0.0.1:8321/>. Open that in a browser. The camera view, sector bands,
object boxes, caption, controls and the question panel are all there.

**This changed on 19 August 2026.** `--show` previously opened an OpenCV window.
That window is still available with `--ui opencv` and is unchanged, but the web
interface is now the default and the one the evaluation records against. Use
`--ui_host` and `--ui_port` to move the server; the host defaults to loopback, so
nothing is exposed off the machine unless that is changed deliberately.

The browser is sent the plain camera frame on one channel and the deterministic
state as JSON on another, and draws the overlays itself. No overlay is composited
into the video, and the only release field the page ever receives is
`caption_text`, so the sole-release-path property is unchanged.

The same `w`/`a`/`s`/`d`/space, `m`, `r` and `q` shortcuts work in the browser. With
`--ui web` the walker no longer watches the keyboard itself: the page sends each
shortcut as an explicit event and suppresses them while the question box has focus.
The operating-system key hook the OpenCV sink uses reports whether a key is
physically down regardless of which window has focus, so leaving it running would
mean typing a question also steered the walker.

### Asking questions

The panel on the right accepts typed questions. A question passes through four
stages:

1. **Measurement pre-check.** If no sector carries a valid clearance, or the most
   recent observation is older than `--more_detail_freshness_s`, the reply is fixed
   and no model is called.
2. **Keyword filter.** A hand-written match recognises a request for a fresh look
   and hands it to the reassessment control without a model call. It is advisory: a
   miss falls through to the classifier, which has the same outcome available.
3. **Admission classifier.** A text-only call constrained by
   `config/hdsg.question_route.v1.gbnf`, which admits exactly three tokens:
   `IN_SCOPE`, `REASSESS`, `OUT_OF_SCOPE`. No image is attached. The classifier
   decides only whether the message is about the space around the walker. It cannot
   produce content.
4. **Answer.** An admitted question receives the whole Fact Packet, the person's own
   wording, and the unchanged entailment gate and release builder. A rejected
   candidate falls back to a deterministic description of the permitted facts.

Until 23 August 2026 the classifier sorted questions into eight navigation topics and
handed the model only the facts belonging to the chosen topic. That was withdrawn.
The topic had to be guessed before generation and a wrong guess could not be
recovered from, because the withheld facts were absent from the call; the gate
already checks the same property afterwards, against the measurements, without
guessing. Answers are now written freely from the full scene and assessed rather than
pre-scoped.

Every answer is prefixed with the authoritative action sentence, and carries the
interaction prompt when one is set, so it reads *"Change direction and continue
towards the right. The centre sector is blocked at 0.60 metres."* That prefix comes
from the deterministic template table by way of the release, never from the model. It
is the one piece of text that tells the person what to do, and it is the reason a
crafted question cannot turn an answer into a movement instruction.

Answers appear in the panel, not on the caption line: the caption keeps showing the
deterministic action and its reason throughout. Questions run on their own worker,
so asking one never delays a guidance update. The panel keeps a visible history of
the session's questions, but no history is ever supplied to the model, which
continues to answer each request statelessly.

`--answer_questions false` disables the channel; a question then receives the
out-of-scope reply and no model is called. `--route_grammar` moves the admission
constraint.

During an evaluation run the `question_route` record stores the question text in the
clear alongside its hash. Whether an admission decision or an answer was correct
cannot be judged without reading what was asked. The release record, which is the
path that reaches the user, carries only the hash.

#### `--unconstrained`, a diagnostic mode

`--unconstrained` sends the frame and the typed text straight to the model with no
routing, no permitted-fact packet, no grammar and no entailment gate. The model gets
one instruction: describe the scene ahead for navigation, and do not say what the
person should do. The reply appears verbatim in the chat panel.

This exists to show what the same model says without the architecture, which is the
`C0_VLM_ONLY` comparison made interactive. Its answers are ungrounded by
construction and carry none of the guarantees the release path provides.

Three things mark it, because a diagnostic answer must never be mistaken for a
released one: the console prints a banner at startup, the page shows a red banner
and outlines each ungated answer in red, and every telemetry envelope carries
`unconstrained_diagnostic_run`. **A run started with this flag is not evaluation
evidence.** The guidance caption is unaffected and stays deterministic throughout,
so the sole-release-path property still holds for everything on the caption line.

The measurement pre-check does not apply in this mode, since it reads the frame
rather than the measurements. The prompt lives in
`config/hdsg_request_catalogue.v1.json` under `unconstrained_diagnostic`, so it can
be edited without touching code.

**Two parts of the policy are not implemented.** Section 6's phrasing variety
(several approved variants per fact, and deterministic joining with a connective
set) is deferred, because its variant set is still an open decision under the
policy's section 13 and it is the change that forces the candidate grammar to move.
Answers are therefore terser than the policy intends. The routing grammar and the
`question_route` telemetry record are also not in `hdsg.schemas.v2`; the policy's
section 7 places them in the next schema set with their fixtures.

The script always uses persistent BoT-SORT tracking. The normal display shows a
bounding box only after the local movement classifier confirms that a tracked
object is moving. Add `--debug_objects` to show every detection during calibration.

The More detail button requests more information about the observation supporting
the current caption. The Reassess button requests a fresh observation. The `M` and
`R` keys provide equivalent test shortcuts. The interface accepts no free text.

Caption generation is intent driven rather than event driven, per
`HDSG_INTENT_TRIGGERED_EXPLANATION_POLICY.md` (implemented 19 August 2026). The
model is called only when a movement intent starts or changes, when the guidance
becomes more restrictive while an intent remains active, or on a More detail or
Reassess request. A less restrictive change updates the action and reason lines at
once from the deterministic templates without calling the model. No caption is
shown while no movement intent is expressed; the sector display and CLEAR SECTORS
badge remain live regardless. While a request is in flight the action line is
shown immediately and the reason line reads "Assessing the environment." until the
model responds or the request times out, except in the temporary interaction
states (`AWAITING_SECTOR_CHOICE`, `REORIENTATION_REQUIRED`,
`POST_REORIENTATION_STABILISING`), which already carry a complete deterministic
account and are left unchanged by a pending request.

Normal development runs use `--evaluate false` and do not write telemetry. In this
mode, `--eval_name` has no effect if it is also present. To record a named evaluation
run, change the option and provide the evaluation name:

```powershell
--evaluate true --eval_name straight_path
```

The complete JSONL record is then written to
`logs/straight_path/run_<timestamp>.jsonl`. The evaluation name is also stored in
every telemetry envelope and as the Fact Packet's scenario identifier. Each distinct
evaluation path should use its own stable name.

The fixed internal prompts are loaded from
`config/hdsg_request_catalogue.v1.json`. Adding or enabling another request changes
the evaluated interface and therefore requires its own validation before use.

An evaluation run writes Full Fact Packets, Restricted Prompt Packets, VLM
candidates, Authoritative Release Objects and one `generation_response` record per
generation to `smart_walker/logs/<eval_name>/<run_id>.jsonl`. Only the
`caption_text` field of an Authoritative Release Object is displayed.

`generation_response` carries the model's raw reply in full, together with its
SHA-256, whether it parsed, the gate reason codes, and four diagnostic durations
(`queue_wait`, `generation`, `gate_and_render`, and `pending`, the window during
which the reason line showed a placeholder while the action line was already
correct). `HDSG_EVALUATION_CONTRACT.md` §12 requires the raw response because a
digest cannot support failure investigation, and a rejected candidate is exactly
the case where the text is the evidence. The record is written to the log and never
read back into the runtime, so raw model text still has no path to the display.

### RGB-D recording

An evaluation run also writes synchronised RGB and depth per observation to
`logs/<eval_name>/<run_id>/`, so the run can later be replayed under other
conditions with identical input. Disable with `--record_rgbd false`.

```
<observation_id>.color.png      8-bit BGR, lossless
<observation_id>.depth_mm.png   16-bit single channel, millimetres
manifest.jsonl                  one line per observation
```

Colour is PNG rather than JPEG because replay feeds the frame back through the
detector, and JPEG artefacts would change detections. Depth is millimetres, the
same `realsense_mm` convention the offline tools already use; the D455f reports
integer millimetres, so the conversion is exact.

Writing happens on a worker thread and never blocks the sensing loop. If a frame
cannot be queued it is counted, and the run reports the recording **incomplete**
at shutdown rather than leaving silent gaps in a replay source.

With recording on, each Fact Packet's `rgb_ref` and `depth_ref` name the recorded
files instead of `memory://`, which is what makes an evaluated event replayable.

### Replaying a run against the comparison conditions

Once a run has been recorded, it can be replayed offline against the two
comparison conditions Chapter 5 defines. Neither can reach the walker display.

```powershell
python scripts\hdsg_replay.py `
  --telemetry logs\straight_path\run_20260820_101500.jsonl `
  --recording logs\straight_path\run_20260820_101500 `
  --out       logs\straight_path\run_20260820_101500.scored.jsonl `
  --endpoint http://127.0.0.1:8080 --model qwen3-vl-4b-instruct
```

| Condition | Sees the image | Sees the measured facts | Output enforced |
|---|---|---|---|
| `C0_VLM_ONLY` | yes | no | no |
| `C1_GROUNDED_UNGATED` | yes | yes | no |
| `C2_FULL_HDSG` | yes | yes | yes |

C0 and C1 are replayed here. C2 is not: its released output is already in the
archive, produced by the live release path under the frozen configuration, and
re-running it would sample a fresh candidate rather than reproduce that release.

The Fact Packet each reply is scored against comes from the archive rather than
being recomputed from the frames. The archive records what the walker actually
measured; recomputing would introduce detector variation between the condition
being scored and the record it is scored against.

Add `--limit 5` for a smoke run. Events whose frame is missing from the recording
are skipped and counted rather than scored against a substitute observation.

The scored output populates Chapter 5's Table 5-9 and Table 5-11. A reply the
scorer cannot resolve is recorded as **unscoreable** and leaves that measure's
denominator, rather than counting as a pass or a failure.

### Schema conformance

The frozen record contracts are in `schemas/`, recorded in
`schema-manifest.v4.json`. Two checks, neither replacing the other:

```powershell
python -m unittest discover -s tests -t .        # 227 tests, jsonschema optional
& .\schemas\validate_schema_fixtures.ps1         # full JSON Schema validation, needs PowerShell 7
```

The Python suite fails if any frozen file's digest drifts from the manifest, which
is how the grammar previously came to differ from its recorded hash unnoticed. It
also builds records through the runtime and validates those, because a fixture
written by hand agrees with its schema by construction and establishes nothing
about what the system emits. `schemas/README.md` records what each set changed.

### The generative contract

The model composes the caption in its own words, writes the measured values into
the prose, and declares each value it stated with the fact it came from. The gate
compares every declaration against the Fact Packet and releases the deterministic
account instead when one disagrees. The action the user acts on is produced by the
deterministic layer in every case and is unaffected by what the model wrote.

**A measurement is stated exactly.** There is no tolerance: a value the model
rounded, approximated or spelled out as a word is a value it altered, and the
caption is refused. An earlier design admitted a 0.10 m window so the prose could
read more naturally, which meant the same sensor reading appeared as 1.74 on a
deterministic fallback and could appear as 1.7 on an accepted caption. The size of
every disagreement is still recorded, so how far the model departed remains
reportable even though any departure rejects.

An earlier design had the model select among controlled sentences the runtime had
written, substituting measurements into placeholders. It was removed on 21 August
2026: assembling the sentence from a fixed predicate table left the model choosing
between one approved synonym and another, so the released wording was the runtime's.
`--generation` is gone with it and there is one generation path.

--------------------------

## Retired: the VLP variant

`scripts/realsense_vlp_on_change_qwen.py` and the `smart_walker_vlp/` package were removed
on 8 August 2026. They were a near-duplicate of the script above. They lacked the
lane-derived risk path and used a prototype stuck-detection interaction. The current
HDSG interface replaces that interaction with fixed More detail and Reassess controls.
The retired files are recoverable from git history at commit `d9b877a`.

`realsense_vlm_on_change_qwen.py` is the single canonical entry point and is the script the
evaluation records against.

## Retired: earlier generative-component prototypes, archived

On 19 August 2026, four scripts documenting the thesis's Chapter 4 development-progression
table (LLaVA-1.5, then Qwen2.5-VL, before the canonical Qwen3-VL client) were moved to
`scripts\archive\` rather than deleted, since Chapter 4 cites them as development history:
`vlm_realsense_live.py`, `realsense_per_frame_with_risk.py`, `realsense_vlm_on_change.py`
and `realsense_vlm_last_version.py`. None is imported by the canonical path. History is
preserved with `git mv`, so `git log --follow` on a file under `scripts\archive\` reaches
its full history at the old path.

In the same pass, `realsense_vlm_on_change_qwen.py`'s own retired direct-caption route
(`_legacy_main` and nine private helper functions, roughly 1,060 lines, unreachable from
`main()`) was deleted rather than archived, since it was dead code inside the canonical
file rather than a separate historical script. It remains in git history on the commit
before this change.
