# Activate the venv 
& D:/Swinburne/PostGrad/PhD/Implementation/.venv/Scripts/Activate.ps1

# Test the GPU is detected 

## Start the local LLM server (CUDA on RTX)

Use the launcher to start llama.cpp with GPU offload. You should see CUDA/cuBLAS in the logs and how many layers were offloaded to the GPU.

PowerShell (from repo root):

```powershell
# Use script defaults, or pass -ModelPath to a specific GGUF
& .\scripts\start_llama_server.ps1 -NGL 28 -Ctx 2048 -Threads 8 -Port 8080

# Example with explicit model path
& .\scripts\start_llama_server.ps1 -ModelPath "models\llm\qwen2.5-7b-instruct\Qwen2.5-7B-Instruct-Q4_K_M.gguf" -NGL 28 -Ctx 2048 -Threads 8 -Port 8080
```

Verify RTX usage:


## True VLM (vision-language) live demo

For a synchronized, ChatGPT-video-style loop, use a vision-capable model/server (e.g., LLaVA 1.5 7B in llama.cpp vision build or a cloud VLM) and run the RealSense VLM client below.

1) Start a VLM server (choose one):

- Local (llama.cpp vision): download a LLaVA GGUF + mmproj and start the CUDA server via `scripts\start_llama_server.ps1`, pointing `-ModelPath` to the vision GGUF. Note: requires a llama.cpp build that supports images.
- Cloud alternative: use an OpenAI-compatible endpoint (e.g., GPT-4o/mini) that accepts image parts in Chat Completions.

2) Run the RealSense VLM client:

```powershell
# From repo root; sends 1 FPS frames with image+text messages
python scripts\vlm_realsense_live.py --endpoint http://localhost:8080 --model llava-1.5-7b --hz 1 --width 640 --height 480
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
python scripts\realsense_per_frame_with_risk.py --model yolov8n.pt --imgsz 640 --conf 0.25

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
python scripts\realsense_vlm_on_change.py `
  --endpoint http://localhost:8080 `
  --model llava-1.5-7b `
  --width 640 --height 480 --fps 15 `
  --image_size 256 --encode jpeg --jpeg_quality 60 `
  --det_model yolov8n.pt --imgsz 448 --conf 0.25 --half `
  --process_hz 2 `
  --min_interval_s 0.6 `
  --max_tokens 160 `
  --warmup `
  --show
## Qwen2.5-VL 3B (faster local VLM)

1) Start the local llama.cpp server with Qwen2.5‑VL 3B:

```powershell
& .\scripts\start_llama_server.ps1 `
  -ModelPath "models\llm\qwen2.5-vl-3b\hf_repo\qwen2.5-vl-3b-instruct-q4_k_m.gguf" `
  -MmprojPath "models\llm\qwen2.5-vl-3b\mmproj\mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf" `
  -NGL 24 -Ctx 2048 -Threads 8 -Port 8080 -MainGpu 0 -Verbose
```

3) Run the RealSense VLM client specialized for Qwen2‑VL:

```powershell
python scripts\realsense_vlm_on_change_qwen.py `
  --endpoint http://localhost:8080 `
  --model qwen2.5-vl-3b-instruct `
  --width 640 --height 480 --fps 15 `
  --image_size 224 --encode jpeg --jpeg_quality 70 `
  --det_model yolov8n.pt --imgsz 448 --conf 0.25 --half `
  --process_hz 2 `
  --min_interval_s 1.5 `
  --max_tokens 120 `
  --show `
  --debug_lanes
```