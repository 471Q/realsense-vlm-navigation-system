#!/usr/bin/env python3
"""
Benchmark Qwen2.5-VL 3B and LLaVA-1.5 7B on a single image to collect
latency, throughput, and memory metrics on the local machine.

Requirements (install if missing):
  pip install --upgrade transformers accelerate safetensors sentencepiece timm einops
Optional (for 4-bit GPU quantization):
  pip install bitsandbytes

Examples (PowerShell):
  python scripts/benchmark_vlm.py --all --image path/to/frame.jpg --device auto --precision bf16 --runs 5 --warmup 1
  python scripts/benchmark_vlm.py --model qwen --image tests/example.jpg --device cuda --precision fp16 --max-new-tokens 48

Outputs:
  - JSON and CSV written under outputs/benchmarks/

NOTE: This script does not download models ahead of time; the first run will
      trigger downloads from the HF Hub.
"""
from __future__ import annotations
import argparse
import os
import time
import json
import csv
from datetime import datetime
from typing import Any, Dict, Optional

import psutil
import torch
from PIL import Image, ImageOps

# Lazy imports for transformers to avoid import cost when not installed
try:
    import transformers
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
    )
    from transformers import LlavaForConditionalGeneration  # type: ignore
except Exception as e:  # pragma: no cover
    transformers = None  # type: ignore


QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
# Model cache directory (can be overridden with --cache-dir). If not provided, we
# use the default HF cache. Set to a folder to avoid re-downloading in future runs.
CACHE_DIR: Optional[str] = None


def human_device(arg: str) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def load_image(path: Optional[str], target: int = 256) -> Image.Image:
    if path and os.path.exists(path):
        img = Image.open(path).convert("RGB")
    else:
        # Generate a synthetic pattern if no image is provided
        img = Image.new("RGB", (640, 480), color=(200, 210, 220))
    # Keep aspect ratio; pad to square
    img = ImageOps.contain(img, (target, target))
    img = ImageOps.pad(img, (target, target), color=(128, 128, 128))
    return img


def _maybe_4bit(load_4bit: bool):
    if not load_4bit:
        return None
    try:
        from transformers import BitsAndBytesConfig  # type: ignore

        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    except Exception:
        print("[warn] bitsandbytes not available; continuing without 4-bit.")
        return None


@torch.inference_mode()
def bench_qwen(image: Image.Image, device: str, precision: str, max_new_tokens: int, runs: int, warmup: int, load_4bit: bool) -> Dict[str, Any]:
    """Benchmark Qwen2.5-VL 3B (multimodal). Uses the dedicated conditional generation class.

    If the local transformers version does not expose the required class, instruct user to upgrade:
      pip install --upgrade transformers
    """
    assert transformers is not None, "transformers is required"

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(precision, torch.float16)

    quant_cfg = _maybe_4bit(load_4bit) if device == "cuda" else None

    # Import specialized class (remote code) rather than generic AutoModelForCausalLM
    qwen_mm_cls = None
    for candidate in ["Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration"]:
        qwen_mm_cls = getattr(transformers, candidate, None)
        if qwen_mm_cls is not None:
            break
    if qwen_mm_cls is None:
        raise RuntimeError(
            "Qwen2.5-VL conditional generation class not found. Upgrade transformers or enable trust_remote_code."
        )

    processor = AutoProcessor.from_pretrained(
        QWEN_ID, trust_remote_code=True, cache_dir=CACHE_DIR)
    if quant_cfg is not None:
        model = qwen_mm_cls.from_pretrained(  # type: ignore
            QWEN_ID,
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=quant_cfg,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
    else:
        model = qwen_mm_cls.from_pretrained(  # type: ignore
            QWEN_ID,
            torch_dtype=dtype,
            trust_remote_code=True,
            cache_dir=CACHE_DIR,
        )
        model.to(device)

    messages = [
        {"role": "system", "content": "You are a precise captioner."},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe the scene in one sentence."},
            ],
        },
    ]
    chat_text = processor.apply_chat_template(
        messages, add_generation_prompt=True)

    def run_once() -> Dict[str, Any]:
        inputs = processor(text=[chat_text], images=[
                           image], return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        seq = getattr(out, "sequences", out)
        gen_len = seq.shape[-1] - input_len
        text_out = processor.batch_decode(seq, skip_special_tokens=True)[0]
        gpu_peak = int(torch.cuda.max_memory_allocated()
                       ) if device == "cuda" else 0
        cpu_mem = int(psutil.Process(os.getpid()).memory_info().rss)
        return {
            "latency_s": t1 - t0,
            "gen_tokens": int(gen_len),
            "throughput_tok_s": float(gen_len) / max(t1 - t0, 1e-6),
            "gpu_bytes_peak": gpu_peak,
            "cpu_bytes_rss": cpu_mem,
            "sample_text": text_out[:180],
        }

    for _ in range(max(warmup, 0)):
        _ = run_once()
    results = [run_once() for _ in range(max(runs, 1))]
    agg = _aggregate(results)
    agg.update({
        "model": "qwen2.5-vl-3b",
        "device": device,
        "precision": precision,
        "max_new_tokens": max_new_tokens,
        "load_4bit": bool(quant_cfg is not None),
    })
    return agg


@torch.inference_mode()
def bench_llava(image: Image.Image, device: str, precision: str, max_new_tokens: int, runs: int, warmup: int, load_4bit: bool) -> Dict[str, Any]:
    """Benchmark LLaVA-1.5 7B.

    Fixes prior image token mismatch by using chat template with an <image> placeholder.
    Also enables fast image processor when available and consistent decoding.
    """
    assert transformers is not None, "transformers is required"

    dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }.get(precision, torch.float16)

    quant_cfg = _maybe_4bit(load_4bit) if device == "cuda" else None

    # use_fast may switch to a faster processor variant (warning if not saved fast).
    processor = AutoProcessor.from_pretrained(
        LLAVA_ID, cache_dir=CACHE_DIR, use_fast=True)
    try:
        if quant_cfg is not None and device == "cuda":
            model = LlavaForConditionalGeneration.from_pretrained(  # type: ignore
                LLAVA_ID,
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=quant_cfg,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True,
            )
        else:
            model = LlavaForConditionalGeneration.from_pretrained(  # type: ignore
                LLAVA_ID,
                torch_dtype=dtype,
                cache_dir=CACHE_DIR,
                low_cpu_mem_usage=True,
            )
            model.to(device)
    except (ValueError, RuntimeError) as e:
        print("[warn] LLaVA GPU/4-bit load failed; falling back to CPU fp32. Reason:",
              str(e)[:240], "...")
        model = LlavaForConditionalGeneration.from_pretrained(  # type: ignore
            LLAVA_ID,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            cache_dir=CACHE_DIR,
            low_cpu_mem_usage=True,
        )

    # Chat-style template required so the processor injects image placeholder tokens
    messages = [
        {"role": "system", "content": "You are a precise captioner."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe the scene in one sentence."}
        ]}
    ]
    chat_text = processor.apply_chat_template(
        messages, add_generation_prompt=True)

    def run_once() -> Dict[str, Any]:
        # images expects a list; text already contains placeholder tokens
        inputs = processor(text=[chat_text], images=[
                           image], return_tensors="pt").to(device)
        input_len = inputs["input_ids"].shape[-1]
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        seq = getattr(out, "sequences", out)
        gen_len = seq.shape[-1] - input_len
        text_out = processor.batch_decode(seq, skip_special_tokens=True)[0]
        gpu_peak = int(torch.cuda.max_memory_allocated()
                       ) if device == "cuda" else 0
        cpu_mem = int(psutil.Process(os.getpid()).memory_info().rss)
        return {
            "latency_s": t1 - t0,
            "gen_tokens": int(gen_len),
            "throughput_tok_s": float(gen_len) / max(t1 - t0, 1e-6),
            "gpu_bytes_peak": gpu_peak,
            "cpu_bytes_rss": cpu_mem,
            "sample_text": text_out[:160],
        }

    for _ in range(max(warmup, 0)):
        _ = run_once()
    results = [run_once() for _ in range(max(runs, 1))]
    agg = _aggregate(results)
    agg.update({
        "model": "llava-1.5-7b",
        "device": device,
        "precision": precision,
        "max_new_tokens": max_new_tokens,
        "load_4bit": bool(quant_cfg is not None),
    })
    return agg


def _aggregate(rows: list[Dict[str, Any]]) -> Dict[str, Any]:
    import statistics as stats

    lat = [r["latency_s"] for r in rows]
    thr = [r["throughput_tok_s"] for r in rows]
    tok = [r["gen_tokens"] for r in rows]
    gpu = [r["gpu_bytes_peak"] for r in rows]
    cpu = [r["cpu_bytes_rss"] for r in rows]

    def s(x):
        return {
            "min": float(min(x)),
            "p50": float(stats.median(x)),
            "mean": float(stats.mean(x)),
            "max": float(max(x)),
        }

    agg = {
        "n": len(rows),
        "latency_s": s(lat),
        "throughput_tok_s": s(thr),
        "gen_tokens": s(tok),
        "gpu_bytes_peak": s(gpu),
        "cpu_bytes_rss": s(cpu),
        "sample_text": rows[-1].get("sample_text", ""),
    }
    return agg


def save_results(res: list[Dict[str, Any]], outdir: str) -> Dict[str, str]:
    os.makedirs(outdir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(outdir, f"vlm_bench_{stamp}.json")
    csv_path = os.path.join(outdir, f"vlm_bench_{stamp}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    # Flatten CSV
    keys = [
        "model",
        "device",
        "precision",
        "max_new_tokens",
        "load_4bit",
        "n",
        "latency_s.p50",
        "throughput_tok_s.p50",
        "gen_tokens.p50",
        "gpu_bytes_peak.p50",
        "cpu_bytes_rss.p50",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in res:
            row = [
                r.get("model"),
                r.get("device"),
                r.get("precision"),
                r.get("max_new_tokens"),
                r.get("load_4bit"),
                r.get("n"),
                r["latency_s"]["p50"],
                r["throughput_tok_s"]["p50"],
                r["gen_tokens"]["p50"],
                r["gpu_bytes_peak"]["p50"],
                r["cpu_bytes_rss"]["p50"],
            ]
            w.writerow(row)
    return {"json": json_path, "csv": csv_path}


def main():
    parser = argparse.ArgumentParser(description="VLM micro-benchmark")
    m = parser.add_mutually_exclusive_group(required=False)
    m.add_argument(
        "--model", choices=["qwen", "llava"], help="Single model to run")
    m.add_argument("--all", action="store_true", help="Run both models")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to an input image")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda"], help="Device selection")
    parser.add_argument("--precision", type=str, default="fp16",
                        choices=["fp16", "bf16", "fp32"], help="Computation dtype")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--target-size", type=int, default=256,
                        help="Image side size (square pad)")
    parser.add_argument("--load-4bit", action="store_true",
                        help="Try 4-bit GPU quantization (requires bitsandbytes)")
    parser.add_argument("--outdir", type=str,
                        default=os.path.join("outputs", "benchmarks"))
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory for HF/Transformers cache. If omitted, the default user cache is used. Point this to your existing cache to avoid re-downloads.")

    args = parser.parse_args()

    if transformers is None:
        raise SystemExit(
            "transformers not installed. Run: pip install transformers accelerate safetensors sentencepiece timm einops")

    # Set global cache dir so from_pretrained uses it (if provided)
    global CACHE_DIR
    CACHE_DIR = args.cache_dir
    if CACHE_DIR:
        os.makedirs(CACHE_DIR, exist_ok=True)
        print(f"[bench] Using model cache dir: {CACHE_DIR}")
    else:
        print("[bench] Using default Hugging Face cache (set --cache-dir to override).")

    device = human_device(args.device)
    image = load_image(args.image, args.target_size)

    results = []
    if args.all or args.model == "qwen":
        results.append(
            bench_qwen(
                image=image,
                device=device,
                precision=args.precision,
                max_new_tokens=args.max_new_tokens,
                runs=args.runs,
                warmup=args.warmup,
                load_4bit=args.load_4bit,
            )
        )
    if args.all or args.model == "llava":
        results.append(
            bench_llava(
                image=image,
                device=device,
                precision=args.precision,
                max_new_tokens=args.max_new_tokens,
                runs=args.runs,
                warmup=args.warmup,
                load_4bit=args.load_4bit,
            )
        )

    paths = save_results(results, args.outdir)
    print("\nBenchmark complete.")
    print("JSON:", paths["json"])
    print("CSV:", paths["csv"])
    for r in results:
        print("-", r["model"], "p50 latency(s)=", round(r["latency_s"]
              ["p50"], 3), ", tok/s=", round(r["throughput_tok_s"]["p50"], 2))


if __name__ == "__main__":
    main()
