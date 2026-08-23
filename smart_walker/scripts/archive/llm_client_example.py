import argparse
import json
import os
from typing import Any, Dict, List, Optional

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

DEFAULT_QWEN_DIR = os.path.join(
    ROOT, "models", "llm", "qwen2.5-7b-instruct"
)
DEFAULT_SYSTEM_PATH = os.path.join(DEFAULT_QWEN_DIR, "prompt_system.txt")
DEFAULT_GRAMMAR_PATH = os.path.join(DEFAULT_QWEN_DIR, "grammar.gbnf")


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def call_llama_openai_chat(
    endpoint: str,
    system_prompt: str,
    user_content: str,
    grammar: Optional[str],
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 128,
    stop: List[str] | None = None,
    model_name: str = "qwen2.5-7b-instruct",
) -> str:
    """Call llama.cpp server (OpenAI-compatible) chat completions with grammar.

    Returns the assistant content (string). Raises for HTTP errors.
    """
    url = endpoint.rstrip("/") + "/v1/chat/completions"
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        # do not stream for simplicity
        "stream": False,
    }
    if stop:
        payload["stop"] = stop

    # Try grammar enforcement in a robust way across llama.cpp versions.
    attempts: List[Dict[str, Any]] = []
    if grammar is not None:
        # Newer shape: object with type/value
        p1 = dict(payload)
        p1["grammar"] = {"type": "gbnf", "value": grammar}
        attempts.append(p1)
        # Older shape: plain string
        p2 = dict(payload)
        p2["grammar"] = grammar
        attempts.append(p2)
    # Fallback without grammar
    attempts.append(dict(payload))

    last_error_text = None
    for i, pl in enumerate(attempts, 1):
        resp = requests.post(url, json=pl, timeout=60)
        if resp.status_code < 400:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        else:
            try:
                last_error_text = resp.text
            except Exception:
                last_error_text = f"HTTP {resp.status_code}"

    raise RuntimeError(
        f"llama.cpp request failed. Last error: {last_error_text}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Test Qwen 2.5 7B via llama.cpp with JSON grammar")
    ap.add_argument("--endpoint", default="http://localhost:8080",
                    help="llama.cpp server endpoint (OpenAI-compatible)")
    ap.add_argument("--system", default=DEFAULT_SYSTEM_PATH,
                    help="Path to system prompt file")
    ap.add_argument("--grammar", default=DEFAULT_GRAMMAR_PATH,
                    help="Path to GBNF grammar file")
    ap.add_argument("--model-name", default="qwen2.5-7b-instruct",
                    help="Model name label")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--stop", nargs="*", default=["\n\n"])  # optional
    ap.add_argument("--no-grammar", action="store_true",
                    help="Disable grammar enforcement for debugging")
    ap.add_argument("--scene", default=None,
                    help="Inline scene summary text (if omitted, a demo will be used)")

    args = ap.parse_args()

    system_prompt = read_text(args.system)
    grammar = None if args.no_grammar else read_text(args.grammar)

    # Minimal demo scene summary (replace with your live JSON summary later)
    demo_scene = (
        "intent: forward; obstacles_ahead: 1 at 1.2m; risk: 0.6; motion: yaw_rate=2.0 deg/s; "
        "hazards: curb_right at 0.8m; suggestion context: narrow passage"
    )
    user_content = args.scene if args.scene else demo_scene

    content = call_llama_openai_chat(
        endpoint=args.endpoint,
        system_prompt=system_prompt,
        user_content=user_content,
        grammar=grammar,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stop=args.stop,
        model_name=args.model_name,
    )

    print("\nRaw model output:\n", content)
    try:
        obj = json.loads(content)
        print("\nParsed JSON:\n", json.dumps(obj, indent=2))
    except json.JSONDecodeError:
        print("\nWarning: Output was not valid JSON. Check server grammar support and prompts.")


if __name__ == "__main__":
    main()
