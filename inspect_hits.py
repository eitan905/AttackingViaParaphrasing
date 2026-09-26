#!/usr/bin/env python3
"""
Re-run probe hits showing the FULL raw model output, so we can tell apart:
  - genuine model failures (the model really got it wrong)
  - extraction failures (the model was right; our parser mangled the output)

Usage:
    python inspect_hits.py --model qwen3:4b --url http://127.0.0.1:11435
"""
import argparse
import json

import requests

from victim import _SYSTEM_PROMPT, _extract_short_answer

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="qwen3:4b")
parser.add_argument("--url",   default="http://127.0.0.1:11435")
parser.add_argument("--hits",  default="results/probe_results.json")
parser.add_argument("--max-tokens", type=int, default=600)
args = parser.parse_args()

with open(args.hits, encoding="utf-8") as f:
    hits = json.load(f)["hits"]

print(f"Inspecting {len(hits)} hits with raw output\n")

for i, h in enumerate(hits, 1):
    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": h["paraphrase"]},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": args.max_tokens},
    }
    data = requests.post(f"{args.url}/api/chat", json=payload, timeout=300).json()
    raw = data.get("message", {}).get("content", "").strip()

    print("=" * 78)
    print(f"[{i}] set={h['set']}")
    print(f"    original  : {h['original']}")
    print(f"    paraphrase: {h['paraphrase']}")
    print(f"    extracted : {h['answer']!r}")
    print(f"    --- RAW ({len(raw)} chars) ---")
    print(raw)
    print()
