#!/usr/bin/env python3
"""
Ask the victim model a single question and print its full raw output
plus the extracted short answer.

Usage:
    python ask.py "Which ancient civilization built Machu Picchu?"
    python ask.py "Who painted the Mona Lisa?" --model gemma3:12b
    python ask.py "What is the capital of France?" --url http://127.0.0.1:11435
    python ask.py "Who built Machu Picchu?" --model qwen3:4b --think
"""
import argparse
import json
import requests

from ollama_client import OllamaClient
from victim import _SYSTEM_PROMPT, _extract_short_answer

parser = argparse.ArgumentParser()
parser.add_argument("question", help="The question to ask")
parser.add_argument("--model",  default="llama3.1:8b")
parser.add_argument("--url",    default="http://127.0.0.1:11434")
parser.add_argument("--think",  action="store_true",
                    help="Enable chain-of-thought (only works with qwen3 and similar models)")
parser.add_argument("--max-tokens", type=int, default=2048)
args = parser.parse_args()

client = OllamaClient(base_url=args.url)

print(f"Q: {args.question}")
print(f"   model: {args.model}  think={args.think}\n")

# Call Ollama API directly to capture full response structure
payload = {
    "model": args.model,
    "messages": [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": args.question},
    ],
    "stream": False,
    "think": args.think,
    "options": {
        "temperature": 0.0,
        "num_predict": args.max_tokens,
    },
}

resp = requests.post(f"{args.url}/api/chat", json=payload, timeout=300)
data = resp.json()

# Show thinking block if present (qwen3 and similar)
thinking = data.get("message", {}).get("thinking", "")
content  = data.get("message", {}).get("content", "").strip()

if thinking:
    print("=== Thinking (chain-of-thought) ===")
    print(thinking)
    print()

print("=== Raw model output ===")
print(content)
print()

extracted = _extract_short_answer(content)
print("=== Extracted answer ===")
print(extracted)
