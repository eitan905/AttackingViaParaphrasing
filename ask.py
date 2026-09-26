#!/usr/bin/env python3
"""
Ask the victim model a single question and print its full raw output
plus the extracted short answer.

Usage:
    python3 ask.py "Which ancient civilization built Machu Picchu?"
    python3 ask.py "Who painted the Mona Lisa?" --model gemma3:12b
    python3 ask.py "What is the capital of France?" --url http://127.0.0.1:11435
"""
import argparse

from ollama_client import OllamaClient
from victim import _SYSTEM_PROMPT, _extract_short_answer

parser = argparse.ArgumentParser()
parser.add_argument("question", help="The question to ask")
parser.add_argument("--model", default="llama3.1:8b")
parser.add_argument("--url",   default="http://127.0.0.1:11434")
args = parser.parse_args()

client = OllamaClient(base_url=args.url)

print(f"Q: {args.question}")
print(f"   model: {args.model}\n")

# Get raw output directly (bypass VictimModel wrapper to see everything)
raw = client.chat(
    model=args.model,
    messages=[
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": args.question},
    ],
    temperature=0.0,
    max_tokens=512,
    think=False,
)

extracted = _extract_short_answer(raw)

print("=== Raw model output ===")
print(raw)
print()
print("=== Extracted answer ===")
print(extracted)
