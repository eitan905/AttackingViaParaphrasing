#!/usr/bin/env python3
"""
Ask the victim model a single question and print its answer.

Usage:
    python3 ask.py "Which ancient civilization built Machu Picchu?"
    python3 ask.py "Who painted the Mona Lisa?" --model gemma3:12b
    python3 ask.py "What is the capital of France?" --url http://127.0.0.1:11435
"""
import argparse
import sys

from ollama_client import OllamaClient
from victim import VictimModel

parser = argparse.ArgumentParser()
parser.add_argument("question", help="The question to ask")
parser.add_argument("--model", default="llama3.1:8b", help="Ollama model tag")
parser.add_argument("--url", default="http://127.0.0.1:11434", help="Ollama base URL")
args = parser.parse_args()

client = OllamaClient(base_url=args.url)
victim = VictimModel(client, model=args.model, temperature=0.0)

print(f"Q: {args.question}")
answer = victim.answer(args.question)
print(f"A: {answer}")
