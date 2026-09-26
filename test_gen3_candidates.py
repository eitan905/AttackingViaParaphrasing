"""Test the 16 hand-picked Gen3 paraphrases on the victim model."""
import argparse
import json
import logging

logging.basicConfig(level=logging.WARNING)

from ollama_client import OllamaClient
from victim import VictimModel
from evaluators.llm_judge import LLMJudgeEvaluator

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="llama3.1:8b")
parser.add_argument("--url",   default="http://127.0.0.1:11434")
args = parser.parse_args()

client = OllamaClient(base_url=args.url)
victim = VictimModel(client, args.model, temperature=0.0)
judge  = LLMJudgeEvaluator(client, model=args.model, temperature=0.0)

original     = "Which ancient civilization built Machu Picchu?"
ground_truth = ["Inca", "Incas", "Inca Empire", "Incan civilization"]

baseline = victim.answer(original)
print(f"BASELINE: {original!r}")
print(f"  → {baseline!r}\n")

with open("results/gen3_candidates_of_interest.json") as f:
    candidates = json.load(f)["selected_from_gen3"]

results = []
for i, para in enumerate(candidates, 1):
    answer  = victim.answer(para)
    correct = judge.evaluate(answer, ground_truth).correct
    results.append({"paraphrase": para, "victim_answer": answer, "correct": correct})
    label = "  correct" if correct else "✓ ATTACK!"
    print(f"[{i:02d}] [{label}]  {para}")
    print(f"       → {answer!r}")

n_attacks = sum(1 for r in results if not r["correct"])
print(f"\n{'='*50}")
print(f"Attacks: {n_attacks}/{len(results)} ({n_attacks/len(results):.0%})")

with open("results/gen3_candidates_tested.json", "w") as f:
    json.dump({
        "original": original,
        "ground_truth": ground_truth,
        "baseline_answer": baseline,
        "results": results,
        "attacks": [r for r in results if not r["correct"]],
    }, f, indent=2, ensure_ascii=False)

print("Saved → results/gen3_candidates_tested.json")
