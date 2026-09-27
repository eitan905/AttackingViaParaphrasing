#!/usr/bin/env python3
"""
Entry point for the AttackingViaParaphrasing experiment.

Usage examples
--------------
# Single-shot attack (default), 1 question, 5 paraphrases:
python run_experiment.py --n-questions 1 --n-paraphrases 5

# Evolutionary attack, 1 question, 10 paraphrases, 3 generations:
python run_experiment.py --attack-method evolutionary \
    --n-questions 1 --n-paraphrases 10 --n-generations 3

# Change victim/attacker model:
python run_experiment.py --victim-model gemma3:12b --attacker-model llama3.1:8b

# Use a different paraphrase strategy:
python run_experiment.py --paraphrase-strategy synonym --n-paraphrases 5
"""

import logging
import sys

from config import parse_args
from experiment import run_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    cfg = parse_args()

    print("\n=== AttackingViaParaphrasing ===")
    print(f"  Attack method      : {cfg.attack_method}")
    print(f"  Paraphrase strategy: {cfg.paraphrase_strategy}")
    print(f"  Attacker model     : {cfg.attacker_model}")
    print(f"  Victim model       : {cfg.victim_model}")
    print(f"  Judge model        : {cfg.judge_model}")
    print(f"  Questions          : {cfg.n_questions}")
    print(f"  Paraphrases/Q      : {cfg.n_paraphrases}")
    if cfg.attack_method == "evolutionary":
        print(f"  Generations        : {cfg.n_generations}")
        print(f"  Pop multiplier     : {cfg.pop_multiplier}×")
        print(f"  Mutations/survivor : {cfg.n_mutations}")
        print(f"  Fitness            : {cfg.fitness}")
        if cfg.fitness != "none":
            print(f"  Elite size         : {cfg.elite_size}")
    print(f"  Semantic threshold : {cfg.semantic_threshold}")
    print(f"  Run ID             : {cfg.run_id()}")
    print()

    output = run_experiment(cfg)

    stats = output["stats"]
    print("\n=== Summary ===")
    print(f"  Questions tested      : {stats['n_questions']}")
    print(f"  Baseline accuracy     : {stats['baseline_accuracy']:.0%}")
    print(f"  Valid paraphrases     : {stats['total_valid_paraphrases']}")
    print(f"  Successful attacks    : {stats['total_attack_successes']}")
    print(f"  Attack success rate   : {stats['overall_attack_success_rate']:.0%}")

    # Per-question detail
    print("\n--- Per-question breakdown ---")
    for q in output["questions"]:
        print(f"\n  Q: {q['question']}")
        print(f"     Baseline answer  : {q['baseline_answer']}  "
              f"({'✓' if q['baseline_correct'] else '✗'})")
        print(f"     Valid paraphrases: {q['n_valid_paraphrases']}/{q['n_candidates']}")
        print(f"     Attacks succeeded: {q['n_attacks_succeeded']}")
        for p in q["paraphrases"]:
            status = "✗ invalid" if not p["semantic_valid"] else (
                "✓ attack!" if p["attack_success"] else "  correct"
            )
            sem_score = f"{p['semantic_score']:.2f}" if p["semantic_score"] is not None else "n/a"
            print(f"       [{status}] sem={sem_score}  {p['paraphrase']!r}")
            if p.get("victim_answer"):
                print(f"              victim: {p['victim_answer']!r}")


if __name__ == "__main__":
    main()
