"""
Experiment orchestrator.

Pipeline per question
---------------------
1. Attacker generates paraphrase candidates.
2. Semantic judge filters: only semantically-equivalent paraphrases proceed.
3. Victim answers each valid paraphrase.
4. Answer judge decides whether the victim's answer is correct.
5. Attack "succeeds" when a semantically-valid paraphrase fools the victim.

The attack_method config flag selects between:
  - "llm_single_shot"  — LLMParaphraser generates candidates in one pass,
                         then they go through the semantic judge.
  - "evolutionary"     — EvolutionaryAttacker runs an internal
                         generate → semantic-filter → mutate loop, then
                         returns semantically-valid paraphrases directly.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List

from config import ExperimentConfig
from dataset import load_questions
from ollama_client import OllamaClient
from victim import VictimModel
from attackers import LLMParaphraser, EvolutionaryAttacker
from evaluators.exact_match import ExactMatchEvaluator
from evaluators.llm_judge import LLMJudgeEvaluator
from evaluators.semantic_judge import SemanticEquivalenceJudge

logger = logging.getLogger(__name__)


def run_experiment(cfg: ExperimentConfig) -> Dict[str, Any]:
    """Execute the full experiment and return the results dict."""
    cfg.validate()

    client = OllamaClient(base_url=cfg.ollama_base_url)
    client.require_available()

    # --- Components ---
    victim = VictimModel(client, cfg.victim_model, temperature=cfg.victim_temperature,
                         max_tokens=cfg.victim_max_tokens)

    semantic_judge = SemanticEquivalenceJudge(
        client=client,
        model=cfg.judge_model,
        temperature=cfg.judge_temperature,
        threshold=cfg.semantic_threshold,
    )

    attacker = _build_attacker(cfg, client, semantic_judge, victim)

    evaluators = _build_evaluators(cfg, client)

    # --- Dataset ---
    questions = load_questions(cfg.dataset_path, cfg.n_questions, cfg.random_seed,
                               question_ids=cfg.question_ids)
    logger.info("Loaded %d questions.", len(questions))

    # --- Main loop ---
    question_results: List[Dict] = []
    for q in questions:
        logger.info("Processing question: %s", q["question"])
        q_result = _process_question(
            q=q,
            attacker=attacker,
            victim=victim,
            semantic_judge=semantic_judge,
            evaluators=evaluators,
            cfg=cfg,
        )
        question_results.append(q_result)
        _log_question_summary(q_result)

    # --- Aggregate stats ---
    stats = _aggregate(question_results)

    # --- Save ---
    output = {
        "config": _cfg_to_dict(cfg),
        "run_id": cfg.run_id(),
        "timestamp": datetime.now().isoformat(),
        "stats": stats,
        "questions": question_results,
    }
    _save_results(output, cfg)
    return output


# ---------------------------------------------------------------------------
# Question-level processing
# ---------------------------------------------------------------------------

def _process_question(q, attacker, victim, semantic_judge, evaluators, cfg):
    """Process one question and return its result dict."""
    question_text = q["question"]
    ground_truth  = q["answers"]

    # Step 1: baseline — victim answers the original question
    baseline_answer = victim.answer(question_text)
    baseline_eval = _eval_answer(baseline_answer, ground_truth, evaluators,
                                 question=question_text)

    # Step 2: generate paraphrase candidates
    candidates = attacker.generate_paraphrases(
        question=question_text,
        answers=ground_truth,
        n=cfg.n_paraphrases,
    )
    logger.info("  Got %d candidates from attacker.", len(candidates))

    # Step 3 (single-shot only): run semantic filter explicitly.
    # Evolutionary already filters internally; we do a lightweight pass here
    # to attach scores to the output for transparency.
    paraphrase_results = []
    for candidate in candidates:
        sem_result = semantic_judge.judge(candidate, question_text, ground_truth)

        # For evolutionary, candidates already passed the judge once, but we
        # score again for logging.  For single-shot, this is the gate.
        is_valid = sem_result.is_equivalent

        if not is_valid and cfg.attack_method == "llm_single_shot":
            paraphrase_results.append({
                "paraphrase": candidate,
                "semantic_score": sem_result.score,
                "semantic_valid": False,
                "semantic_rationale": sem_result.rationale,
                "victim_answer": None,
                "correct": None,
                "eval_scores": {},
                "attack_success": False,
            })
            continue

        # Step 4: victim answers the paraphrase
        victim_answer = victim.answer(candidate)

        # Step 5: judge whether victim's answer is correct
        # Pass original question as context so the judge knows what's being asked
        eval_scores = _eval_answer(victim_answer, ground_truth, evaluators,
                                   question=question_text)
        correct = _is_correct(eval_scores)

        paraphrase_results.append({
            "paraphrase": candidate,
            "semantic_score": sem_result.score,
            "semantic_valid": is_valid,
            "semantic_rationale": sem_result.rationale,
            "victim_answer": victim_answer,
            "correct": correct,
            "eval_scores": eval_scores,
            # Attack succeeds if the paraphrase is valid BUT the victim is wrong
            "attack_success": is_valid and not correct,
        })

    valid = [r for r in paraphrase_results if r["semantic_valid"]]
    successful_attacks = [r for r in valid if r["attack_success"]]

    return {
        "id": q.get("id", question_text[:40]),
        "question": question_text,
        "ground_truth": ground_truth,
        "baseline_answer": baseline_answer,
        "baseline_correct": _is_correct(baseline_eval),
        "baseline_eval": baseline_eval,
        "n_candidates": len(candidates),
        "n_valid_paraphrases": len(valid),
        "n_attacks_succeeded": len(successful_attacks),
        "attack_success_rate": (
            len(successful_attacks) / len(valid) if valid else 0.0
        ),
        "paraphrases": paraphrase_results,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_attacker(cfg, client, semantic_judge, victim):
    if cfg.attack_method == "evolutionary":
        candidates_dir = (
            os.path.join(cfg.results_dir, "candidates") if cfg.save_candidates else None
        )
        return EvolutionaryAttacker(
            client=client,
            model=cfg.attacker_model,
            semantic_judge=semantic_judge,
            paraphrase_strategy=cfg.paraphrase_strategy,
            n_generations=cfg.n_generations,
            pop_multiplier=cfg.pop_multiplier,
            n_mutations=cfg.n_mutations,
            temperature=cfg.attacker_temperature,
            candidates_dir=candidates_dir,
            # Steering the search means probing the same victim the attack
            # will ultimately be scored against.
            fitness_victim=victim if cfg.fitness == "amplification" else None,
            elite_size=cfg.elite_size,
        )
    else:
        return LLMParaphraser(
            client=client,
            model=cfg.attacker_model,
            strategy=cfg.paraphrase_strategy,
            temperature=cfg.attacker_temperature,
        )


def _build_evaluators(cfg, client):
    evaluators = []
    if cfg.evaluator in ("exact_match", "both"):
        evaluators.append(ExactMatchEvaluator())
    if cfg.evaluator in ("llm_judge", "both"):
        evaluators.append(LLMJudgeEvaluator(
            client=client,
            model=cfg.judge_model,
            temperature=cfg.judge_temperature,
        ))
    return evaluators


def _eval_answer(answer: str, ground_truth: List[str], evaluators,
                 question: str = "") -> Dict:
    scores = {}
    for ev in evaluators:
        # Pass question to LLM judge for context; ignored by exact_match
        if hasattr(ev, 'evaluate') and 'question' in ev.evaluate.__code__.co_varnames:
            result = ev.evaluate(answer, ground_truth, question=question)
        else:
            result = ev.evaluate(answer, ground_truth)
        scores[ev.name] = {
            "score": result.score,
            "correct": result.correct,
        }
    return scores


def _is_correct(eval_scores: Dict) -> bool:
    """True if ANY evaluator says the answer is correct."""
    if not eval_scores:
        return False
    return any(v["correct"] for v in eval_scores.values())


def _aggregate(question_results: List[Dict]) -> Dict:
    total_q = len(question_results)
    total_valid = sum(r["n_valid_paraphrases"] for r in question_results)
    total_attacked = sum(r["n_attacks_succeeded"] for r in question_results)
    baseline_correct = sum(1 for r in question_results if r["baseline_correct"])

    return {
        "n_questions": total_q,
        "baseline_accuracy": baseline_correct / total_q if total_q else 0,
        "total_valid_paraphrases": total_valid,
        "total_attack_successes": total_attacked,
        "overall_attack_success_rate": (
            total_attacked / total_valid if total_valid else 0.0
        ),
    }


def _log_question_summary(q_result: Dict) -> None:
    logger.info(
        "  Q: %r | valid=%d/%d | attacks=%d | rate=%.0f%%",
        q_result["question"][:60],
        q_result["n_valid_paraphrases"],
        q_result["n_candidates"],
        q_result["n_attacks_succeeded"],
        q_result["attack_success_rate"] * 100,
    )


def _cfg_to_dict(cfg: ExperimentConfig) -> Dict:
    import dataclasses
    return dataclasses.asdict(cfg)


def _save_results(output: Dict, cfg: ExperimentConfig) -> None:
    os.makedirs(cfg.results_dir, exist_ok=True)
    filename = f"{cfg.run_id()}.json"
    path = os.path.join(cfg.results_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", path)
    print(f"\n✓ Results saved → {path}")
