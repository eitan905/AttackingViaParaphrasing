"""
Evolutionary Adversarial Attacker.

Uses an iterative generate → filter → mutate loop to find semantically
equivalent paraphrases that cause the victim LLM to answer incorrectly.

Algorithm
---------
Generation 0:
  Use LLMParaphraser to create an initial population (size = n * pop_multiplier).

Each subsequent generation:
  1. Evaluate: score each candidate for semantic equivalence.
  2. Select: keep only semantically valid candidates (score ≥ threshold).
     These become the "survivors".
  3. Mutate: prompt the LLM to subtly vary each survivor in n_mutations ways,
     producing the next generation's population.

After all generations, return all unique semantically valid paraphrases seen
across every generation (up to n total).

Why this helps:
  The LLM paraphraser may initially produce candidates that are too similar
  (not diverse) or drift slightly outside semantic validity.  By iterating
  and mutating survivors, the search explores a wider region of the paraphrase
  space while staying semantically grounded.
"""

import json
import logging
import os
from typing import List, Optional

from ollama_client import OllamaClient, _parse_json_robust
from attackers.base import BaseAttacker
from attackers.llm_paraphraser import LLMParaphraser, _extract_string_list
from evaluators.semantic_judge import SemanticEquivalenceJudge

logger = logging.getLogger(__name__)

# Mutation prompt — takes one paraphrase and asks for slight variations
_MUTATE_SYSTEM = (
    "You are a paraphrase mutation engine for NLP research.\n"
    "Task: take an existing paraphrase of a question and generate subtle "
    "variations — small changes to word choice, phrasing, or sentence structure "
    "— while keeping the EXACT same meaning as both the paraphrase and the "
    "original question.\n"
    "Critical rules:\n"
    "- Do NOT change any factual content, names, dates, or numbers.\n"
    "- Each mutation must be different from the input paraphrase.\n"
    "- Aim for linguistic diversity across mutations.\n"
    'Output ONLY this JSON: {"paraphrases": ["m1", "m2", ...]}'
)

_MUTATE_USER = (
    'Original question: "{original}"\n'
    'Paraphrase to mutate: "{paraphrase}"\n\n'
    "Generate {n} subtle mutations that preserve the exact same meaning.\n"
    'Output ONLY valid JSON: {{"paraphrases": ["m1", "m2", ...]}}'
)


class EvolutionaryAttacker(BaseAttacker):
    """
    Evolutionary paraphrase search guided by a semantic equivalence judge.

    Args:
        client:           Shared OllamaClient.
        model:            Ollama model tag for mutation calls.
        semantic_judge:   SemanticEquivalenceJudge used to filter candidates.
        paraphrase_strategy: Strategy passed to the internal LLMParaphraser
                          for initial population generation.
        n_generations:    Number of evolution cycles (0 = single-shot, no evolution).
        pop_multiplier:   Initial population size = n × pop_multiplier.
        n_mutations:      Mutations generated per survivor per generation.
        temperature:      Sampling temperature for mutation calls.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "llama3.1:8b",
        semantic_judge: Optional[SemanticEquivalenceJudge] = None,
        paraphrase_strategy: str = "combined",
        n_generations: int = 3,
        pop_multiplier: int = 2,
        n_mutations: int = 3,
        temperature: float = 0.8,
        candidates_dir: Optional[str] = None,  # if set, save full candidate pool here
    ):
        self.client = client
        self.model = model
        self.semantic_judge = semantic_judge
        self.n_generations = n_generations
        self.pop_multiplier = pop_multiplier
        self.n_mutations = n_mutations
        self.temperature = temperature
        self.candidates_dir = candidates_dir

        # Internal single-shot paraphraser for initial population
        self._seed_gen = LLMParaphraser(
            client=client,
            model=model,
            strategy=paraphrase_strategy,
            temperature=temperature,
        )

    @property
    def name(self) -> str:
        return f"evolutionary_{self._seed_gen.strategy}_g{self.n_generations}"

    def generate_paraphrases(
        self,
        question: str,
        answers: List[str],
        n: int = 10,
    ) -> List[str]:
        """
        Run the evolutionary search and return up to `n` semantically
        valid paraphrases.

        Returns all unique valid paraphrases found across all generations,
        capped at `n`.  If no semantic_judge is configured, returns the
        initial population unfiltered.
        """
        # --- Generation 0: seed population ---
        initial_n = n * self.pop_multiplier
        population = self._seed_gen.generate_paraphrases(question, answers, initial_n)
        logger.info("[evo] Initial population: %d candidates", len(population))

        # all_valid_by_gen[g] = list of unique valid paraphrases from generation g
        # We keep them separated so we can prefer later (more evolved) generations.
        all_valid_by_gen: List[List[str]] = []
        seen: set = set()

        def _add_valid(candidates: List[str], gen_idx: int) -> None:
            while len(all_valid_by_gen) <= gen_idx:
                all_valid_by_gen.append([])
            for c in candidates:
                key = c.strip().lower()
                if key not in seen:
                    seen.add(key)
                    all_valid_by_gen[gen_idx].append(c.strip())

        for gen_idx in range(self.n_generations + 1):
            if not population:
                logger.info("[evo] Empty population at generation %d, stopping.", gen_idx)
                break

            # --- Filter: semantic equivalence ---
            if self.semantic_judge is not None:
                survivors = []
                for candidate in population:
                    result = self.semantic_judge.judge(candidate, question, answers)
                    logger.info(
                        "[evo] Gen%d  score=%.2f  valid=%s  %r",
                        gen_idx, result.score, result.is_equivalent, candidate,
                    )
                    if result.is_equivalent:
                        survivors.append(candidate)
                logger.info(
                    "[evo] Gen%d: %d/%d passed semantic filter",
                    gen_idx, len(survivors), len(population),
                )
            else:
                # No judge: accept everything
                survivors = list(population)

            _add_valid(survivors, gen_idx)

            # Stop after final generation (don't mutate)
            if gen_idx == self.n_generations:
                break

            if not survivors:
                logger.info("[evo] No survivors in gen%d, stopping evolution.", gen_idx)
                break

            # --- Mutate survivors → next generation ---
            next_population: List[str] = []
            for survivor in survivors:
                mutations = self._mutate(survivor, question, self.n_mutations)
                next_population.extend(mutations)

            logger.info(
                "[evo] Gen%d produced %d mutations from %d survivors",
                gen_idx, len(next_population), len(survivors),
            )
            population = next_population

        # Flatten preferring LATER generations (most evolved / diverse first)
        all_valid_flat: List[str] = []
        for gen_candidates in reversed(all_valid_by_gen):
            all_valid_flat.extend(gen_candidates)
        # Deduplicate while preserving order
        final_pool: List[str] = []
        final_seen: set = set()
        for c in all_valid_flat:
            k = c.lower()
            if k not in final_seen:
                final_seen.add(k)
                final_pool.append(c)

        total_valid = sum(len(g) for g in all_valid_by_gen)
        logger.info(
            "[evo] Done. Pool=%d unique valid paraphrases across %d gens. Returning %d.",
            total_valid, len(all_valid_by_gen), min(n, len(final_pool)),
        )

        # Save full candidate pool to disk if requested
        if self.candidates_dir:
            self._save_candidates(question, all_valid_by_gen, self.candidates_dir)

        return final_pool[:n]

    def _save_candidates(self, question: str, by_gen: List[List[str]], directory: str) -> None:
        """Persist all valid candidates grouped by generation."""
        os.makedirs(directory, exist_ok=True)
        # Safe filename from question
        import re
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:60]
        path = os.path.join(directory, f"candidates__{slug}.json")
        payload = {
            "question": question,
            "total_valid": sum(len(g) for g in by_gen),
            "by_generation": {
                f"gen{i}": candidates for i, candidates in enumerate(by_gen)
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("[evo] Full candidate pool saved → %s", path)

    # ------------------------------------------------------------------
    # Internal mutation
    # ------------------------------------------------------------------

    def _mutate(self, paraphrase: str, original: str, n: int) -> List[str]:
        """Ask the LLM to generate `n` subtle mutations of `paraphrase`."""
        messages = [
            {"role": "system", "content": _MUTATE_SYSTEM},
            {"role": "user",   "content": _MUTATE_USER.format(
                original=original, paraphrase=paraphrase, n=n
            )},
        ]
        try:
            raw = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
                think=False,
            )
            parsed = _parse_json_robust(raw)
            return _extract_string_list(parsed, paraphrase)
        except Exception as exc:
            logger.warning("[evo] Mutation failed for %r: %s", paraphrase, exc)
            return []
