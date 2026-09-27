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
  3. Rank: score each survivor with a fitness function (see below).
  4. Mutate: prompt the LLM to subtly vary the fittest survivors in
     n_mutations ways, producing the next generation's population.

After all generations, return the unique semantically valid paraphrases seen
across every generation (up to n total), fittest first.

Fitness
-------
Without a fitness victim the search is undirected: every survivor is mutated
and the pool comes back ordered by generation.  That explores widely but has
no pressure towards questions the victim actually struggles with.

With a fitness victim, each survivor is put to the victim and scored on
reasoning-length amplification — how much longer the victim's raw output is
for the paraphrase than for the original question.  Measured over knowledge
chains, amplification separates the cases the victim gets wrong (6.34x on the
one composition failure found) from the ones it handles (≤1.5x for nearly
all correct answers), so deliberation length is a usable proxy for strain.
A survivor whose answer already looks wrong gets a large bonus so that its
neighbourhood is explored in the next generation.

Why this helps:
  The LLM paraphraser may initially produce candidates that are too similar
  (not diverse) or drift slightly outside semantic validity.  By iterating
  and mutating survivors, the search explores a wider region of the paraphrase
  space while staying semantically grounded.
"""

import json
import logging
import os
from dataclasses import dataclass
from typing import List, Optional

from ollama_client import OllamaClient
from attackers.base import BaseAttacker
from attackers.llm_paraphraser import (
    _PARAPHRASE_SCHEMA,
    LLMParaphraser,
    _extract_string_list,
)
from evaluators.semantic_judge import SemanticEquivalenceJudge
from victim import VictimModel

logger = logging.getLogger(__name__)

# Added to the fitness of a survivor the victim already appears to answer
# wrongly, so it outranks every amplification score and its neighbourhood is
# always explored.  This is a cheap substring heuristic, not the answer judge —
# the experiment still decides what counts as a successful attack.
_WRONG_ANSWER_BONUS = 10.0


@dataclass
class _Scored:
    """A semantically valid paraphrase together with its fitness measurement."""
    text: str
    generation: int
    fitness: float = 0.0
    amplification: float = 0.0
    victim_answer: str = ""
    victim_raw_len: int = 0
    looks_wrong: bool = False

    def to_dict(self) -> dict:
        return {
            "paraphrase": self.text,
            "generation": self.generation,
            "fitness": round(self.fitness, 3),
            "amplification": round(self.amplification, 3),
            "victim_answer": self.victim_answer,
            "victim_raw_len": self.victim_raw_len,
            "looks_wrong": self.looks_wrong,
        }

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

# Used when fitness is amplification: keep meaning, but push the parent
# further along the axis that actually strained the victim — replace remaining
# proper nouns with true, uniquely identifying attributes.
_MUTATE_INDIRECT_SYSTEM = (
    "You are a paraphrase mutation engine for adversarial NLP research.\n"
    "Task: take a paraphrase that already identifies its subject indirectly "
    "and push it further in that direction.\n"
    "Replace any remaining proper names, dates, or well-known labels with "
    "TRUE, uniquely identifying peripheral attributes (where it is housed, "
    "what it is made of, who rediscovered it, a related event, a material, "
    "a location). Do NOT invent facts.\n"
    "Critical rules:\n"
    "- The meaning and correct answer must stay IDENTICAL to the original.\n"
    "- Every attribute must be factually true and uniquely identifying.\n"
    "- Do NOT embed the answer in the question. Never include the year, name, "
    "or other value that the original question is asking for.\n"
    "- Each mutation must differ from the input paraphrase.\n"
    'Output ONLY this JSON: {"paraphrases": ["m1", "m2", ...]}'
)

_MUTATE_INDIRECT_USER = (
    'Original question: "{original}"\n'
    'Paraphrase to mutate: "{paraphrase}"\n\n'
    "Generate {n} mutations that identify the subject even more indirectly "
    "while keeping the exact same meaning and correct answer.\n"
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
        fitness_victim:   If given, survivors are scored by reasoning-length
                          amplification against this victim and only the
                          fittest are mutated.  If None, the search is
                          undirected and every survivor is mutated.
        elite_size:       How many top-fitness survivors to mutate per
                          generation.  Ignored when fitness_victim is None.
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
        fitness_victim: Optional[VictimModel] = None,
        elite_size: int = 4,
    ):
        self.client = client
        self.model = model
        self.semantic_judge = semantic_judge
        self.n_generations = n_generations
        self.pop_multiplier = pop_multiplier
        self.n_mutations = n_mutations
        self.temperature = temperature
        self.candidates_dir = candidates_dir
        self.fitness_victim = fitness_victim
        self.elite_size = elite_size

        # Internal single-shot paraphraser for initial population
        self._seed_gen = LLMParaphraser(
            client=client,
            model=model,
            strategy=paraphrase_strategy,
            temperature=temperature,
        )

    @property
    def name(self) -> str:
        suffix = "_fit" if self.fitness_victim is not None else ""
        return f"evolutionary_{self._seed_gen.strategy}_g{self.n_generations}{suffix}"

    def generate_paraphrases(
        self,
        question: str,
        answers: List[str],
        n: int = 10,
    ) -> List[str]:
        """
        Run the evolutionary search and return up to `n` semantically
        valid paraphrases.

        Returns unique valid paraphrases found across all generations, capped
        at `n` and ordered fittest first (or most-evolved first when no
        fitness victim is configured).  If no semantic_judge is configured,
        returns the initial population unfiltered.
        """
        # --- Generation 0: seed population ---
        initial_n = n * self.pop_multiplier
        population = self._seed_gen.generate_paraphrases(question, answers, initial_n)
        logger.info("[evo] Initial population: %d candidates", len(population))

        # Baseline deliberation length on the unmodified question — the
        # denominator for every amplification score below.
        baseline_len = 0
        if self.fitness_victim is not None:
            _, raw = self.fitness_victim.answer_verbose(question)
            baseline_len = len(raw)
            logger.info("[evo] Fitness baseline: victim wrote %d chars on the "
                        "original question", baseline_len)

        # all_valid_by_gen[g] = unique valid paraphrases from generation g.
        # We keep them separated so we can prefer later (more evolved) generations.
        all_valid_by_gen: List[List[_Scored]] = []
        seen: set = set()

        def _add_valid(scored: List[_Scored], gen_idx: int) -> None:
            while len(all_valid_by_gen) <= gen_idx:
                all_valid_by_gen.append([])
            for s in scored:
                key = s.text.strip().lower()
                if key not in seen:
                    seen.add(key)
                    all_valid_by_gen[gen_idx].append(s)

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

            scored = self._score_fitness(survivors, question, answers,
                                         baseline_len, gen_idx)
            _add_valid(scored, gen_idx)

            # Stop after final generation (don't mutate)
            if gen_idx == self.n_generations:
                break

            if not scored:
                logger.info("[evo] No survivors in gen%d, stopping evolution.", gen_idx)
                break

            # --- Select: only the fittest get to reproduce ---
            parents = self._select_parents(scored)

            # --- Mutate parents → next generation ---
            next_population: List[str] = []
            for parent in parents:
                mutations = self._mutate(parent.text, question, self.n_mutations)
                next_population.extend(mutations)

            logger.info(
                "[evo] Gen%d produced %d mutations from %d/%d survivors",
                gen_idx, len(next_population), len(parents), len(scored),
            )
            population = next_population

        if self.fitness_victim is not None:
            # Directed search: fittest first, regardless of generation.
            final_pool = sorted(
                (s for gen in all_valid_by_gen for s in gen),
                key=lambda s: s.fitness,
                reverse=True,
            )
        else:
            # Undirected search: prefer LATER generations (most evolved / diverse).
            final_pool = [s for gen in reversed(all_valid_by_gen) for s in gen]

        total_valid = len(final_pool)
        logger.info(
            "[evo] Done. Pool=%d unique valid paraphrases across %d gens. Returning %d.",
            total_valid, len(all_valid_by_gen), min(n, total_valid),
        )
        if self.fitness_victim is not None and final_pool:
            logger.info("[evo] Top of pool by fitness:")
            for s in final_pool[:5]:
                logger.info(
                    "[evo]   fit=%.2f amp=%.2fx gen%d wrong=%s %r -> %r",
                    s.fitness, s.amplification, s.generation, s.looks_wrong,
                    s.text, s.victim_answer,
                )

        # Save full candidate pool to disk if requested
        if self.candidates_dir:
            self._save_candidates(question, all_valid_by_gen, baseline_len,
                                  self.candidates_dir)

        return [s.text for s in final_pool[:n]]

    def _save_candidates(self, question: str, by_gen: List[List[_Scored]],
                         baseline_len: int, directory: str) -> None:
        """Persist all valid candidates grouped by generation."""
        os.makedirs(directory, exist_ok=True)
        # Safe filename from question
        import re
        slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")[:60]
        path = os.path.join(directory, f"candidates__{slug}.json")
        payload = {
            "question": question,
            "total_valid": sum(len(g) for g in by_gen),
            "fitness": "amplification" if self.fitness_victim is not None else "none",
            "baseline_raw_len": baseline_len,
            "by_generation": {
                f"gen{i}": [s.to_dict() for s in candidates]
                for i, candidates in enumerate(by_gen)
            },
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("[evo] Full candidate pool saved → %s", path)

    # ------------------------------------------------------------------
    # Fitness and selection
    # ------------------------------------------------------------------

    def _score_fitness(
        self,
        survivors: List[str],
        question: str,
        answers: List[str],
        baseline_len: int,
        gen_idx: int,
    ) -> List[_Scored]:
        """
        Attach a fitness score to each survivor.

        With no fitness victim every survivor scores 0 and selection is a
        no-op.  Otherwise fitness is the reasoning-length amplification over
        the original question, plus a bonus when the victim's answer already
        looks wrong.
        """
        if self.fitness_victim is None:
            return [_Scored(text=s.strip(), generation=gen_idx) for s in survivors]

        accepted = [a.lower() for a in answers]
        scored: List[_Scored] = []
        for survivor in survivors:
            try:
                short, raw = self.fitness_victim.answer_verbose(survivor)
            except Exception as exc:
                logger.warning("[evo] Fitness probe failed for %r: %s", survivor, exc)
                scored.append(_Scored(text=survivor.strip(), generation=gen_idx))
                continue

            amplification = len(raw) / baseline_len if baseline_len else 0.0
            looks_wrong = not any(a in short.lower() for a in accepted)
            scored.append(_Scored(
                text=survivor.strip(),
                generation=gen_idx,
                fitness=amplification + (_WRONG_ANSWER_BONUS if looks_wrong else 0.0),
                amplification=amplification,
                victim_answer=short,
                victim_raw_len=len(raw),
                looks_wrong=looks_wrong,
            ))
            logger.info(
                "[evo] Gen%d  amp=%.2fx  wrong=%-5s  %r -> %r",
                gen_idx, amplification, looks_wrong, survivor, short,
            )
        return scored

    def _select_parents(self, scored: List[_Scored]) -> List[_Scored]:
        """Pick the survivors whose neighbourhood is worth exploring."""
        if self.fitness_victim is None:
            return scored
        ranked = sorted(scored, key=lambda s: s.fitness, reverse=True)
        return ranked[:max(1, self.elite_size)]

    # ------------------------------------------------------------------
    # Internal mutation
    # ------------------------------------------------------------------

    def _mutate(self, paraphrase: str, original: str, n: int) -> List[str]:
        """Ask the LLM to generate `n` mutations of `paraphrase`."""
        if self.fitness_victim is not None:
            system, user_tmpl = _MUTATE_INDIRECT_SYSTEM, _MUTATE_INDIRECT_USER
        else:
            system, user_tmpl = _MUTATE_SYSTEM, _MUTATE_USER
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_tmpl.format(
                original=original, paraphrase=paraphrase, n=n
            )},
        ]
        try:
            parsed = self.client.chat_json(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
                think=False,
                schema=_PARAPHRASE_SCHEMA,
            )
            return _extract_string_list(parsed, paraphrase)
        except Exception as exc:
            logger.warning("[evo] Mutation failed for %r: %s", paraphrase, exc)
            return []
