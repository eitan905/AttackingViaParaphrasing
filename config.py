"""
Central configuration for attack experiments.

Every axis that might vary across runs (model choice, attack strategy,
number of samples, …) lives here so that swapping components is a
one-liner on the command line.
"""

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from typing import List, Literal, Optional


# ---------------------------------------------------------------------------
# Available strategies (prompt templates live in attackers/llm_paraphraser.py)
# ---------------------------------------------------------------------------

PARAPHRASE_STRATEGIES = [
    "combined",    # mixture of synonym, structural, and contextual transforms (default)
    "synonym",     # replace words/phrases with synonyms
    "structural",  # change sentence structure (active/passive, word order)
    "contextual",  # add neutral framing without changing meaning
    "unnatural",   # syntactically valid but maximally non-human / awkward phrasing
]

ATTACK_METHODS = [
    "llm_single_shot",   # generate paraphrases in one LLM call, filter by semantic judge
    "evolutionary",      # iterative generate → semantic-filter → mutate loop
]

EVALUATOR_CHOICES = ["exact_match", "llm_judge", "both"]


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    # ---- Ollama server ----
    ollama_base_url: str = "http://127.0.0.1:11434"

    # ---- Models ----
    attacker_model: str = "llama3.1:8b"
    victim_model: str = "llama3.1:8b"
    judge_model: str = "llama3.1:8b"

    # ---- Attack method ----
    attack_method: str = "llm_single_shot"   # "llm_single_shot" | "evolutionary"
    paraphrase_strategy: str = "combined"    # see PARAPHRASE_STRATEGIES

    # ---- Paraphrase generation ----
    n_paraphrases: int = 10   # how many valid paraphrases to return per question

    # ---- Evolutionary settings (only used when attack_method="evolutionary") ----
    n_generations: int = 3        # evolution cycles
    pop_multiplier: int = 2       # initial population = n_paraphrases × pop_multiplier
    n_mutations: int = 3          # LLM mutations per survivor per generation
    semantic_threshold: float = 0.7  # min semantic score to accept a paraphrase

    # ---- Dataset ----
    dataset_path: Optional[str] = None   # None → built-in data/sample_questions.json
    n_questions: int = 3
    random_seed: int = 42
    question_ids: Optional[List[str]] = None  # if set, use these specific IDs in order

    # ---- Evaluation ----
    evaluator: str = "both"    # "exact_match" | "llm_judge" | "both"

    # ---- Output ----
    results_dir: str = "results"
    save_candidates: bool = True   # save full evolutionary candidate pool to disk

    # ---- Generation temperatures ----
    attacker_temperature: float = 0.9
    victim_temperature: float = 0.0
    judge_temperature: float = 0.0

    def validate(self) -> None:
        if self.attack_method not in ATTACK_METHODS:
            raise ValueError(
                f"Unknown attack_method {self.attack_method!r}. "
                f"Choose from: {ATTACK_METHODS}"
            )
        if self.paraphrase_strategy not in PARAPHRASE_STRATEGIES:
            raise ValueError(
                f"Unknown paraphrase_strategy {self.paraphrase_strategy!r}. "
                f"Choose from: {PARAPHRASE_STRATEGIES}"
            )
        if self.evaluator not in EVALUATOR_CHOICES:
            raise ValueError(
                f"Unknown evaluator {self.evaluator!r}. "
                f"Choose from: {EVALUATOR_CHOICES}"
            )
        if self.n_paraphrases < 1:
            raise ValueError("n_paraphrases must be >= 1")
        if self.n_questions < 1:
            raise ValueError("n_questions must be >= 1")

    def run_id(self) -> str:
        """Unique, filesystem-safe identifier — used as the output filename stem."""
        def slug(s: str) -> str:
            return re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_")

        return (
            f"method_{slug(self.attack_method)}_{slug(self.paraphrase_strategy)}"
            f"__victim_{slug(self.victim_model)}"
            f"__judge_{slug(self.judge_model)}"
            f"__q{self.n_questions}_p{self.n_paraphrases}"
            + (f"_g{self.n_generations}" if self.attack_method == "evolutionary" else "")
        )

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> ExperimentConfig:
    defaults = ExperimentConfig()

    parser = argparse.ArgumentParser(
        description="Run a subtle-paraphrasing adversarial attack experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Ollama
    parser.add_argument("--ollama-base-url", default=defaults.ollama_base_url)

    # Models
    parser.add_argument("--attacker-model", default=defaults.attacker_model)
    parser.add_argument("--victim-model",   default=defaults.victim_model)
    parser.add_argument("--judge-model",    default=defaults.judge_model)

    # Attack method
    parser.add_argument("--attack-method", default=defaults.attack_method,
                        choices=ATTACK_METHODS,
                        help="llm_single_shot: one LLM call per question; "
                             "evolutionary: iterative mutate+filter loop")
    parser.add_argument("--paraphrase-strategy", default=defaults.paraphrase_strategy,
                        choices=PARAPHRASE_STRATEGIES,
                        help="Type of linguistic transformation applied")
    parser.add_argument("--n-paraphrases", type=int, default=defaults.n_paraphrases,
                        help="Target number of valid paraphrases per question")

    # Evolutionary settings
    parser.add_argument("--n-generations",    type=int,   default=defaults.n_generations)
    parser.add_argument("--pop-multiplier",   type=int,   default=defaults.pop_multiplier)
    parser.add_argument("--n-mutations",      type=int,   default=defaults.n_mutations)
    parser.add_argument("--semantic-threshold", type=float, default=defaults.semantic_threshold)

    # Dataset
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--n-questions",  type=int, default=defaults.n_questions)
    parser.add_argument("--random-seed",  type=int, default=defaults.random_seed)
    parser.add_argument("--question-ids", nargs="+", default=None,
                        metavar="ID",
                        help="Use specific question IDs in order, e.g. --question-ids nq_004 nq_009 nq_001")

    # Evaluation
    parser.add_argument("--evaluator", default=defaults.evaluator,
                        choices=EVALUATOR_CHOICES)

    # Output
    parser.add_argument("--results-dir", default=defaults.results_dir)
    parser.add_argument("--no-save-candidates", dest="save_candidates",
                        action="store_false", default=True,
                        help="Skip saving full evolutionary candidate pool to disk")

    # Temperatures
    parser.add_argument("--attacker-temperature", type=float,
                        default=defaults.attacker_temperature)
    parser.add_argument("--victim-temperature", type=float,
                        default=defaults.victim_temperature)
    parser.add_argument("--judge-temperature", type=float,
                        default=defaults.judge_temperature)

    args = parser.parse_args(argv)

    cfg = ExperimentConfig(
        ollama_base_url=args.ollama_base_url,
        attacker_model=args.attacker_model,
        victim_model=args.victim_model,
        judge_model=args.judge_model,
        attack_method=args.attack_method,
        paraphrase_strategy=args.paraphrase_strategy,
        n_paraphrases=args.n_paraphrases,
        n_generations=args.n_generations,
        pop_multiplier=args.pop_multiplier,
        n_mutations=args.n_mutations,
        semantic_threshold=args.semantic_threshold,
        dataset_path=args.dataset_path,
        n_questions=args.n_questions,
        random_seed=args.random_seed,
        question_ids=args.question_ids,
        evaluator=args.evaluator,
        results_dir=args.results_dir,
        save_candidates=args.save_candidates,
        attacker_temperature=args.attacker_temperature,
        victim_temperature=args.victim_temperature,
        judge_temperature=args.judge_temperature,
    )
    cfg.validate()
    return cfg
