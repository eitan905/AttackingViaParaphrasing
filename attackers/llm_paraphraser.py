"""
LLM-based paraphraser.

All strategies produce SEMANTICALLY EQUIVALENT paraphrases — questions that
mean the same thing as the original and have the same correct answer, just
phrased differently.  The "attack" is the discovery that even semantically
identical questions can cause an LLM to answer incorrectly due to framing
sensitivity.

Each strategy applies a different type of linguistic transformation.
Adding a new strategy: add an entry to the STRATEGIES dict below.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ollama_client import OllamaClient, _parse_json_robust
from attackers.base import BaseAttacker

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Strategy definitions
# Each value is (system_prompt, user_message_template).
# Placeholder available in user message: {question}, {n}
# ALL strategies must preserve the question's meaning and correct answer.
# ---------------------------------------------------------------------------

STRATEGIES: Dict[str, Tuple[str, str]] = {
    # ------------------------------------------------------------------
    # synonym: replace words/phrases with synonyms or equivalent expressions
    # ------------------------------------------------------------------
    "synonym": (
        "You are a paraphrase generator for NLP research.\n"
        "Task: rewrite questions by substituting key words and phrases with "
        "synonyms or equivalent expressions.\n"
        "Critical rules:\n"
        "- The meaning and correct answer must be IDENTICAL to the original.\n"
        "- Do NOT change dates, names, numbers, or facts — only the surrounding words.\n"
        "- Every variant must be a natural, fluent question.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} paraphrases using synonyms and equivalent expressions.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
    # ------------------------------------------------------------------
    # structural: change sentence structure while preserving meaning
    # ------------------------------------------------------------------
    "structural": (
        "You are a paraphrase generator for NLP research.\n"
        "Task: rewrite questions by changing their grammatical structure — "
        "active↔passive voice, question word choice, clause order, nominalisation — "
        "while keeping the meaning and correct answer IDENTICAL.\n"
        "Critical rules:\n"
        "- Do NOT change any factual content.\n"
        "- Every variant must be a natural, fluent question.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} structurally varied paraphrases with identical meaning.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
    # ------------------------------------------------------------------
    # contextual: add neutral framing or embed the question differently
    # ------------------------------------------------------------------
    "contextual": (
        "You are a paraphrase generator for NLP research.\n"
        "Task: rewrite questions by adding neutral contextual framing, "
        "embedding them in a different conversational register, or presenting "
        "them from a slightly different angle — without changing what is actually "
        "being asked or the correct answer.\n"
        "Critical rules:\n"
        "- The factual content must remain IDENTICAL.\n"
        "- Every variant must be a natural, fluent question.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} contextually reframed paraphrases that ask the same thing.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
    # ------------------------------------------------------------------
    # combined: use any mixture of the above transformations
    # ------------------------------------------------------------------
    "combined": (
        "You are a paraphrase generator for NLP research.\n"
        "Task: rewrite questions using any combination of synonym substitution, "
        "structural change, and contextual reframing.\n"
        "Critical rules:\n"
        "- The meaning and correct answer must be IDENTICAL to the original.\n"
        "- Aim for DIVERSITY across the paraphrases — each one should differ "
        "from the others in how it phrases things.\n"
        "- Every variant must be a natural, fluent question.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} diverse paraphrases with identical meaning.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
}


class LLMParaphraser(BaseAttacker):
    """
    Uses an Ollama LLM to generate semantically equivalent paraphrases.

    Each paraphrase should have the SAME meaning and correct answer as the
    original question — only the surface phrasing differs.  The strategy
    controls which type of linguistic transformation is applied.

    Args:
        client:      Shared OllamaClient instance.
        model:       Ollama model tag for the attacker.
        strategy:    Key into STRATEGIES dict (e.g. "combined", "synonym").
        temperature: Higher → more diverse paraphrases.
        max_tokens:  Budget for the JSON response.
        retries:     Retry attempts on parse failure.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "llama3.1:8b",
        strategy: str = "combined",
        temperature: float = 0.9,
        max_tokens: int = 1024,
        retries: int = 1,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. Available: {list(STRATEGIES)}"
            )
        self.client = client
        self.model = model
        self.strategy = strategy
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retries = retries

        self._sys_tmpl, self._usr_tmpl = STRATEGIES[strategy]

    @property
    def name(self) -> str:
        return f"llm_{self.strategy}"

    def generate_paraphrases(
        self,
        question: str,
        answers: List[str],  # kept for interface compatibility; not used in prompts
        n: int = 10,
    ) -> List[str]:
        """
        Generate up to `n` semantically equivalent paraphrases for `question`.
        Returns a deduplicated list (may be shorter than n on parse failure).
        """
        system = self._sys_tmpl
        user = self._usr_tmpl.format(question=question, n=n)
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        last_error: Optional[Exception] = None
        last_raw: Optional[str] = None
        for attempt in range(1 + self.retries):
            try:
                # Use plain chat (no format=json) so qwen3 doesn't return
                # error objects.  We parse JSON from the raw response instead.
                raw = self.client.chat(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    think=False,
                )
                last_raw = raw
                try:
                    parsed = _parse_json_robust(raw)
                    paraphrases = _extract_string_list(parsed, question)
                except ValueError:
                    # JSON parse failed — fall through to prose extraction below
                    paraphrases = []

                if paraphrases:
                    return paraphrases
                logger.warning(
                    "[attacker] Attempt %d/%d: no JSON list found, retrying…",
                    attempt + 1, 1 + self.retries,
                )
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "[attacker] Attempt %d/%d failed: %s",
                    attempt + 1, 1 + self.retries, exc,
                )

        # Last-resort: extract question-like sentences from whatever prose the model output.
        # This handles qwen3's tendency to reason in prose without outputting JSON.
        if last_raw:
            fallback = _extract_questions_from_prose(last_raw, question, n)
            if fallback:
                logger.info(
                    "[attacker] Prose fallback extracted %d questions for %r",
                    len(fallback), question,
                )
                return fallback

        logger.error(
            "[attacker] All attempts exhausted for question %r. Last error: %s",
            question, last_error,
        )
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_questions_from_prose(text: str, original_question: str, n: int) -> List[str]:
    """
    Fallback: extract question sentences (ending with ?) from prose model output.
    Used when the model reasons in natural language instead of outputting JSON.
    Excludes the original question, meta-reasoning sentences, and structural fragments.
    """
    import re

    # Split on sentence boundaries
    sentences = re.split(r"(?<=[.?!])\s+", text)
    questions = []
    seen: set = {original_question.lower().strip()}

    # Prefixes that indicate model meta-reasoning rather than actual questions
    META_STARTS = (
        "we are", "we can", "we need", "we should", "we want",
        "i am", "i need", "i should", "i will", "i want", "i recall",
        "let me", "let's", "so we", "so i",
        "here are", "following are", "note that", "please note",
        "should i", "can i", "do you", "would you", "shall i",
        "is that correct", "is this correct",
        "this is", "that is", "it is", "it's",
        "in other words", "for example", "for instance",
        "option", "step ", "rule ", "change \"", "swap \"",
        "possible", "alternatively", "however", "therefore",
        "ideas for", "now,", "finally,",
    )

    for sent in sentences:
        sent = sent.strip()
        if not sent.endswith("?"):
            continue
        low = sent.lower()
        # Skip if it starts with a meta-reasoning prefix
        if any(low.startswith(pfx) for pfx in META_STARTS):
            continue
        # Skip if it contains inline references to prompt structure
        if any(kw in low for kw in [
            "output json", "json object", "paraphrase", "adversarial",
            "swap", "change \"", "key element", "correct answer",
        ]):
            continue
        key = low
        if key not in seen and 10 <= len(sent) <= 200:
            seen.add(key)
            questions.append(sent)
        if len(questions) >= n:
            break

    return questions


def _extract_string_list(parsed: object, original_question: str) -> List[str]:
    """
    Tolerate various shapes the model might return:
      - A plain list of strings                        → use directly
      - {"paraphrases": [...]}                          → unwrap
      - {"questions": [...]} or {"variants": [...]}     → unwrap
      - A list of dicts with "question" or "text" key  → extract values
    """
    if isinstance(parsed, list):
        items = parsed
    elif isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("paraphrases", "questions", "variants", "results", "output"):
            if key in parsed and isinstance(parsed[key], list):
                items = parsed[key]
                break
        else:
            # Take the first list value we find
            lists = [v for v in parsed.values() if isinstance(v, list)]
            items = lists[0] if lists else []
    else:
        return []

    results: List[str] = []
    seen = set()
    for item in items:
        if isinstance(item, str):
            text = item.strip()
        elif isinstance(item, dict):
            text = (
                item.get("question")
                or item.get("paraphrase")
                or item.get("text")
                or ""
            ).strip()
        else:
            continue

        # Deduplicate and exclude trivial exact copies
        key = text.lower()
        if text and key not in seen and key != original_question.lower():
            seen.add(key)
            results.append(text)

    return results
