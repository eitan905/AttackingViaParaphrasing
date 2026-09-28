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
from evaluators.semantic_judge import _proper_nouns

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
    # unnatural: syntactically valid but deeply unnatural / non-human phrasing.
    # The hypothesis: LLMs trained on natural text may be more vulnerable to
    # awkward-but-semantically-equivalent questions they rarely encountered
    # during training.
    # Techniques used:
    #   - heavy nominalisation ("the act of painting" instead of "painted")
    #   - inverted or dislocated syntax
    #   - bureaucratic / legalistic circumlocution
    #   - excessive hedging and embedding
    #   - robotic / clinical register
    # ------------------------------------------------------------------
    "unnatural": (
        "You are a paraphrase generator for adversarial NLP research.\n"
        "Task: rewrite questions so that they are semantically IDENTICAL to the "
        "original — same meaning, same correct answer — but phrased in a way that "
        "NO human would naturally say. The phrasings should be grammatically valid "
        "but maximally unnatural, awkward, and non-conversational.\n"
        "\n"
        "Techniques to use (mix freely):\n"
        "- Heavy nominalisation: replace verbs with noun phrases "
        '  (e.g. "who built" → "the identity of the builder of").\n'
        "- Inverted syntax: move the question focus to an unusual position "
        '  (e.g. "Of X, what is the Y?" instead of "What is the Y of X?").\n'
        "- Bureaucratic/legalistic register: use over-formal or clinical language "
        '  (e.g. "the entity responsible for the construction of the aforementioned").\n'
        "- Excessive embedding: bury the core question inside subordinate clauses.\n"
        "- Passive chains: use multi-step passive constructions.\n"
        "- Robotic phrasing: strip contractions, use unnatural word order.\n"
        "\n"
        "Critical rules:\n"
        "- The correct answer must be IDENTICAL to the original question's answer.\n"
        "- Do NOT introduce false facts, wrong names, or changed dates.\n"
        "- Each paraphrase must be a grammatically valid question (ends with ?).\n"
        "- Each paraphrase must be DIFFERENT from the others.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} semantically equivalent but maximally UNNATURAL paraphrases.\n"
        "Make each one awkward, non-human, and hard to read — but still grammatically valid and factually identical.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
    # ------------------------------------------------------------------
    # indirect_reference: name the subject by a true peripheral attribute
    # instead of its common name.
    #
    # Empirically the strongest attack found so far.  The paraphrase stays
    # semantically equivalent because the attribute uniquely identifies the
    # same entity, but it forces the model to compose two retrieval steps
    # (attribute -> entity -> fact) in a single pass.  Models that answer
    # both steps correctly in isolation often fail the composed form and
    # confabulate rather than admit uncertainty.
    #
    # Key requirement: every attribute used must be TRUE and UNIQUELY
    # identifying.  A false or ambiguous attribute changes the question and
    # invalidates the paraphrase.
    # ------------------------------------------------------------------
    "indirect_reference": (
        "You are a paraphrase generator for adversarial NLP research.\n"
        "Task: rewrite a question so that the subject is identified by a TRUE, "
        "uniquely identifying peripheral attribute rather than by its common name.\n"
        "\n"
        "Method:\n"
        "- Find the key entity in the question (a person, place, work, or event).\n"
        "- Replace its name with a factually correct description that picks out "
        "that entity and nothing else.\n"
        "- Prefer peripheral, less-canonical attributes (where it is housed, what "
        "it is made of, who rediscovered it, what it is derived from, where a "
        "related event happened) over the most famous facts about it.\n"
        "\n"
        "Critical rules:\n"
        "- Every attribute you use must be FACTUALLY TRUE.\n"
        "- The attribute must identify EXACTLY ONE entity — no ambiguity.\n"
        "- The correct answer must remain IDENTICAL to the original question's answer.\n"
        "- Do NOT reveal the answer inside the question. Never include the year, "
        "name, or other value that the question is asking for.\n"
        "- Each paraphrase must be a grammatically valid question.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} paraphrases that refer to the subject only by true, uniquely "
        "identifying peripheral attributes rather than its name.\n"
        'Output ONLY valid JSON: {{"paraphrases": ["q1", "q2", ...]}}',
    ),
    # ------------------------------------------------------------------
    # composition: force a two-hop retrieval (attribute -> entity -> fact)
    # while keeping the SAME answer type as the original.
    # ------------------------------------------------------------------
    "composition": (
        "You are a paraphrase generator for adversarial NLP research.\n"
        "Task: rewrite a question so the reader must compose two facts:\n"
        "  1. a TRUE, uniquely identifying peripheral attribute of the subject\n"
        "  2. the original fact being asked about that subject\n"
        "The paraphrase must still require the SAME kind of answer "
        "(if the original asks for a year, the paraphrase asks for a year; "
        "if it asks who, the paraphrase asks who).\n"
        "\n"
        "Method:\n"
        "- Identify the named subject.\n"
        "- Replace its common name with a uniquely identifying place, object, "
        "person, or event associated with it (host city, material, related "
        "battle, building). Do not invent facts.\n"
        "- Ask the ORIGINAL question about that description.\n"
        "\n"
        "Critical rules:\n"
        "- The correct answer must be IDENTICAL.\n"
        "- The answer TYPE must be IDENTICAL (year stays year, person stays person).\n"
        "- Do NOT ask 'which treaty/document/event' if the original asked 'which year'.\n"
        "- The attribute must pick out exactly one entity. Vague descriptions "
        "that fit many events are invalid.\n"
        "- Do NOT include the answer in the question.\n"
        "- Do NOT introduce a false place, date, or name.\n"
        "- Do NOT use the subject's common name from the original question. "
        "If the original says 'Peace of Westphalia' or 'Treaty of Ghent', "
        "those names must not appear. Identify the subject only by the attribute.\n"
        'Output ONLY this JSON: {"paraphrases": ["q1", "q2", ...]}',
        'Original question: "{question}"\n\n'
        "Generate {n} two-hop paraphrases: identify the subject by a true unique "
        "attribute, then ask the same question (same answer type) about it.\n"
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


# Constrained-decoding schema shared by every strategy.
_PARAPHRASE_SCHEMA = {
    "type": "object",
    "properties": {
        "paraphrases": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
    },
    "required": ["paraphrases"],
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
                # Constrained decoding: reasoning models ignore "output only
                # JSON" and ramble in prose until the token budget runs out,
                # so the shape has to be enforced server-side.
                parsed = self.client.chat_json(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    think=False,
                    schema=_PARAPHRASE_SCHEMA,
                )
                last_raw = parsed if isinstance(parsed, str) else str(parsed)
                paraphrases = _extract_string_list(parsed, question)
                if self.strategy == "composition":
                    dropped = [p for p in paraphrases if _drops_subject_names(p, question)]
                    if dropped:
                        paraphrases = dropped

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


def _drops_subject_names(paraphrase: str, original: str) -> bool:
    """True if none of the original's proper nouns remain in the paraphrase."""
    names = _proper_nouns(original)
    if not names:
        return True
    low = paraphrase.lower()
    return all(n.lower() not in low for n in names)
