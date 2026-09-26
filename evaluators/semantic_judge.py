"""
Semantic Equivalence Judge.

Determines whether a paraphrase asks the SAME question as the original
(i.e. has identical meaning and the same correct answer), regardless of
how the words are arranged.

This is distinct from the answer judge (llm_judge.py), which checks whether
the victim's *answer* is correct.  Here we check whether the *question*
is semantically valid.
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from ollama_client import OllamaClient, normalise_text

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You are a semantic equivalence judge for questions.\n"
    "Task: decide whether a paraphrased question is semantically equivalent "
    "to the original — meaning both questions ask about the same thing and "
    "would have the same correct answer.\n"
    "\n"
    "Scoring rules:\n"
    "- Score 1: the paraphrase asks the SAME question. Minor rewording, synonym "
    "substitution, voice change, or clause reordering are fine.\n"
    "- Score 0 in ANY of these cases:\n"
    "  * The paraphrase changes what is being asked (different subject, time period, scope).\n"
    "  * The paraphrase adds a qualifier or false premise that alters the meaning.\n"
    "  * The paraphrase is circular or paradoxical.\n"
    "- A paraphrase that restates a true, well-known attribute of the subject is "
    "still equivalent, even if that attribute hints at the answer.\n"
    "- Ignore trivial formatting differences (capitalisation, punctuation).\n"
    "\n"
    'Output ONLY this JSON: {"score": 0 or 1, "rationale": "one sentence"}'
)

_USER_TMPL = (
    "/no_think\n"
    'Original question: "{original}"\n'
    'Paraphrase: "{paraphrase}"\n\n'
    "Are these semantically equivalent (same meaning, same correct answer)?\n"
    'Output ONLY: {{"score": 0 or 1, "rationale": "..."}}'
)

_SCORE_RE = re.compile(r'["\']?score["\']?\s*:\s*["\']?(-?\d+(?:\.\d+)?)', re.I)

# Constrained-decoding schema. qwen3 and friends ignore "output only JSON"
# and reason in prose until they hit the token limit, so the shape has to be
# enforced by the server rather than requested in the prompt.
_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "enum": [0, 1]},
        "rationale": {"type": "string"},
    },
    "required": ["score", "rationale"],
}


@dataclass
class SemanticResult:
    is_equivalent: bool
    score: float          # 0.0 – 1.0
    rationale: str


class SemanticEquivalenceJudge:
    """
    Uses an LLM to judge whether a paraphrase is semantically equivalent to
    the original question.

    Args:
        client:      Shared OllamaClient.
        model:       Ollama model tag for the judge.
        temperature: 0 for determinism.
        threshold:   Minimum score to consider a paraphrase valid (default 0.7).
                     Use 0.5 for binary yes/no models.
    """

    def __init__(
        self,
        client: OllamaClient,
        model: str = "llama3.1:8b",
        temperature: float = 0.0,
        threshold: float = 0.7,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.threshold = threshold

    def judge(self, paraphrase: str, original: str, ground_truth_answers=None) -> SemanticResult:
        """
        Returns a SemanticResult indicating whether the paraphrase is
        semantically equivalent to the original question.
        """
        # Fast path: trivially identical after normalisation
        if normalise_text(paraphrase) == normalise_text(original):
            return SemanticResult(is_equivalent=True, score=1.0,
                                  rationale="Identical after normalisation.")

        # Hard filter: the attack is only interesting if the paraphrase is
        # clean text.  Typos and invisible characters are a different (and
        # already well-studied) attack surface, and the mutation operator
        # produces mangled tokens like "Machu Pic: Picchu" that inflate the
        # victim's deliberation without saying anything about paraphrasing.
        malformed = _check_malformed(paraphrase, original)
        if malformed:
            return SemanticResult(
                is_equivalent=False,
                score=0.0,
                rationale=f"Malformed text: {malformed}",
            )

        # Hard filter: dangling anaphora check.
        # Paraphrases that use "aforementioned", "said X", "the subject in question"
        # without establishing a referent are incomplete as standalone questions.
        dangling = _check_dangling_reference(paraphrase, original)
        if dangling:
            return SemanticResult(
                is_equivalent=False,
                score=0.0,
                rationale=f"Dangling reference: '{dangling}' has no referent in a standalone question.",
            )


        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _USER_TMPL.format(
                original=original, paraphrase=paraphrase
            )},
        ]

        try:
            parsed = self.client.chat_json(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=512,
                think=False,
                schema=_SCHEMA,
            )
        except Exception as exc:
            logger.warning("[sem_judge] LLM call failed: %s", exc)
            return SemanticResult(is_equivalent=False, score=0.0,
                                  rationale=f"Judge error: {exc}")

        score, rationale = _parse_score_and_rationale(parsed)
        if score is None:
            logger.warning("[sem_judge] Could not parse score from: %r", parsed)
            return SemanticResult(is_equivalent=False, score=0.0,
                                  rationale="Parse error.")

        is_equiv = score >= self.threshold
        return SemanticResult(is_equivalent=is_equiv, score=score,
                              rationale=rationale)


# ---------------------------------------------------------------------------
# Well-formedness detection
# ---------------------------------------------------------------------------

# Zero-width and bidirectional formatting characters.
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u2064\ufeff\xad]")

# A colon or semicolon glued to the inside of a word, e.g. "Pic:u".
_INTERNAL_PUNCT = re.compile(r"\b\w+[:;]\w+\b")


def _check_malformed(paraphrase: str, original: str) -> Optional[str]:
    """
    Return a description of the defect if the paraphrase is not clean text.

    Catches the three ways the mutation operator corrupts a question:
    invisible characters, punctuation glued inside a word, and proper nouns
    truncated to a prefix ("Machu Pic" for "Machu Picchu").
    """
    if _INVISIBLE.search(paraphrase):
        return "contains invisible characters"

    m = _INTERNAL_PUNCT.search(paraphrase)
    if m:
        return f"punctuation inside the word {m.group(0)!r}"

    # Proper nouns from the original must survive intact, not as fragments.
    # A fragment counts as a truncation only if at least two characters were
    # dropped, so that plurals like "Inca"/"Incas" are left alone.
    original_nouns = set(re.findall(r"\b[A-Z][a-zA-Z]{3,}\b", original))
    para_nouns = set(re.findall(r"\b[A-Z][a-zA-Z]{2,}\b", paraphrase))
    for frag in para_nouns - original_nouns:
        for noun in original_nouns:
            if noun.startswith(frag) and len(noun) - len(frag) >= 2:
                return f"{noun!r} truncated to {frag!r}"

    return None


# ---------------------------------------------------------------------------
# Dangling reference detection
# ---------------------------------------------------------------------------

# Phrases that presuppose prior context that doesn't exist in a standalone question
_DANGLING_PHRASES = [
    "aforementioned",
    "the subject in question",
    "said structure",
    "said site",
    "said work",
    "said artifact",
    "said building",
    "said city",
    "the above",
    "the above-mentioned",
    "the previously mentioned",
    "the previously stated",
]


def _check_dangling_reference(paraphrase: str, original: str) -> str | None:
    """
    Returns the dangling phrase if the paraphrase contains a reference that
    presupposes prior context (e.g. 'aforementioned') but the specific subject
    from the original question is NOT also mentioned in the paraphrase.

    If the paraphrase says "the aforementioned Mona Lisa" — that's fine (explicit).
    If it says just "the aforementioned work" — that's dangling (no referent).
    """
    para_lower = paraphrase.lower()

    for phrase in _DANGLING_PHRASES:
        if phrase not in para_lower:
            continue

        # Extract key nouns from the original question (proper nouns / quoted names)
        # as a simple heuristic: words longer than 4 chars that are capitalised
        original_keywords = {
            w.lower() for w in re.findall(r'\b[A-Z][a-zA-Z]{3,}\b', original)
            if w.lower() not in {"which", "what", "where", "when", "who", "whom",
                                  "whose", "that", "this", "these", "those",
                                  "ancient", "modern", "famous", "great"}
        }

        # If none of the original's key nouns appear in the paraphrase,
        # the reference is dangling
        if original_keywords and not any(kw in para_lower for kw in original_keywords):
            return phrase

    return None


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_score_and_rationale(parsed: object) -> Tuple[Optional[float], str]:
    """Extract score and rationale from the judge's decoded output."""
    score: Optional[float] = None
    rationale = ""

    if isinstance(parsed, dict):
        for k, v in parsed.items():
            if k.lower() == "score":
                try:
                    score = float(v)
                except (TypeError, ValueError):
                    pass
        rationale = (parsed.get("rationale") or parsed.get("reason") or "")

    # Anything else came back as prose; fall through to the text scrapers.
    raw = parsed if isinstance(parsed, str) else str(parsed)

    # Regex fallback
    if score is None:
        m = _SCORE_RE.search(raw)
        if m:
            try:
                score = float(m.group(1))
            except ValueError:
                pass

    # Last resort: look for explicit "score: 1" / "score: 0"
    if score is None:
        low = raw.lower()
        if "score: 1" in low or '"score": 1' in low:
            score = 1.0
        elif "score: 0" in low or '"score": 0' in low:
            score = 0.0

    if not rationale:
        rationale = raw[:120]

    return score, rationale
