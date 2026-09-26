#!/usr/bin/env python3
"""
Batch probe tool for manual attack investigation.

Unlike run_experiment.py (which generates paraphrases with an LLM), this runs
a hand-written list of paraphrases against the victim so we can test specific
hypotheses about what kinds of rephrasing cause failures.

Usage:
    python probe.py                                  # run all probe sets
    python probe.py --set definite_description       # run one set
    python probe.py --model qwen3:4b --url http://127.0.0.1:11435
"""
import argparse
import json
import os

from ollama_client import OllamaClient
from victim import VictimModel

# ---------------------------------------------------------------------------
# Probe sets.
#
# Each entry:  original question, accepted answers, and paraphrases grouped by
# the hypothesis they test.  Every paraphrase must be SEMANTICALLY EQUIVALENT
# to the original — same meaning, same correct answer.
# ---------------------------------------------------------------------------

PROBE_SETS = {
    # -------------------------------------------------------------------
    # H1: Definite descriptions.  Refer to the entity by a uniquely
    # identifying description instead of its name.  Semantically identical
    # (the description picks out exactly one thing) but forces an extra
    # retrieval hop.
    # -------------------------------------------------------------------
    "definite_description": [
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "Who painted the portrait known as La Gioconda?",
                "Who painted the portrait of Lisa Gherardini that hangs in the Louvre?",
                "Which artist created the early-16th-century Florentine portrait famous for its sitter's ambiguous smile?",
                "The most visited painting in the Louvre was created by which artist?",
            ],
        },
        {
            "original": "Which ancient civilization built Machu Picchu?",
            "answers": ["inca", "incas", "incan"],
            "paraphrases": [
                "Which civilization built the 15th-century citadel perched above the Urubamba valley in Peru?",
                "The stone citadel in the Peruvian Andes rediscovered by Hiram Bingham in 1911 was built by which civilization?",
                "Which pre-Columbian civilization constructed the mountain settlement widely regarded as Peru's most famous archaeological site?",
            ],
        },
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "In which year did the Great War begin?",
                "In which year did the global conflict that ended in 1918 begin?",
                "The war triggered by the assassination of Archduke Franz Ferdinand began in which year?",
            ],
        },
        {
            "original": "What is the chemical symbol for gold?",
            "answers": ["au"],
            "paraphrases": [
                "What is the chemical symbol for the element with atomic number 79?",
                "Which symbol on the periodic table denotes the metal used as the traditional standard for currency reserves?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # H2: Answer-type reframing.  Ask for the same fact but phrase the
    # question so the expected answer TYPE is stated unusually.
    # -------------------------------------------------------------------
    "answer_type_reframe": [
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "What four-digit number corresponds to the year World War I began?",
                "World War I began in which year of the Gregorian calendar?",
                "State the calendar year in which World War I commenced.",
                "World War I's year of commencement was which?",
            ],
        },
        {
            "original": "Who was the first person to walk on the Moon?",
            "answers": ["armstrong", "neil"],
            "paraphrases": [
                "What is the full name of the first person to walk on the Moon?",
                "During Apollo 11, which crew member was first to set foot on the lunar surface?",
                "The first human to place a foot on the Moon's surface was whom?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # H3: Distractor loading.  Mention other true, related entities in the
    # question.  Meaning is unchanged but nearby facts may pull the model
    # toward the wrong retrieval.
    # -------------------------------------------------------------------
    "distractor_loaded": [
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "World War II began in 1939 and ended in 1945; in which year did World War I begin?",
                "Of the two world wars, the first one began in which year?",
                "Setting aside the 1939 conflict, in which year did the earlier world war begin?",
            ],
        },
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "Michelangelo sculpted David; who painted the Mona Lisa?",
                "Among Renaissance masters such as Raphael, Michelangelo, and Botticelli, who painted the Mona Lisa?",
                "The Mona Lisa hangs near works by Titian and Caravaggio in the Louvre; who painted it?",
            ],
        },
        {
            "original": "Who discovered penicillin?",
            "answers": ["fleming", "alexander"],
            "paraphrases": [
                "Florey and Chain later developed it into a usable drug, but who discovered penicillin?",
                "Who discovered penicillin, the antibiotic derived from Penicillium mould?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # H4: Deep nominalisation and clause embedding.  Pure syntactic load
    # with all factual anchors intact.
    # -------------------------------------------------------------------
    "syntactic_load": [
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "What is the year of commencement of the armed conflict designated as World War I?",
                "The date of the initiation of hostilities constituting World War I falls within which year?",
                "With respect to World War I, the year in which its commencement occurred is which?",
            ],
        },
        {
            "original": "Which ancient civilization built Machu Picchu?",
            "answers": ["inca", "incas", "incan"],
            "paraphrases": [
                "What is the identity of the civilization to which the construction of Machu Picchu is attributed?",
                "The attribution of the construction of Machu Picchu belongs to which ancient civilization?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # H5: Context stripping.  Replace the named entity with an anaphoric
    # reference that has no antecedent.  (Expected to succeed often, but
    # arguably NOT semantically equivalent — included as a control.)
    # -------------------------------------------------------------------
    "context_strip_control": [
        {
            "original": "Which ancient civilization built Machu Picchu?",
            "answers": ["inca", "incas", "incan"],
            "paraphrases": [
                "Which ancient civilization built the aforementioned archaeological site?",
                "Regarding the site in question, which ancient civilization constructed it?",
            ],
        },
    ],
}


def check(answer: str, accepted) -> bool:
    """Loose containment check against accepted answer fragments."""
    low = answer.lower()
    return any(a in low for a in accepted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3:4b")
    parser.add_argument("--url",   default="http://127.0.0.1:11435")
    parser.add_argument("--set",   default=None, help="Run only this probe set")
    parser.add_argument("--out",   default="results/probe_results.json")
    args = parser.parse_args()

    client = OllamaClient(base_url=args.url)
    victim = VictimModel(client, args.model, temperature=0.0)

    sets_to_run = {args.set: PROBE_SETS[args.set]} if args.set else PROBE_SETS

    all_records = []
    hits = []

    for set_name, groups in sets_to_run.items():
        print(f"\n{'='*70}")
        print(f"SET: {set_name}")
        print("=" * 70)

        for group in groups:
            original = group["original"]
            accepted = group["answers"]

            baseline = victim.answer(original)
            base_ok  = check(baseline, accepted)
            print(f"\n  ORIGINAL: {original}")
            print(f"    baseline → {baseline!r}  {'OK' if base_ok else 'BASELINE FAILED'}")

            for para in group["paraphrases"]:
                ans = victim.answer(para)
                ok  = check(ans, accepted)
                rec = {
                    "set": set_name,
                    "original": original,
                    "paraphrase": para,
                    "answer": ans,
                    "correct": ok,
                    "baseline_answer": baseline,
                    "baseline_correct": base_ok,
                }
                all_records.append(rec)
                if not ok and base_ok:
                    hits.append(rec)
                marker = "   ok  " if ok else " >>HIT "
                print(f"    [{marker}] {para}")
                print(f"              → {ans!r}")

    print(f"\n\n{'='*70}")
    print(f"SUMMARY: {len(hits)} hits out of {len(all_records)} probes")
    print("=" * 70)
    for h in hits:
        print(f"\n  [{h['set']}]  {h['original']}")
        print(f"    paraphrase: {h['paraphrase']}")
        print(f"    answer    : {h['answer']!r}   (expected ~ {h['baseline_answer']!r})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": all_records, "hits": hits}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
