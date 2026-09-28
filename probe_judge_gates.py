#!/usr/bin/env python3
"""Check the semantic judge against known good and known-bad paraphrases."""

import argparse
from ollama_client import OllamaClient
from evaluators.semantic_judge import SemanticEquivalenceJudge

CASES = [
    # (label, original, paraphrase, should_accept)
    ("good_westphalia_compose",
     "In which year was the Peace of Westphalia signed?",
     "The treaties concluded at Osnabruck and Munster were signed in which year?",
     True),
    ("bad_nuremberg",
     "In which year was the Peace of Westphalia signed?",
     "In which year did the treaty in Nuremberg that ended the era of imperial rule occur?",
     False),
    ("bad_vague_empire",
     "In which year was the Peace of Westphalia signed?",
     "When did a treaty marking the decline of a European imperial system become effective?",
     False),
    ("bad_hre_end",
     "In which year was the Peace of Westphalia signed?",
     "In which year did the treaty marking the end of the Holy Roman Empire's supremacy take place?",
     False),
    ("bad_type_versailles",
     "In which year was the Treaty of Versailles signed?",
     "Which document signed in the year marking the end of a global conflict mandated a defeated nation to pay reparations?",
     False),
    ("bad_type_civil",
     "In which year did the American Civil War end?",
     "In what epochal event did the United States shed its fratricidal bloodshed?",
     False),
    ("good_appomattox",
     "In which year did the American Civil War end?",
     "The war that ended with the surrender at Appomattox Court House concluded in which year?",
     True),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:11435")
    p.add_argument("--model", default="qwen3:4b")
    args = p.parse_args()

    client = OllamaClient(base_url=args.url)
    client.require_available()
    judge = SemanticEquivalenceJudge(client, model=args.model, threshold=0.7)

    ok = 0
    for label, original, paraphrase, should_accept in CASES:
        result = judge.judge(paraphrase, original)
        match = result.is_equivalent == should_accept
        ok += int(match)
        mark = "OK" if match else "FAIL"
        print(f"[{mark}] {label}  accept={result.is_equivalent} "
              f"(want {should_accept})  score={result.score:.2f}")
        print(f"      {result.rationale}")
        print(f"      {paraphrase!r}")
    print(f"\n{ok}/{len(CASES)} cases behaved as expected")
    return 0 if ok == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
