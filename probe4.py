#!/usr/bin/env python3
"""
More host-city / signing-place composition chains.

The working attack is: drop the canonical name, identify the subject only
by a uniquely identifying place or object, keep the original answer type.
"""
import argparse
import json
import os

import requests

from victim import _SYSTEM_PROMPT, _extract_short_answer

CHAINS = [
    {
        "name": "westphalia_cities",
        "bridge":   ("Which peace settlement was concluded at Osnabruck and Munster?", ["westphalia"]),
        "direct":   ("In which year was the Peace of Westphalia signed?", ["1648"]),
        "composed": ("The treaties concluded at Osnabruck and Munster were signed in which year?", ["1648"]),
    },
    {
        "name": "ghent_war",
        "bridge":   ("Which treaty was signed in Ghent in December 1814?", ["ghent"]),
        "direct":   ("Which war was ended by the Treaty of Ghent?", ["1812"]),
        "composed": ("Which war was ended by the treaty signed in Ghent in December 1814?", ["1812"]),
    },
    {
        "name": "paris_1783",
        "bridge":   ("Which treaty was signed in Paris in 1783 ending a colonial war?", ["paris"]),
        "direct":   ("Which war was ended by the Treaty of Paris in 1783?", ["revolution", "independence", "american"]),
        "composed": ("Which war was ended by the treaty signed in Paris in 1783?", ["revolution", "independence", "american"]),
    },
    {
        "name": "utrecht",
        "bridge":   ("Which peace was signed at Utrecht in 1713?", ["utrecht"]),
        "direct":   ("Which war was ended by the Peace of Utrecht?", ["spanish succession"]),
        "composed": ("Which war was ended by the peace signed at Utrecht in 1713?", ["spanish succession"]),
    },
    {
        "name": "vienna_congress",
        "bridge":   ("Which diplomatic congress met in Vienna after Napoleon's first abdication?", ["vienna"]),
        "direct":   ("In which year did the Congress of Vienna conclude?", ["1815"]),
        "composed": ("The diplomatic congress that met in Vienna after Napoleon's first abdication concluded in which year?", ["1815"]),
    },
    {
        "name": "appomattox_year",
        "bridge":   ("Which war ended with the surrender at Appomattox Court House?", ["civil war"]),
        "direct":   ("In which year did the American Civil War end?", ["1865"]),
        "composed": ("The war that ended with the surrender at Appomattox Court House concluded in which year?", ["1865"]),
    },
    {
        "name": "hall_of_mirrors",
        "bridge":   ("Which treaty was signed in the Hall of Mirrors?", ["versailles"]),
        "direct":   ("In which year was the Treaty of Versailles signed?", ["1919"]),
        "composed": ("The treaty signed in the Hall of Mirrors was concluded in which year?", ["1919"]),
    },
    {
        "name": "bayeux",
        "bridge":   ("Which battle is depicted on the Bayeux Tapestry?", ["hastings"]),
        "direct":   ("In which year did the Battle of Hastings take place?", ["1066"]),
        "composed": ("The battle depicted on the Bayeux Tapestry took place in which year?", ["1066"]),
    },
]


def ask(url, model, question, max_tokens):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": question},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.0, "num_predict": max_tokens},
    }
    data = requests.post(f"{url}/api/chat", json=payload, timeout=600).json()
    raw = data.get("message", {}).get("content", "").strip()
    return _extract_short_answer(raw), raw


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--url", default="http://127.0.0.1:11435")
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--out", default="results/probe4_hostcity.json")
    args = p.parse_args()

    records, clean_failures = [], []
    for chain in CHAINS:
        print(f"\n{'='*74}\nCHAIN: {chain['name']}\n{'='*74}")
        row = {"name": chain["name"]}
        for role in ("bridge", "direct", "composed"):
            q, accepted = chain[role]
            ans, raw = ask(args.url, args.model, q, args.max_tokens)
            ok = any(a in ans.lower() for a in accepted)
            row[role] = {"question": q, "answer": ans, "correct": ok, "raw_len": len(raw)}
            print(f"  {role:9} {'OK ' if ok else 'BAD'}  {q}")
            print(f"            -> {ans!r}")
        clean = (row["bridge"]["correct"] and row["direct"]["correct"]
                 and not row["composed"]["correct"])
        row["clean_composition_failure"] = clean
        records.append(row)
        if clean:
            clean_failures.append(row)
            print("  *** CLEAN COMPOSITION FAILURE ***")

    print(f"\nCLEAN COMPOSITION FAILURES: {len(clean_failures)} / {len(CHAINS)}")
    for r in clean_failures:
        print(f"  [{r['name']}] {r['composed']['question']!r} -> {r['composed']['answer']!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records, "clean_failures": [
            {"name": r["name"], "composed": r["composed"]} for r in clean_failures
        ]}, f, indent=2, ensure_ascii=False)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
