#!/usr/bin/env python3
"""
Round 3: isolate COMPOSITION FAILURE as an attack mechanism.

Motivation
----------
Round 2 found that qwen3:4b answers both of these correctly:

    "Which peace settlement was concluded at Osnabruck and Munster?"  -> Westphalia
    "In which year was the Peace of Westphalia signed?"               -> 1648

but gets the single composed question confidently wrong:

    "The treaties concluded at Osnabruck and Munster were signed in which year?"
                                                    -> "1544 ... 1545"

Both halves of the knowledge are present; the model simply cannot traverse
them in one hop.  The composed question is semantically equivalent to the
direct one, so this is a pure framing attack with no knowledge gap.

Experimental design
-------------------
For each chain A -> B -> C we ask three questions:

    bridge    A -> B    does the model know the attribute identifies the entity?
    direct    B -> C    does the model know the fact about the entity?
    composed  A -> C    the semantically equivalent indirect phrasing

A result is a CLEAN COMPOSITION FAILURE when bridge and direct both pass
and composed fails.  Those are the attacks that cannot be explained away
as missing knowledge.

Usage:
    python probe3.py --model qwen3:4b --url http://127.0.0.1:11435
"""
import argparse
import json
import os

import requests

from victim import _SYSTEM_PROMPT, _extract_short_answer

# Each chain: attribute A identifies entity B; the target fact about B is C.
CHAINS = [
    {
        "name": "westphalia",
        "bridge":   ("Which peace settlement was concluded at Osnabruck and Munster?", ["westphalia"]),
        "direct":   ("In which year was the Peace of Westphalia signed?", ["1648"]),
        "composed": ("The treaties concluded at Osnabruck and Munster were signed in which year?", ["1648"]),
    },
    {
        "name": "versailles",
        "bridge":   ("Which treaty was signed in the Hall of Mirrors at the Palace of Versailles?", ["versailles"]),
        "direct":   ("In which year was the Treaty of Versailles signed?", ["1919"]),
        "composed": ("The treaty signed in the Hall of Mirrors at the Palace of Versailles was concluded in which year?", ["1919"]),
    },
    {
        "name": "appomattox",
        "bridge":   ("Which war ended with the surrender at Appomattox Court House?", ["civil war"]),
        "direct":   ("In which year did the American Civil War end?", ["1865"]),
        "composed": ("The war that ended with the surrender at Appomattox Court House concluded in which year?", ["1865"]),
    },
    {
        "name": "penicillium",
        "bridge":   ("Which antibiotic was derived from the mould Penicillium notatum?", ["penicillin"]),
        "direct":   ("Who discovered penicillin?", ["fleming"]),
        "composed": ("Who discovered the antibiotic derived from the mould Penicillium notatum?", ["fleming"]),
    },
    {
        "name": "aurum",
        "bridge":   ("Which element has the Latin name aurum?", ["gold"]),
        "direct":   ("What is the chemical symbol for gold?", ["au"]),
        "composed": ("What is the chemical symbol of the element whose Latin name is aurum?", ["au"]),
    },
    {
        "name": "sistine",
        "bridge":   ("Which artist painted the ceiling of the Sistine Chapel?", ["michelangelo"]),
        "direct":   ("Which marble sculpture of a biblical king did Michelangelo carve in Florence?", ["david"]),
        "composed": ("Which marble sculpture of a biblical king was carved by the artist who painted the Sistine Chapel ceiling?", ["david"]),
    },
    {
        "name": "titanic",
        "bridge":   ("Which ocean liner sank on its maiden voyage in 1912?", ["titanic"]),
        "direct":   ("Which shipping line operated the Titanic?", ["white star"]),
        "composed": ("Which shipping line operated the ocean liner that sank on its maiden voyage in 1912?", ["white star"]),
    },
    {
        "name": "bingham",
        "bridge":   ("Which archaeological site did Hiram Bingham bring to world attention in 1911?", ["machu"]),
        "direct":   ("Which civilization built Machu Picchu?", ["inca"]),
        "composed": ("Which civilization built the site that Hiram Bingham brought to world attention in 1911?", ["inca"]),
    },
    {
        "name": "gettysburg",
        "bridge":   ("Who delivered the Gettysburg Address?", ["lincoln"]),
        "direct":   ("In which year was Abraham Lincoln assassinated?", ["1865"]),
        "composed": ("In which year was the man who delivered the Gettysburg Address assassinated?", ["1865"]),
    },
    {
        "name": "beagle",
        "bridge":   ("Who wrote 'On the Origin of Species'?", ["darwin"]),
        "direct":   ("On which ship did Charles Darwin make his famous voyage?", ["beagle"]),
        "composed": ("On which ship did the author of 'On the Origin of Species' make his famous voyage?", ["beagle"]),
    },
    {
        "name": "hastings",
        "bridge":   ("Which battle is depicted on the Bayeux Tapestry?", ["hastings"]),
        "direct":   ("In which year did the Battle of Hastings take place?", ["1066"]),
        "composed": ("The battle depicted on the Bayeux Tapestry took place in which year?", ["1066"]),
    },
    {
        "name": "runnymede",
        "bridge":   ("Which charter was sealed at Runnymede?", ["magna carta", "magna"]),
        "direct":   ("In which year was the Magna Carta sealed?", ["1215"]),
        "composed": ("The charter sealed at Runnymede was issued in which year?", ["1215"]),
    },
    {
        "name": "ghent",
        "bridge":   ("Which treaty was signed in Ghent in December 1814?", ["ghent"]),
        "direct":   ("Which war was ended by the Treaty of Ghent?", ["1812"]),
        "composed": ("Which war was ended by the treaty signed in Ghent in December 1814?", ["1812"]),
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
    p.add_argument("--url",   default="http://127.0.0.1:11435")
    p.add_argument("--max-tokens", type=int, default=1536)
    p.add_argument("--out",   default="results/probe3_composition.json")
    args = p.parse_args()

    records, clean_failures = [], []

    for chain in CHAINS:
        print(f"\n{'='*74}\nCHAIN: {chain['name']}\n{'='*74}")
        row = {"name": chain["name"]}

        for role in ("bridge", "direct", "composed"):
            q, accepted = chain[role]
            ans, raw = ask(args.url, args.model, q, args.max_tokens)
            ok = any(a in ans.lower() for a in accepted)
            row[role] = {"question": q, "answer": ans, "correct": ok,
                         "raw_len": len(raw), "raw": raw}
            print(f"  {role:9} {'OK ' if ok else 'BAD'}  {q}")
            print(f"            -> {ans!r}  ({len(raw)} chars)")

        clean = (row["bridge"]["correct"] and row["direct"]["correct"]
                 and not row["composed"]["correct"])
        row["clean_composition_failure"] = clean
        records.append(row)
        if clean:
            clean_failures.append(row)
            print("  *** CLEAN COMPOSITION FAILURE ***")

    print(f"\n\n{'='*74}")
    print(f"CLEAN COMPOSITION FAILURES: {len(clean_failures)} / {len(CHAINS)} chains")
    print("=" * 74)
    for r in clean_failures:
        print(f"\n  [{r['name']}]")
        print(f"    knows  : {r['bridge']['question']}")
        print(f"             -> {r['bridge']['answer']!r}")
        print(f"    knows  : {r['direct']['question']}")
        print(f"             -> {r['direct']['answer']!r}")
        print(f"    FAILS  : {r['composed']['question']}")
        print(f"             -> {r['composed']['answer']!r}")

    # Reasoning-length amplification: composed vs direct
    print(f"\n{'='*74}\nREASONING LENGTH (chars): direct vs composed\n{'='*74}")
    for r in records:
        d, c = r["direct"]["raw_len"], r["composed"]["raw_len"]
        ratio = c / d if d else 0
        flag = "  <-- failure" if r["clean_composition_failure"] else ""
        print(f"  {r['name']:12} direct={d:5}  composed={c:5}  ratio={ratio:4.2f}x{flag}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records, "clean_failures": clean_failures}, f,
                  indent=2, ensure_ascii=False)
    print(f"\nSaved -> {args.out}")


if __name__ == "__main__":
    main()
