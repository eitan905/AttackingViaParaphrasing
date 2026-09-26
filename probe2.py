#!/usr/bin/env python3
"""
Round 2 probes: chase the "induced overthinking" mechanism found in round 1.

Round 1 finding: a definite description vague enough to require multi-hop
inference sent qwen3 into a self-doubt loop that never converged, even though
it answers the direct form of the question instantly.

This round:
  A. Controls   — is that a real failure or just a token-budget artifact?
  B. Riddle     — scale up indirect definite descriptions.
  C. Multi-hop  — descriptions needing two or more inference steps.
  D. Archaic    — low-frequency lexicon, same meaning.
  E. Meta       — wrap the question in a frame.
  F. Weak-fact  — same techniques on facts the model knows less confidently.

Usage:
    python probe2.py --model qwen3:4b --url http://127.0.0.1:11435
    python probe2.py --set riddle --max-tokens 2048
"""
import argparse
import json
import os

import requests

from victim import _SYSTEM_PROMPT, _extract_short_answer

PROBE_SETS = {
    # -------------------------------------------------------------------
    # A. Token-budget control.  The round-1 winner, plus its direct form.
    #    If the "attack" disappears at a larger budget it was an artifact.
    # -------------------------------------------------------------------
    "budget_control": [
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "Who painted the Mona Lisa?",
                "Which artist created the early-16th-century Florentine portrait famous for its sitter's ambiguous smile?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # B. Riddle descriptions.  Identify the entity only by indirect
    #    properties.  Still uniquely referring, so meaning is preserved.
    # -------------------------------------------------------------------
    "riddle": [
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "Which artist painted the woman whose enigmatic expression art historians have debated for five centuries?",
                "Who painted the Renaissance portrait whose 1911 theft from a Paris museum made it globally famous?",
                "Which artist produced the panel painting on poplar wood that now sits behind bulletproof glass in the Salle des Etats?",
            ],
        },
        {
            "original": "Which ancient civilization built Machu Picchu?",
            "answers": ["inca", "incas", "incan"],
            "paraphrases": [
                "Which civilization built the terraced stone complex in the Andes that the outside world did not learn of until 1911?",
                "Which Andean civilization constructed the royal estate abandoned around the time of the Spanish conquest and never found by the conquistadors?",
            ],
        },
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "In which year did the conflict begin that participants optimistically expected to be over by Christmas?",
                "In which year did the war begin that would introduce trench warfare to the Western Front?",
            ],
        },
        {
            "original": "Who discovered penicillin?",
            "answers": ["fleming"],
            "paraphrases": [
                "Which scientist noticed that a contaminating mould had cleared a bacteria-free zone on a culture plate he had left out?",
                "Which researcher's untidy laboratory led to the accidental discovery of the first widely used antibiotic in 1928?",
            ],
        },
        {
            "original": "Who was the first person to walk on the Moon?",
            "answers": ["armstrong"],
            "paraphrases": [
                "Who spoke the words 'one small step for man' while descending a ladder in July 1969?",
                "Which astronaut's boot made the first human imprint in lunar regolith?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # C. Multi-hop.  The description requires chaining two facts.
    # -------------------------------------------------------------------
    "multi_hop": [
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "The artist who also designed a flying machine and wrote in mirror script painted which famous portrait's creator — that is, who is he?",
                "Who painted the portrait that shares its creator with The Last Supper?",
            ],
        },
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "In which year did the war begin whose armistice is commemorated each eleventh of November?",
                "The conflict that the Treaty of Versailles formally concluded began in which year?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # D. Archaic / low-frequency lexicon.  Same meaning, rare wording.
    # -------------------------------------------------------------------
    "archaic": [
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "By whose hand was the Mona Lisa limned?",
                "Unto which master is the authorship of the Mona Lisa ascribed?",
            ],
        },
        {
            "original": "Which ancient civilization built Machu Picchu?",
            "answers": ["inca", "incas", "incan"],
            "paraphrases": [
                "Which folk did raise the stone citadel known as Machu Picchu?",
                "By what people was the edifice of Machu Picchu reared?",
            ],
        },
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "In what year was the Great War joined?",
                "Anno what did the First World War break forth?",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # E. Meta framing.  Wrap the question rather than rephrase it.
    # -------------------------------------------------------------------
    "meta_frame": [
        {
            "original": "Who painted the Mona Lisa?",
            "answers": ["leonardo", "da vinci", "vinci"],
            "paraphrases": [
                "If a quiz asked who painted the Mona Lisa, what would the correct answer be?",
                "Complete this statement: the Mona Lisa was painted by",
                "A student answered this question wrongly. Who painted the Mona Lisa?",
            ],
        },
        {
            "original": "In which year did World War I begin?",
            "answers": ["1914"],
            "paraphrases": [
                "If a quiz asked in which year World War I began, what would the correct answer be?",
                "Complete this statement: World War I began in the year",
            ],
        },
    ],

    # -------------------------------------------------------------------
    # F. Weaker facts.  Same techniques applied where the model's knowledge
    #    is thinner, so there is less redundancy to fall back on.
    # -------------------------------------------------------------------
    "weak_fact": [
        {
            "original": "Who was the second person to walk on the Moon?",
            "answers": ["aldrin", "buzz"],
            "paraphrases": [
                "Which Apollo 11 crew member followed the mission commander onto the lunar surface?",
                "Who was the lunar module pilot who became the second human to stand on the Moon?",
            ],
        },
        {
            "original": "In which year was the Peace of Westphalia signed?",
            "answers": ["1648"],
            "paraphrases": [
                "In which year was the settlement signed that ended the Thirty Years' War?",
                "The treaties concluded at Osnabruck and Munster were signed in which year?",
            ],
        },
        {
            "original": "Which element has the atomic number 42?",
            "answers": ["molybdenum"],
            "paraphrases": [
                "Which element sits at position 42 on the periodic table?",
                "Which transition metal has 42 protons in its nucleus?",
            ],
        },
        {
            "original": "Who wrote the novel 'Things Fall Apart'?",
            "answers": ["achebe", "chinua"],
            "paraphrases": [
                "Which Nigerian author wrote the 1958 novel about Okonkwo and the coming of missionaries to Umuofia?",
                "Who wrote the novel whose title is drawn from Yeats's poem 'The Second Coming'?",
            ],
        },
    ],
}


def ask(url, model, question, max_tokens):
    """Return (extracted_answer, raw_output, had_think_close)."""
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
    return _extract_short_answer(raw), raw, ("</think>" in raw)


def check(answer, accepted):
    low = answer.lower()
    return any(a in low for a in accepted)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="qwen3:4b")
    p.add_argument("--url",   default="http://127.0.0.1:11435")
    p.add_argument("--set",   default=None)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--out",   default="results/probe2_results.json")
    args = p.parse_args()

    sets_to_run = {args.set: PROBE_SETS[args.set]} if args.set else PROBE_SETS

    records, hits = [], []

    for set_name, groups in sets_to_run.items():
        print(f"\n{'='*74}\nSET: {set_name}   (max_tokens={args.max_tokens})\n{'='*74}")

        for g in groups:
            original, accepted = g["original"], g["answers"]
            base, base_raw, base_closed = ask(args.url, args.model, original, args.max_tokens)
            base_ok = check(base, accepted)
            print(f"\n  ORIGINAL: {original}")
            print(f"    baseline → {base!r}  {'OK' if base_ok else '*** BASELINE FAILED ***'}")

            for para in g["paraphrases"]:
                ans, raw, closed = ask(args.url, args.model, para, args.max_tokens)
                ok = check(ans, accepted)
                # Distinguish failure modes
                if not ok and not closed:
                    mode = "no_converge"   # ran out of tokens still reasoning
                elif not ok:
                    mode = "wrong_answer"  # finished reasoning, answered wrong
                else:
                    mode = "ok"
                rec = {
                    "set": set_name, "original": original, "paraphrase": para,
                    "answer": ans, "correct": ok, "mode": mode,
                    "raw_len": len(raw), "reasoning_closed": closed,
                    "baseline_answer": base, "baseline_correct": base_ok,
                    "max_tokens": args.max_tokens,
                }
                records.append(rec)
                if not ok and base_ok:
                    rec["raw"] = raw
                    hits.append(rec)
                tag = "  ok  " if ok else f">>{mode}"
                print(f"    [{tag:^13}] {para}")
                print(f"                    → {ans!r}   (raw {len(raw)} chars, closed={closed})")

    print(f"\n\n{'='*74}\nSUMMARY: {len(hits)} hits / {len(records)} probes")
    by_mode = {}
    for r in records:
        by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
    print(f"  modes: {by_mode}")
    print("=" * 74)
    for h in hits:
        print(f"\n  [{h['set']} / {h['mode']}]  {h['original']}")
        print(f"    {h['paraphrase']}")
        print(f"    → {h['answer']!r}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"records": records, "hits": hits}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {args.out}")


if __name__ == "__main__":
    main()
