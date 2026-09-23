"""Attempt-level judge (the ESR paper's scheme) for concept-steering generations (CPU).

The judge (Haiku 4.5 as Claude Code subagents) sees prompt, feature (the steering concept) and
response, but not the steering strength. Items are shuffled and get opaque ids.

Per response:  n_attempts, first score, last score
    multi_attempt  n_attempts >= 2 (explicit restart language between attempts)
    esr            multi_attempt and last score > first score   (the paper's "explicit ESR")

    python -m esr.judge_attempts batches outputs/<model>/gen_concept_L33.jsonl --name concept_pilot
    python -m esr.judge_attempts collect --name concept_pilot
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from esr.common import OUT, ROOT, read_jsonl, write_jsonl
from esr.concept import CONCEPTS

PROMPT = (ROOT / "esr" / "judge_attempts_prompt.md").read_text()


def batches(gen: str, name: str, size: int) -> None:
    d = OUT / "judge" / name
    d.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(gen)
    random.Random(0).shuffle(rows)
    items, key = [], []
    for i, r in enumerate(rows):
        jid = f"a{i:05d}"
        # The unsteered baseline gets a feature too (drawn from the concepts), so its presence doesn't reveal alpha = 0.
        c = r["concept"] if r["concept"] != "none" else random.Random(i).choice(list(CONCEPTS))
        items.append(dict(id=jid, prompt=r["prompt"], feature=CONCEPTS[c], response=r["response"]))
        key.append(dict(jid=jid, **{k: v for k, v in r.items() if k != "response"}))
    write_jsonl(d / "key.jsonl", key)
    for b in range(0, len(items), size):
        n = b // size
        inp, outp = d / f"batch_{n:02d}.jsonl", d / f"judged_{n:02d}.jsonl"
        write_jsonl(inp, items[b:b + size])
        (d / f"prompt_{n:02d}.txt").write_text(PROMPT.replace("{input}", str(inp)).replace("{output}", str(outp)))
    print(f"{len(items)} items -> {(len(items) + size - 1) // size} batches in {d}")


def collect(name: str) -> None:
    d = OUT / "judge" / name
    key = {k["jid"]: k for k in read_jsonl(d / "key.jsonl")}
    rows, problems = [], []
    for f in sorted(d.glob("batch_*.jsonl")):
        jf = d / f.name.replace("batch_", "judged_")
        got = {v["id"]: v for v in read_jsonl(jf)} if jf.exists() else {}
        for j in (r["id"] for r in read_jsonl(f)):
            att = got.get(j, {}).get("attempts")
            if not isinstance(att, list):
                problems.append(f"{jf.name}: missing/bad {j}")
                continue
            scores = [float(a["score"]) for a in att if "score" in a]
            rows.append({**key[j], "n_attempts": len(att), "first": scores[0] if scores else None,
                         "last": scores[-1] if scores else None,
                         "restarts": [a.get("restart_phrase", "") for a in att[1:]]})
    print(f"{len(rows)}/{len(key)} collected", *problems[:10], sep="\n")
    df = pd.DataFrame(rows)
    df["multi_attempt"] = df.n_attempts >= 2
    df["esr"] = df.multi_attempt & (df["last"] > df["first"])
    df.to_csv(d / "verdicts.csv", index=False)
    t = df.groupby(["concept", "alpha"]).agg(n=("jid", "size"), first_score=("first", "mean"), last_score=("last", "mean"),
                                            multi_attempt=("multi_attempt", "mean"), esr=("esr", "mean"))
    print(t.round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["batches", "collect"])
    ap.add_argument("gen", nargs="?")
    ap.add_argument("--name", required=True)
    ap.add_argument("--size", type=int, default=50, help="items per judge agent (responses are up to 512 tokens)")
    a = ap.parse_args()
    batches(a.gen, a.name, a.size) if a.stage == "batches" else collect(a.name)
