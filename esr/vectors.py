"""Sycophancy steering vectors (CAA-style) and an A/B steering sweep (GPU).

extract  For each A/B item, run "<chat prompt>(X" for X = matching and non-matching letter and
         take the residual stream at the letter token, all layers. vec[l] = mean(match - nonmatch),
         averaged with equal weight over items whose matching letter is A and B.
         Saves outputs/vec_{src}.pt: vec [L+1, D], resid_norm [L+1] (mean ||h|| at that token).
sweep    For one vector and layer, steer (add mult * vec[l] at all positions) at each multiplier and
         measure P(matching letter) on each held-out A/B set, normalised over the two options, from a
         single forward pass, reading the letter after the prefill "The answer is (".
         p_either = P(A) + P(B) there, to check the model still answers in format.
         Writes outputs/ab_sweep_{vec}_L{layer}.csv.

    python -m esr.vectors extract
    python -m esr.vectors sweep [--vec caa] [--layer 13] [--mults -2 -1 0 1 2 4]
"""
from __future__ import annotations

import argparse

import pandas as pd
import torch
from tqdm import tqdm

from esr.common import DATA, OUT, chat_prompt, load_model, read_jsonl, steer

SOURCES = ["caa", "fact"]
# The model rarely opens with "(A"; it writes e.g. "The correct answer is (B)". Score the letter after this prefill.
PREFILL = "The answer is ("


@torch.no_grad()
def last_hidden(tok, model, texts: list[str], batch_size: int) -> torch.Tensor:
    """Hidden states at the last token of each text: [N, L+1, D] float32 on CPU."""
    outs = []
    for b in tqdm(range(0, len(texts), batch_size), leave=False):
        enc = tok(texts[b:b + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        hs = model(**enc, output_hidden_states=True).hidden_states
        outs.append(torch.stack([h[:, -1] for h in hs], dim=1).float().cpu())
    return torch.cat(outs)


@torch.no_grad()
def letter_logprobs(tok, model, prompts: list[str], batch_size: int) -> torch.Tensor:
    """log p("A"), log p("B") as the next token after prompt + PREFILL: [N, 2]."""
    ids = [tok.convert_tokens_to_ids(c) for c in "AB"]
    res = []
    for b in range(0, len(prompts), batch_size):
        enc = tok([p + PREFILL for p in prompts[b:b + batch_size]], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        res.append(model(**enc, logits_to_keep=1).logits[:, -1].float().log_softmax(-1)[:, ids].cpu())
    return torch.cat(res)


def extract(batch_size: int) -> None:
    tok, model = load_model()
    for src in SOURCES:
        rows = read_jsonl(DATA / f"ab_{src}_train.jsonl")
        prompts = [chat_prompt(tok, r["user"]) for r in rows]
        hm = last_hidden(tok, model, [p + f"({r['match']}" for p, r in zip(prompts, rows)], batch_size)
        hn = last_hidden(tok, model, [p + f"({r['nonmatch']}" for p, r in zip(prompts, rows)], batch_size)
        # Equal weight on match=A and match=B items, so the "(A" vs "(B" token identity cancels exactly.
        is_a = torch.tensor([r["match"] == "A" for r in rows])
        d = hm - hn
        vec = (d[is_a].mean(0) + d[~is_a].mean(0)) / 2
        resid_norm = torch.cat([hm, hn]).norm(dim=-1).mean(0)
        torch.save(dict(vec=vec, resid_norm=resid_norm, n=len(rows)), OUT / f"vec_{src}.pt")
        ratio = (vec.norm(dim=-1) / resid_norm)
        print(src, "vec", tuple(vec.shape), "||v||/||h|| by layer:", " ".join(f"{x:.3f}" for x in ratio.tolist()))
    c = torch.nn.functional.cosine_similarity(*(torch.load(OUT / f"vec_{s}.pt")["vec"] for s in SOURCES), dim=-1)
    print("cos(caa, fact) by layer:", " ".join(f"{x:.2f}" for x in c.tolist()))


def sweep(vsrc: str, layer: int, mults: list[float], batch_size: int) -> None:
    tok, model = load_model()
    vec = torch.load(OUT / f"vec_{vsrc}.pt")["vec"][layer]
    out = []
    for tsrc in SOURCES:
        rows = read_jsonl(DATA / f"ab_{tsrc}_test.jsonl")
        prompts = [chat_prompt(tok, r["user"]) for r in rows]
        is_a = torch.tensor([r["match"] == "A" for r in rows])
        for m in mults:
            with steer(model, layer, m * vec if m else None):
                lp = letter_logprobs(tok, model, prompts, batch_size)
            lm = torch.where(is_a, lp[:, 0], lp[:, 1])
            ln = torch.where(is_a, lp[:, 1], lp[:, 0])
            p = torch.sigmoid(lm - ln)
            out.append(dict(vec=vsrc, layer=layer, mult=m, test=tsrc, p_match=p.mean().item(),
                            frac_match=(p > 0.5).float().mean().item(), p_either=(lm.exp() + ln.exp()).mean().item()))
            print(out[-1], flush=True)
    df = pd.DataFrame(out)
    df.to_csv(OUT / f"ab_sweep_{vsrc}_L{layer}.csv", index=False)
    print(df.pivot_table(index="test", columns="mult", values=["p_match", "p_either"]).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["extract", "sweep"])
    ap.add_argument("--vec", default="caa")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--mults", type=float, nargs="+", default=[-2, -1, 0, 1, 2, 4])
    ap.add_argument("--batch_size", type=int, default=16)
    a = ap.parse_args()
    extract(a.batch_size) if a.stage == "extract" else sweep(a.vec, a.layer, a.mults, a.batch_size)
