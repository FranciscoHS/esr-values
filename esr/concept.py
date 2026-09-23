"""Positive control: off-topic concept steering on the ESR paper's 38 "explain how" prompts (GPU).

Replicates the setting where McKenzie et al. (2026, arXiv 2602.06941) report explicit ESR on
Llama-3.3-70B (3.8% of steered responses; 3.6% with contrastive vectors instead of SAE latents),
with our own difference-of-means vectors instead of Goodfire SAE latents.

vectors   For each concept, the model writes N_PER_CONCEPT short texts about it ("Write a paragraph
          about <concept>.", sampled at T=1). Residual activations are mean-pooled over the response
          tokens, all layers. vec_c = mean_c - mean over all concepts. Saves vec_concepts.pt with
          vec {concept: [L+1, D]} and resid_norm [L+1] (mean per-token ||h|| over those responses).
generate  Steer with alpha * resid_norm[l] * unit(vec_c[l]) at all positions, on data/esr_prompts.txt,
          sampling at T=0.6 with up to 512 new tokens (the paper's settings). alpha = 0 is the
          unsteered baseline. Writes gen_concept_L{layer}.jsonl: {id, prompt, concept, alpha, sample, response}.

    python -m esr.concept vectors
    python -m esr.concept generate --layer 33 --concepts body_positions --alphas 0 0.5 0.75 1 1.25 1.5 --samples 1
"""
from __future__ import annotations

import argparse
import zlib

import torch
from tqdm import tqdm

from esr.common import DATA, OUT, chat_prompt, load_model, read_jsonl, steer, write_jsonl
from esr.generate import generate

CONCEPTS = {
    "body_positions": "body positions and postures (sitting, kneeling, crouching, lying down, standing on one leg)",
    "volcanoes": "volcanoes and volcanic eruptions",
    "medieval_castles": "medieval castles and their architecture",
    "deep_sea_creatures": "deep-sea creatures such as anglerfish and giant squid",
}
N_PER_CONCEPT = 48
TEMPERATURE, MAX_NEW_TOKENS = 0.6, 512


@torch.no_grad()
def vectors(batch_size: int, n_per_concept: int) -> None:
    tok, model = load_model()
    torch.manual_seed(0)
    means, norms = {}, []
    for name, desc in CONCEPTS.items():
        prompts = [chat_prompt(tok, f"Write a paragraph about {desc}.")] * n_per_concept
        texts = generate(tok, model, prompts, batch_size, 200, temperature=1.0)
        acc, n = None, 0
        for b in tqdm(range(0, len(texts), batch_size), desc=name, leave=False):
            ps, ts = prompts[b:b + batch_size], texts[b:b + batch_size]
            enc = tok([p + t for p, t in zip(ps, ts)], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
            hs = torch.stack(model(**enc, output_hidden_states=True).hidden_states, dim=2).float()  # [B, T, L+1, D]
            T = enc.input_ids.shape[1]
            for j, (p, t) in enumerate(zip(ps, ts)):
                k = len(tok(p + t, add_special_tokens=False).input_ids) - len(tok(p, add_special_tokens=False).input_ids)
                h = hs[j, T - k:]  # response tokens (left padding: they are the last k)
                acc = h.sum(0) if acc is None else acc + h.sum(0)
                n += h.shape[0]
                norms.append(h.norm(dim=-1).mean(0).cpu())
        means[name] = (acc / n).cpu()
        print(name, "tokens pooled:", n, "| e.g.", repr(texts[0][:150]), flush=True)
    grand = torch.stack(list(means.values())).mean(0)
    vec = {c: m - grand for c, m in means.items()}
    torch.save(dict(vec=vec, resid_norm=torch.stack(norms).mean(0), concepts=CONCEPTS), OUT / "vec_concepts.pt")
    rn = torch.stack(norms).mean(0)
    ls = [round(f * (len(rn) - 1)) for f in (0.25, 0.41, 0.6)]
    for c, v in vec.items():
        print(c, f"||v||/||h|| at layers {ls}:", [round((v[l].norm() / rn[l]).item(), 3) for l in ls])


def main(layer: int, concepts: list[str], alphas: list[float], samples: int, batch_size: int, limit: int | None) -> None:
    tok, model = load_model()
    d = torch.load(OUT / "vec_concepts.pt")
    prompts = [l.strip() for l in open(DATA / "esr_prompts.txt") if l.strip()][:limit]
    path = OUT / f"gen_concept_L{layer}.jsonl"
    out = read_jsonl(path) if path.exists() else []
    have = {(o["id"], o["alpha"]) for o in out}
    for c in concepts:
        u = d["vec"][c][layer] / d["vec"][c][layer].norm()
        for a in alphas:
            if a == 0 and c != concepts[0]:
                continue  # one unsteered baseline, not one per concept
            rows = [dict(id=f"p{i}_{c if a else 'none'}_s{s}", prompt=p, concept=c if a else "none", alpha=a, sample=s)
                    for i, p in enumerate(prompts) for s in range(samples)]
            rows = [r for r in rows if (r["id"], a) not in have]
            print(f"{c} alpha={a}: {len(rows)} to generate", flush=True)
            if not rows:
                continue
            torch.manual_seed(int(1000 * a) + zlib.crc32(c.encode()) % 1000)
            with steer(model, layer, a * d["resid_norm"][layer] * u if a else None):
                resps = generate(tok, model, [chat_prompt(tok, r["prompt"]) for r in rows], batch_size, MAX_NEW_TOKENS, TEMPERATURE)
            out += [dict(r, response=g) for r, g in zip(rows, resps)]
            write_jsonl(path, out)
            print(f"  e.g. {rows[0]['prompt']!r} -> {resps[0][:300]!r}", flush=True)
    print("wrote", len(out), "->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["vectors", "generate"])
    ap.add_argument("--layer", type=int, default=33)
    ap.add_argument("--concepts", nargs="+", default=["body_positions"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0, 0.5, 0.75, 1, 1.25, 1.5])
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n", type=int, default=N_PER_CONCEPT, help="texts per concept for vectors")
    ap.add_argument("--limit", type=int, default=None, help="first N prompts only")
    a = ap.parse_args()
    vectors(a.batch_size, a.n) if a.stage == "vectors" else main(a.layer, a.concepts, a.alphas, a.samples, a.batch_size, a.limit)
