"""Free-form generation on the false-claim eval under sycophancy steering (GPU).

Greedy decoding on data/eval_claims.jsonl (neutral + false_claim prompts) for each multiplier.
(row, multiplier) pairs already in the output file are kept and not regenerated.
Writes outputs/gen_{vec}_L{layer}.jsonl: rows {id, qid, condition, mult, response}.

    python -m esr.generate --vec caa --layer 14 --mults -2 0 2 4
"""
from __future__ import annotations

import argparse

import torch
from tqdm import tqdm

from esr.common import DATA, OUT, chat_prompt, load_model, read_jsonl, steer, write_jsonl


@torch.no_grad()
def generate(tok, model, prompts: list[str], batch_size: int, max_new_tokens: int, temperature: float = 0.0) -> list[str]:
    """Batched generation; greedy if temperature == 0, else sampling at that temperature."""
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for b in tqdm(range(0, len(order), batch_size), leave=False):
        idx = order[b:b + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        sampling = dict(do_sample=True, temperature=temperature, top_p=1.0) if temperature > 0 else dict(do_sample=False)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id, **sampling)
        for i, g in zip(idx, tok.batch_decode(out[:, enc.input_ids.shape[1]:], skip_special_tokens=True)):
            res[i] = g.strip()
    return res


def main(vec_src: str, layer: int, mults: list[float], batch_size: int, max_new_tokens: int, limit: int | None) -> None:
    tok, model = load_model()
    vec = torch.load(OUT / f"vec_{vec_src}.pt")["vec"][layer]
    rows = read_jsonl(DATA / "eval_claims.jsonl")[: limit and 2 * limit]
    prompts = [chat_prompt(tok, r["user"]) for r in rows]
    path = OUT / f"gen_{vec_src}_L{layer}.jsonl"
    # Keep earlier runs; only generate (row, multiplier) pairs not already in the file.
    out = read_jsonl(path) if path.exists() else []
    have = {(o["id"], o["mult"]) for o in out}
    for m in mults:
        todo = [i for i, r in enumerate(rows) if (r["id"], m) not in have]
        print(f"mult {m}: {len(rows) - len(todo)} done, {len(todo)} to generate", flush=True)
        if not todo:
            continue
        with steer(model, layer, m * vec if m else None):
            resps = generate(tok, model, [prompts[i] for i in todo], batch_size, max_new_tokens)
        out += [dict(id=rows[i]["id"], qid=rows[i]["qid"], condition=rows[i]["condition"], mult=m, response=g) for i, g in zip(todo, resps)]
        write_jsonl(path, sorted(out, key=lambda o: (o["mult"], o["id"])))
    out.sort(key=lambda o: (o["mult"], o["id"]))
    write_jsonl(path, out)
    print("wrote", len(out), "->", path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--vec", default="caa")
    ap.add_argument("--layer", type=int, default=13)
    ap.add_argument("--mults", type=float, nargs="+", default=[-4, -2, 0, 2, 4, 8])
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--limit", type=int, default=None, help="number of questions")
    a = ap.parse_args()
    main(a.vec, a.layer, a.mults, a.batch_size, a.max_new_tokens, a.limit)
