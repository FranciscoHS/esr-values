# esr-values

Do language models resist being steered toward sycophancy, the way they resist being steered off-topic ([endogenous steering resistance](https://arxiv.org/abs/2602.06941))?

**Status: on hold.** Sycophancy steering (CAA) works on Llama-3.1-8B and Llama-3.3-70B. 70B sometimes silently corrects itself after agreeing with a false claim, but it never explicitly backtracks, and our positive control did not reproduce the paper's ESR rate. See [`log.md`](log.md) for results and next steps.

```
python -m esr.data                              # build datasets (CPU)
python -m esr.vectors extract                   # sycophancy vectors
python -m esr.generate --layer 13 --mults 0 4 6 # steer on false-claim prompts
python -m esr.judge batches <gen.jsonl> --name <run>   # LLM-judge batches
python -m esr.concept vectors                   # positive control (concept steering)
```

Set `ESR_MODEL` to change the model (default `meta-llama/Llama-3.1-8B-Instruct`). The cluster jobs are in `scripts/`.
