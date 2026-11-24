"""MULTILINGUAL logit-lens --- is the latent-English finding MODEL-SPECIFIC? For a given model, feed a
non-English two-hop prompt and track P(answer 1st token) in ALL FIVE languages (en/es/ko/zh/ja) across every
layer. The language whose probability PEAKS in the middle layers is the model's 'thinking language'.
Intervention-free, model-agnostic (hidden_states + final norm + unembedding). Run across Llama / Qwen / Gemma
and scales to test whether the pivot language depends on the model's training (e.g. Qwen -> Chinese?).

    python scripts/multilingual_lens.py --model qwen25_72 --prompt-langs es ko
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.prompting import encode, readout_col
from beacon.models import load_model
from beacon.pipeline.patching import first_answer_token_id

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

LANGS = ["en", "es", "ko", "zh", "ja"]

def _final_norm(model):
    """Robust across architectures (Llama/Qwen model.model.norm; Gemma-3 multimodal nests the text decoder)."""
    for path in ("model.norm", "model.language_model.norm", "model.model.norm",
                 "language_model.model.norm", "model.language_model.model.norm", "transformer.ln_f"):
        obj = model
        for p in path.split("."):
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None and callable(obj):
            return obj
    raise RuntimeError("final norm module not found")

def _lm_head(model):
    h = model.get_output_embeddings()
    if h is None:
        h = getattr(model, "lm_head", None)
    if h is None:                   
        h = model.get_input_embeddings()
    return h

@torch.no_grad()
def _layer_probs(loaded, prompt, ans_toks, chat=False):
    """Per-layer P(answer 1st token) per language at the readout position, via logit lens. chat=True routes
    through the model chat template and reads the last (generation) token; else the raw 'Answer:' colon."""
    norm = _final_norm(loaded.model)
    lm_head = _lm_head(loaded.model)
    nd = next(norm.parameters()).device
    ndt = next(norm.parameters()).dtype
    ld = lm_head.weight.device
    ids = encode(loaded, prompt, chat)
    if chat:

                                                                                                       
        g = loaded.model.generate(input_ids=ids, max_new_tokens=1, do_sample=False,
                                  pad_token_id=loaded.tokenizer.eos_token_id)
        ids = g[:, :ids.shape[1] + 1]
        col = ids.shape[1] - 1
    else:
        col = readout_col(loaded, prompt, chat)
    hs = loaded.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
    per = {L: [] for L in LANGS}
    for h in hs:
        hn = norm(h[0, col].to(device=nd, dtype=ndt))
        logits = lm_head(hn.to(ld)).float().cpu()
        p = torch.softmax(logits, dim=-1)
        for L in LANGS:
            per[L].append(float(p[ans_toks[L]]) if ans_toks[L] >= 0 else 0.0)
    return per

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_70")
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--targets", default="02-data/inputs/es_flip_targets.jsonl")
    ap.add_argument("--prompt-langs", nargs="*", default=["es"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/access")
    args = ap.parse_args()

    load_env()
    cfg = load_configs()
    C = json.load(open(_path(args.chains)))
    chain_ids = [json.loads(x)["chain_id"] for x in open(_path(args.targets))]
    chain_ids = [c for c in chain_ids if c in C]
    if args.limit:
        chain_ids = chain_ids[:args.limit]
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[mlens] loading {args.model} | chains={len(chain_ids)} prompt_langs={args.prompt_langs} ...", flush=True)
    loaded = load_model(args.model, cfg, framework="hf_hooks", attn_implementation="eager")

    results = {"model": args.model, "by_prompt_lang": {}}
    for plang in args.prompt_langs:
        agg = None
        n = 0
        for i, c in enumerate(chain_ids):
            labs = C[c]["answer_label_by_lang"]
            prompt = C[c]["prompts"].get(plang)
            if not prompt:
                continue
            ans_toks = {L: (first_answer_token_id(loaded, labs[L]) if labs.get(L) else -1) for L in LANGS}

            per = _layer_probs(loaded, prompt, ans_toks, chat=args.chat)
            depth = len(per["en"])
            if agg is None:
                agg = {L: [0.0] * depth for L in LANGS}
            for L in LANGS:
                for k in range(depth):
                    agg[L][k] += per[L][k]
            n += 1
            if (i + 1) % 60 == 0:
                print(f"    {plang} [{i + 1}/{len(chain_ids)}] n={n}", flush=True)
        if not n:
            continue
        mean = {L: [x / n for x in agg[L]] for L in LANGS}
        depth = len(mean["en"])
        peak = {L: (max(range(depth), key=lambda k: mean[L][k]), max(mean[L])) for L in LANGS}
        results["by_prompt_lang"][plang] = {"n": n, "depth": depth, "mean_prob": mean, "peak": peak}
        (out_dir / f"mlens_{args.model}.json").write_text(json.dumps(results, indent=2))

    for plang, r in results["by_prompt_lang"].items():
        depth = r["depth"]
        print(f"\n[mlens] ===== {args.model} | {plang}-prompt (n={r['n']}) : P(answer) by layer, per language =====",
              flush=True)
        print("  peak per lang:  " + "  ".join(f"{L}={r['peak'][L][1]:.3f}@L{r['peak'][L][0]}" for L in LANGS),
              flush=True)
        print(f"  {'layer':>5} " + " ".join(f"{L:>7}" for L in LANGS), flush=True)
        step = max(1, depth // 16)
        for k in list(range(0, depth, step)) + [depth - 1]:
            print(f"  {k:>5} " + " ".join(f"{r['mean_prob'][L][k]:>7.3f}" for L in LANGS), flush=True)
        win = max(LANGS, key=lambda L: r["peak"][L][1])
        print(f"  [THINKS-IN] mid-layer peak language = {win} (P={r['peak'][win][1]:.3f} @ L{r['peak'][win][0]})",
              flush=True)

if __name__ == "__main__":
    main()
