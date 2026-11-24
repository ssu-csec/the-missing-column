"""C7: latent language-FORM PROFILE --- in which language forms does the answer exist inside the stack?

For each failing item, trace P(answer's first token in language L) for L in {en,es,ko,ja,zh} at EVERY layer
(logit lens through the final norm), Delta-CONTROLLED: subtract the per-layer probability of a language-matched
BASELINE token set (the same language's generic content tokens), so tokenizer/frequency bias cannot fake a form.
Per (model x target-language) aggregate: the per-layer form-availability curve per language + the item-level
peak form availability. THE TEST: does form availability predict the E3 attractor set (es|Llama, zh|Gemma-4/Qwen,
never ko/ja)? Either answer is a finding --- predictor found, or the mystery is representational-level too.

Population: solvmatrix fails of T with en solved (knowledge exists), labels distinct --- population A.
Output: 02-data/results/access/formprofile_<model>.json with full per-item per-layer curves (provenance-complete).

    python scripts/form_profile.py --model gemma4_31b --solvmatrix 02-data/results/solvability/solvability_gemma4_31b.json
    python scripts/form_profile.py --model qwen36_27b --chat --solvmatrix 02-data/results/solvability/solvability_qwen36_27b.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.prompting import encode
from beacon.models import load_model
from beacon.pipeline.patching import first_answer_token_id

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

_lo = importlib.util.spec_from_file_location("lo", str(REPO_ROOT / "scripts/vt_override_layer.py"))
lo = importlib.util.module_from_spec(_lo); _lo.loader.exec_module(lo)
LANGS = ["en", "es", "ko", "ja", "zh"]

@torch.no_grad()
def _lang_baseline_ids(C, loaded, lang, cap=48):
    """Language-matched baseline token ids: first tokens of OTHER chains' answers in the same language.
    (The scripts/35 Delta control: same language, same construction, different content.)"""
    ids, seen = [], set()
    for c in C:
        lab = C[c]["answer_label_by_lang"].get(lang)
        if not lab:
            continue
        t = first_answer_token_id(loaded, lab)
        if t >= 0 and t not in seen:
            seen.add(t); ids.append(t)
        if len(ids) >= cap:
            break
    return ids

@torch.no_grad()
def _curves_multi(loaded, prompt, chat, tok_ids, base_ids, norm, lm):
    """One forward: per-layer P(token) for each language's answer token AND mean P over its baseline set.
    Returns {lang: [p_l0..]}, {lang: [pbase_l0..]}, depth."""
    full, col = lo._readout(loaded, prompt, chat)
    hs = loaded.model(input_ids=full, use_cache=False, output_hidden_states=True).hidden_states
    nd = next(norm.parameters()).device; ndt = next(norm.parameters()).dtype; ld = lm.weight.device
    P = {L: [] for L in tok_ids}; B = {L: [] for L in tok_ids}
    for h in hs:
        p = torch.softmax(lm(norm(h[0, col].to(nd, ndt)).to(ld)).float().cpu(), dim=-1)
        for L, t in tok_ids.items():
            P[L].append(float(p[t]) if t >= 0 else 0.0)
            bi = base_ids[L]
            B[L].append(float(p[torch.tensor(bi)].mean()) if bi else 0.0)
    return P, B, len(hs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--solvmatrix", help="default: 02-data/results/solvability/solvability_{model}.json")
    ap.add_argument("--targets", nargs="*", default=["es", "ko", "ja", "zh"])
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--n-per-lang", type=int, default=60)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/access")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    pc = json.load(open(_path(a.solvmatrix or f"02-data/results/solvability/solvability_{a.model}.json")))["per_chain"]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fp] loading {a.model} (chat={a.chat})", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    norm = lo._final_norm(loaded.model)
    lm = loaded.model.get_output_embeddings()
    if lm is None:
        raise RuntimeError(f"{a.model}: no output embeddings (lm_head); cannot lens-decode hidden states")
    base_ids = {L: _lang_baseline_ids(C, loaded, L) for L in LANGS}
    print(f"[fp] baseline set sizes: " + " ".join(f"{L}={len(base_ids[L])}" for L in LANGS), flush=True)

    res = {"model": a.model, "protocol": {"solvmatrix": a.solvmatrix, "delta_control": "same-lang other-chain answer tokens",
                                          "n_baseline": {L: len(base_ids[L]) for L in LANGS}}, "by_target": {}}
    for T in a.targets:
        items = [c for c in pc if pc[c]["x_correct"].get(T) is False
                 and (pc[c]["x_correct"].get("en") or pc[c]["any_ok"].get("en"))
                 and c in C and C[c]["answer_label_by_lang"].get(T)
                 and C[c]["answer_label_by_lang"].get(T) != C[c]["answer_label_by_lang"].get("en")][:a.n_per_lang]
        rows = []
        for i, c in enumerate(items):
            labs = C[c]["answer_label_by_lang"]
            toks = {L: (first_answer_token_id(loaded, labs.get(L)) if labs.get(L) else -1) for L in LANGS}

            for L in ("es", "ko", "ja", "zh"):
                if toks[L] >= 0 and toks[L] == toks["en"]:
                    toks[L] = -1
            P, B, depth = _curves_multi(loaded, C[c]["prompts"][T], a.chat, toks, base_ids, norm, lm)
            delta = {L: [p - b for p, b in zip(P[L], B[L])] for L in LANGS}
            peak = {L: (max(delta[L][1:depth - 1]) if toks[L] >= 0 else None) for L in LANGS}
            rows.append({"chain": c, "tok_ids": toks, "peak_delta": peak, "delta_curves": delta})
            if (i + 1) % 20 == 0:
                print(f"    {T} [{i+1}/{len(items)}]", flush=True)
        n = max(1, len(rows))
        agg = {}
        for L in LANGS:
            vals = [r["peak_delta"][L] for r in rows if r["peak_delta"][L] is not None]
            agg[L] = {"n": len(vals), "mean_peak_delta": (sum(vals) / len(vals)) if vals else None,
                      "frac_positive": (sum(1 for v in vals if v > 0) / len(vals)) if vals else None}
        res["by_target"][T] = {"n": len(rows), "agg": agg, "items": rows}
        print(f"[fp] {T}: n={len(rows)} | mean peak Delta-form: " +
              " ".join(f"{L}={agg[L]['mean_peak_delta']:.4f}" if agg[L]["mean_peak_delta"] is not None else f"{L}=-"
                       for L in LANGS), flush=True)
        (out_dir / f"formprofile_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[fp] DONE --- compare per-language form availability with the E3 render attractor set (the C7 test).", flush=True)

if __name__ == "__main__":
    main()
