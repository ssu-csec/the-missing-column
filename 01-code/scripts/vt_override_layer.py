"""Late-override locate-then-steer. Tests whether an instruct two-hop failure carries the correct answer
mid-stack (in the pivot language) that a late translate-to-target step overwrites with a wrong target entity,
and whether editing at the located late layers recovers the correct answer in the target language, compared
with a mid-layer edit.

LOCATE per A-chain: track cp_P (correct pivot token), cp_T (correct target token), wo_T (actually generated
token) per layer. L_peak = argmax cp_P (mid); L_override = first late layer where wo_T exceeds both cp_P and
cp_T for 2 consecutive layers. Band = [median L_override, +k]. F1 check: the override account is not supported
if median L_override <= L_peak (recorded as F1_dead).
INTERVENE (two additive edits in the band, at the generation position, each step): content u_c (answer axis,
orthogonalized against v_lang) + target-language direction v_T. Recovery = designated-form realization only
(strict target-language form + output_language==T).

    python scripts/vt_override_layer.py --model llama31_70_it --pivot en --targets es ko zh ja --chat \
        --solvmatrix 02-data/results/solvability/solvability_llama31_70_it.json --alpha-c 1 --alpha-s 1
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import _form_matches, _has_cjk, _normalize, entity_match
from beacon.data.langid import output_language
from beacon.data.prompting import encode
from beacon.models import load_model
from beacon.pipeline.patching import _decoder_layers, _unwrap_layer_out, first_answer_token_id

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

_rs = importlib.util.spec_from_file_location("rs", str(REPO_ROOT / "scripts/solvability_rescore.py"))
rs = importlib.util.module_from_spec(_rs); _rs.loader.exec_module(rs)
LANGS = ["en", "es", "ko", "zh", "ja"]

def _final_norm(m):
    for p in ("model.norm", "model.language_model.norm", "model.model.norm"):
        o = m
        for a in p.split("."):
            o = getattr(o, a, None)
        if o is not None and callable(o):
            return o

                                                                              
    from beacon.hooks import _text_model
    o = getattr(_text_model(m), "norm", None)
    if o is not None and callable(o):
        return o
    raise RuntimeError("no norm")

@torch.no_grad()
def _readout(loaded, prompt, chat):
    ids = encode(loaded, prompt, chat)
    g = loaded.model.generate(input_ids=ids, max_new_tokens=1, do_sample=False,
                              pad_token_id=loaded.tokenizer.eos_token_id)
    full = g[:, :ids.shape[1] + 1]
    return full, full.shape[1] - 1

@torch.no_grad()
def _curves(loaded, prompt, chat, toks, norm, lm):
    """Per-layer lens probability of each named token at the last prompt position (the LOCATE trace).

    The last prompt position is where each layer's lens distribution decides the first emitted token
    (reading at the generation position would lens token 2, making P(wo_T) structurally ~0 in late layers).
    When 'wo_T' is requested, it is replaced with the actually generated first token id (full[0, -1]), so
    P(wo_T) -> ~1 at the final layer by construction."""
    full, _ = _readout(loaded, prompt, chat)
    col = full.shape[1] - 2
    if "wo_T" in toks:
        toks = {**toks, "wo_T": int(full[0, -1])}
    hs = loaded.model(input_ids=full, use_cache=False, output_hidden_states=True).hidden_states
    nd = next(norm.parameters()).device; ndt = next(norm.parameters()).dtype; ld = lm.weight.device
    per = {k: [] for k in toks}
    for h in hs:
        p = torch.softmax(lm(norm(h[0, col].to(nd, ndt)).to(ld)).float().cpu(), dim=-1)
        for k, t in toks.items():
            per[k].append(float(p[t]) if t >= 0 else 0.0)
    return per, len(hs)

def _locate(cp_P, cp_T, wo_T, depth):
    mc = depth - 4
    L_peak = max(range(1, mc), key=lambda L: cp_P[L])
    for L in range(L_peak + 1, depth - 2):
        if all(wo_T[j] > cp_P[j] and wo_T[j] > cp_T[j] for j in (L, L + 1)):
            return L_peak, L

    rng = range(L_peak + 1, depth - 2)
    return L_peak, max(rng, key=lambda L: wo_T[L] - cp_P[L]) if rng else L_peak

@torch.no_grad()
def _resid(loaded, prompt, chat, layer):
    full, col = _readout(loaded, prompt, chat)
    hs = loaded.model(input_ids=full, use_cache=False, output_hidden_states=True).hidden_states
    return hs[layer + 1][0, col].detach().float().cpu()

@torch.no_grad()
def _resid_all(loaded, prompt, chat, layers):
    """all requested layers' gen-position residuals from ONE forward (avoids per-(chain,layer,condition) recompute)."""
    full, col = _readout(loaded, prompt, chat)
    hs = loaded.model(input_ids=full, use_cache=False, output_hidden_states=True).hidden_states
    return {int(L): hs[int(L) + 1][0, col].detach().float().cpu() for L in layers}

@torch.no_grad()
def _build_vlang(loaded, pairs, layer, tgt, src, chat):
    acc = None; n = 0
    for p in pairs:
        if tgt not in p or src not in p:
            continue
        d = _resid(loaded, p[tgt], chat, layer) - _resid(loaded, p[src], chat, layer)
        acc = d.clone() if acc is None else acc + d
        n += 1
    m = acc / n
    return m / (m.norm() + 1e-8)

def _tforms(labs, al, T):
    forms = [labs.get(T)]
    for a in (al.get(T) or []):
        if a and (_has_cjk(a) and len(a) >= 2 or len(a.split()) >= 2):
            forms.append(a)
    return [f for f in forms if f]

def _recovered_T(out, labs, al, T):
    if output_language(out, labs) != T:
        return False
    p = _normalize(out)
    return bool(p) and any(_form_matches(p, p.split(), f) for f in _tforms(labs, al, T))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--pivot", required=True)
    ap.add_argument("--targets", nargs="*", default=["es", "ko", "zh", "ja"])
    ap.add_argument("--solvmatrix", help="default: 02-data/results/solvability/solvability_{model}.json")
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--pairs", default="02-data/inputs/generic_pairs4.jsonl",
                    help="parallel prompt pairs (not distributed)")
    ap.add_argument("--n", type=int, default=30); ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--alpha-c", type=float, default=1.0); ap.add_argument("--alpha-s", type=float, default=1.0)
    ap.add_argument("--chat", action="store_true"); ap.add_argument("--out", default="02-data/results/vt_override_layer")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    pc = json.load(open(_path(a.solvmatrix or f"02-data/results/solvability/solvability_{a.model}.json")))["per_chain"]
    rs.C = C                                                                                     
    pairs = [json.loads(x) for x in open(_path(a.pairs))]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    P = a.pivot
    print(f"[vt_override_layer] loading {a.model} pivot={P}", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    norm = _final_norm(loaded.model); lm = loaded.model.get_output_embeddings()
    if lm is None:
        raise RuntimeError(f"{a.model}: no output embeddings (lm_head); cannot lens-decode hidden states")
    res = {"model": a.model, "pivot": P, "alpha_c": a.alpha_c, "alpha_s": a.alpha_s, "by_target": {}}

    for T in a.targets:
        if T == P:
            continue
        fails = [c for c in pc if T in pc[c]["gen"] and P in pc[c]["gen"]
                 and (C[c]["answer_label_by_lang"].get(T) or "") != (C[c]["answer_label_by_lang"].get(P) or "")
                 and not rs.strict_x_correct(pc[c]["gen"][T], c, T)]
        f2 = rs.strict_forms
        A = [c for c in fails if bool(entity_match(pc[c]["gen"][P], (f2(c) or [""])[0], f2(c)))][:a.n]
        B = [c for c in fails if not bool(entity_match(pc[c]["gen"][P], (f2(c) or [""])[0], f2(c)))][:a.n]
        if len(A) < 5:
            continue

        Lpk, Lov = [], []
        for c in A:
            labs = C[c]["answer_label_by_lang"]
            toks = {"cp_P": first_answer_token_id(loaded, labs.get(P)), "cp_T": first_answer_token_id(loaded, labs.get(T)),
                    "wo_T": -1}                                                                    
            cur, depth = _curves(loaded, C[c]["prompts"][T], a.chat, toks, norm, lm)
            lp, lo = _locate(cur["cp_P"], cur["cp_T"], cur["wo_T"], depth)
            Lpk.append(lp); Lov.append(lo)
        L_peak = sorted(Lpk)[len(Lpk) // 2]; L_ovs = sorted(Lov)[len(Lov) // 2]
        f1_dead = L_ovs <= L_peak
        band = list(range(L_ovs, min(L_ovs + a.k + 1, depth - 2)))
        early = [max(1, L_peak // 2)]
        needed = sorted(set(band + [L_peak] + early))

                                                                                            
        acc = {L: None for L in needed}; npairs = 0
        for p in pairs:
            if P not in p or T not in p:
                continue
            rP_ = _resid_all(loaded, p[P], a.chat, needed)
            rT_ = _resid_all(loaded, p[T], a.chat, needed)
            for L in needed:
                d = rP_[L] - rT_[L]
                acc[L] = d.clone() if acc[L] is None else acc[L] + d
            npairs += 1
        vlang = {}
        for L in needed:
            m = acc[L] / npairs
            vlang[L] = m / (m.norm() + 1e-8)
        vT = {L: -vlang[L] for L in needed}
        print(f"[vt_override_layer] {T}: A={len(A)} B={len(B)} L_peak={L_peak} L_override={L_ovs} band={band} F1_dead={f1_dead}", flush=True)

        uc_cache = {}
        for c in list(A) + list(B):
            rP = _resid_all(loaded, C[c]["prompts"][P], a.chat, needed)
            rT = _resid_all(loaded, C[c]["prompts"][T], a.chat, needed)
            uc_cache[c] = {}
            for L in needed:
                d = rP[L] - rT[L]
                d = d - float(d @ vlang[L]) * vlang[L]
                uc_cache[c][L] = (d / (d.norm() + 1e-8)).to(loaded.device)

        def run(chains, layers, content=True, lang=True):
            hit = 0
            for c in chains:
                out = _steer_multi(loaded, C[c]["prompts"][T], a.chat, layers, uc_cache[c], vT,
                                   a.alpha_c if content else 0.0, a.alpha_s if lang else 0.0)
                hit += _recovered_T(out, C[c]["answer_label_by_lang"], C[c]["answer_aliases_by_lang"], T)
            return hit / max(1, len(chains))

        cond = {}
        cond["floor"] = sum(_recovered_T(pc[c]["gen"].get(T, ""), C[c]["answer_label_by_lang"], C[c]["answer_aliases_by_lang"], T) for c in A) / len(A)
        cond["band_two"] = run(A, band, True, True)
        cond["mid_content"] = run(A, [L_peak], True, False)                        
        cond["band_content"] = run(A, band, True, False)                      
        cond["band_lang"] = run(A, band, False, True)                              
        cond["early_two"] = run(A, early, True, True)                               
        cond["B_two"] = run(B, band, True, True) if B else 0.0                        
        res["by_target"][T] = {"L_peak": L_peak, "L_override": L_ovs, "band": band, "F1_dead": f1_dead,
                               "nA": len(A), "nB": len(B), **cond}
        (out_dir / f"lateoverride_{a.model}.json").write_text(json.dumps(res, indent=2))
        print(f"[vt_override_layer] {T}: floor={cond['floor']:.0%} BAND={cond['band_two']:.0%} mid={cond['mid_content']:.0%} "
              f"band_content={cond['band_content']:.0%} band_lang={cond['band_lang']:.0%} early={cond['early_two']:.0%} B={cond['B_two']:.0%}", flush=True)
    print("[vt_override_layer] DONE", flush=True)

@torch.no_grad()
def _steer_multi(loaded, prompt, chat, layers, uc_c, vT, alpha_c, alpha_s, max_new=10, steer_from_prompt=False):
    dl = _decoder_layers(loaded.model)
    ids = encode(loaded, prompt, chat)
    eid = getattr(loaded.model.generation_config, "eos_token_id", None)
    stop = set(eid) if isinstance(eid, (list, tuple)) else ({eid} if eid is not None else set())
    stop.add(loaded.tokenizer.eos_token_id); stop.discard(None)
    if steer_from_prompt:

        col0 = ids.shape[1] - 1; o = []
    else:
        g = loaded.model.generate(input_ids=ids, max_new_tokens=1, do_sample=False, pad_token_id=loaded.tokenizer.eos_token_id)
        ids = g[:, :ids.shape[1] + 1]; col0 = ids.shape[1] - 1
        o = [int(ids[0, -1])]                                                                         
        if o[0] in stop:
            return loaded.tokenizer.decode(o, skip_special_tokens=True)
    ucs = {L: (uc_c[L] if alpha_c else None) for L in layers}

    def mk(L):
        def hook(_m, _i, out):
            h, rest = _unwrap_layer_out(out); h = h.clone(); base = h[0, col0].norm()
            if alpha_c and ucs[L] is not None:
                h[0, col0] = h[0, col0] + alpha_c * base * ucs[L].to(h.device, h.dtype)
            if alpha_s:
                h[0, col0] = h[0, col0] + alpha_s * base * vT[L].to(h.device, h.dtype)
            return (h,) + tuple(rest) if rest is not None else h
        return hook
    for _ in range(max_new):
        hs = [dl[int(L)].register_forward_hook(mk(L)) for L in layers]
        try:
            logits = loaded.model(input_ids=ids, use_cache=False).logits
        finally:
            for hd in hs:
                hd.remove()
        nxt = int(logits[0, -1].argmax()); o.append(nxt)
        if nxt in stop:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return loaded.tokenizer.decode(o, skip_special_tokens=True)

if __name__ == "__main__":
    main()
