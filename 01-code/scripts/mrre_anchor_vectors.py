"""MRRE anchor vectors: MRRE-style target-language anchoring applied to cross-lingual two-hop failures.

MRRE (Li et al., ACL-26) restores target-language output on math chain-of-thought with a two-stage injection: a
mid-layer cross-lingual enhancement vector (toward English), then a late-layer target-language output anchoring
vector v_a = h(P_T) - h(P_E), where P_T / P_E are language-forcing prompts ("answer in <T>", written in T / in
English). The injection is norm-preserving at the last token of each step (alpha1=1.0, alpha2=0.75 in MRRE).

This script applies the intervention to cross-lingual two-hop failures and scores whether it produces the
correct answer entity in the designated form for ko/ja/zh/es. Arms per target T: (a) anchor-only v_a@late;
(b) full MRRE v_r@mid + v_a@late; both are compared with a norm-matched random direction at the same
configuration. Protocol: untreated 32-token entity-free floors, calibration/evaluation split, scoring of
designated-form realization / content (entity) / es-repair, and removal of U+FFFD replacement characters.

Adaptation: v_r is built from parallel prompt hidden states (en - fail_lang over the chain prompts), not from
generated reasoning responses, because two-hop answers are too short for a response-level v_r. v_a follows MRRE:
fixed language-forcing prompt pairs, last-token hidden-state difference.

    python scripts/mrre_anchor_vectors.py --model gemma4_31b
    python scripts/mrre_anchor_vectors.py --model qwen36_27b --chat
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import _form_matches, _normalize
from beacon.data.langid import output_language
from beacon.data.prompting import encode
from beacon.hooks import _decoder_layers
from beacon.models import load_model
from beacon.pipeline.patching import _unwrap_layer_out

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def _label_hit(pred, label):
    pn = _normalize(pred)
    return bool(pn and label) and _form_matches(pn, pn.split(), label)

FORCE = {
    "en": "Please answer the above question in English.",
    "es": "Por favor, responde a la pregunta anterior en español.",
    "ko": "위 질문에 한국어로 답하세요.",
    "ja": "上記の質問に日本語で答えてください。",
    "zh": "请用中文回答上述问题。",
}

@torch.no_grad()
def _last_tok_hidden(loaded, text, chat, layer):
    ids = encode(loaded, text, chat)
    hs = loaded.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
    return hs[layer + 1][0, -1].detach().float().cpu()

@torch.no_grad()
def _build_va(loaded, T, chat, layer):
    """MRRE Eq.4: v_a = h(P_T) - h(P_E) at the last token (fixed language-forcing prompts)."""
    d = _last_tok_hidden(loaded, FORCE[T], chat, layer) - _last_tok_hidden(loaded, FORCE["en"], chat, layer)
    return d / (d.norm() + 1e-8)

@torch.no_grad()
def _build_vr(loaded, C, chains, FL, chat, layer, n=60):
    """Adapted MRRE Eq.2-3: mean(h(en prompt) - h(FL prompt)) at the last token over chain prompts."""
    acc, k = None, 0
    for c in chains[:n]:
        if not (C[c]["prompts"].get("en") and C[c]["prompts"].get(FL)):
            continue
        d = _last_tok_hidden(loaded, C[c]["prompts"]["en"], chat, layer)\
            - _last_tok_hidden(loaded, C[c]["prompts"][FL], chat, layer)
        acc = d.clone() if acc is None else acc + d
        k += 1
    if acc is None:
        raise SystemExit(f"[mrre] no chains with both en and {FL} prompts among first {min(n, len(chains))}")
    m = acc / max(1, k)
    return m / (m.norm() + 1e-8)

@torch.no_grad()
def _decode_mrre(loaded, prompt, chat, injections, max_new=32):
    """Greedy decode; at each step add alpha*v at the LAST token of the given layers, RENORMALIZED to the
    pre-injection l2 norm (MRRE Eq.5-6 norm-preserving injection). KV-cached: each position is steered once at
    its own step and the steered value persists in the cache = MRRE's persistent-injection semantics."""
    dl = _decoder_layers(loaded.model)

    def mk(v, alpha):
        def hook(_m, _i, out):
            h, rest = _unwrap_layer_out(out)
            h = h.clone()
            last = h[0, -1]
            n0 = last.norm()
            upd = last + alpha * n0 * v.to(h.device, h.dtype)
            h[0, -1] = upd * (n0 / (upd.norm() + 1e-8))                                       
            return (h,) + tuple(rest) if rest is not None else h
        return hook

    gcfg = getattr(loaded.model, "generation_config", None)
    ge = getattr(gcfg, "eos_token_id", None)
    stop = set([ge] if isinstance(ge, int) else list(ge or []))
    stop.add(loaded.tokenizer.eos_token_id)
    stop.discard(None)

    ids = encode(loaded, prompt, chat)
    out, past, cur = [], None, ids
    hs = [dl[int(L)].register_forward_hook(mk(v, a)) for (L, v, a) in injections]
    try:
        for _ in range(max_new):
            o = loaded.model(input_ids=cur, past_key_values=past, use_cache=True)
            past = o.past_key_values
            nxt = int(o.logits[0, -1].argmax())
            out.append(nxt)
            if nxt in stop:
                break
            cur = torch.tensor([[nxt]], device=ids.device)
    finally:
        for h in hs:
            h.remove()
    return loaded.tokenizer.decode(out, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--fail-lang", default="es")
    ap.add_argument("--targets", nargs="*", default=["ko", "ja", "zh", "es"])
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--alpha2s", nargs="*", type=float, default=[0.75, 1.5], help="anchoring strengths (paper: 0.75)")
    ap.add_argument("--alpha1", type=float, default=1.0, help="enhancement strength for the full-MRRE arm")
    ap.add_argument("--n-calib", type=int, default=30)
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/interventions")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    out_dir = _path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    FL = a.fail_lang
    print(f"[mrre] loading {a.model} (chat={a.chat})", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    nL = loaded.spec.n_layers
    l1, l2s = int(0.71 * nL), sorted({int(0.82 * nL), int(0.90 * nL)})                                         
    print(f"[mrre] n_layers={nL} l1(enhance)={l1} l2(anchor) candidates={l2s}", flush=True)

    chains_all = [c for c in C if C[c]["prompts"].get(FL) and C[c]["answer_label_by_lang"].get(FL)]
    va = {(T, L2): _build_va(loaded, T, a.chat, L2) for T in a.targets for L2 in l2s}

    fails, floors = [], {}
    for c in chains_all:
        if len(fails) >= a.n_calib + a.n_eval:
            break
        floor = _decode_mrre(loaded, C[c]["prompts"][FL], a.chat, [], max_new=a.max_new)
        labs, al = C[c]["answer_label_by_lang"], C[c]["answer_aliases_by_lang"]
        ok = output_language(floor, labs) == FL and any(
            _form_matches(_normalize(floor), _normalize(floor).split(), f)
            for f in [labs.get(FL), *(al.get(FL) or [])] if f)
        ent = any(_label_hit(floor, labs.get(L2)) for L2 in ("es", "en", "ko", "zh", "ja") if labs.get(L2))
        if ok or ent:
            continue
        fails.append(c); floors[c] = floor
    calib, evalset = fails[:a.n_calib], fails[a.n_calib:]
    if len(evalset) < 20:                                                                                     
        k = max(10, len(fails) // 3)                                                                        
        calib, evalset = fails[:k], fails[k:]
        print(f"[mrre] WARNING: few fails ({len(fails)}); calib shrunk to {len(calib)}", flush=True)
    if not evalset:
        raise SystemExit(f"[mrre] ABORT: only {len(fails)} genuine {FL}-fails --- not enough for calib+eval")
    print(f"[mrre] genuine {FL}-fails: {len(fails)} -> calib={len(calib)} eval={len(evalset)}", flush=True)

    vr_chains = [c for c in chains_all if c not in set(evalset)][:60]
    vr = _build_vr(loaded, C, vr_chains, FL, a.chat, l1)
    g = torch.Generator().manual_seed(7)
    rv = torch.randn(vr.shape[0], generator=g); rv = rv / (rv.norm() + 1e-8)

    AMB = {"zh": "zh_ja_amb", "ja": "zh_ja_amb", "es": "en_es_amb", "en": "en_es_amb"}

    def scored(pred, c, lang):
        pred = pred.replace("�", "").strip()
        labs = C[c]["answer_label_by_lang"]
        ol = output_language(pred, labs)
        ent = any(_label_hit(pred, labs.get(L2)) for L2 in ("es", "en", "ko", "zh", "ja") if labs.get(L2))
        return {"out": ol, "entity": ent,
                "correct": (ol == lang or ol == AMB.get(lang)) and _label_hit(pred, labs.get(lang)),
                "es_repair": (ol == FL or ol == AMB.get(FL)) and _label_hit(pred, labs.get(FL)), "txt": pred}

    def run_arm(items, T, mode, L2, a2, vec=None):
        inj = []
        if mode in ("full",):
            inj.append((l1, vr, a.alpha1))
        inj.append((L2, vec if vec is not None else va[(T, L2)], a2))
        return [scored(_decode_mrre(loaded, C[c]["prompts"][FL], a.chat, inj, max_new=a.max_new), c, T) for c in items]

    print(f"\n[mrre] ===== CALIBRATION (n={len(calib)}) =====", flush=True)
    best = {}
    for T in a.targets:
        rows = []
        for mode in ("anchor", "full"):
            for L2 in l2s:
                for a2 in a.alpha2s:
                    rs = run_arm(calib, T, mode, L2, a2)
                    cor = sum(r["correct"] for r in rs) / max(1, len(calib))
                    ent = sum(r["entity"] for r in rs) / max(1, len(calib))
                    amb = sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rs)
                    rows.append({"mode": mode, "L2": L2, "a2": a2, "correct": cor, "entity": ent, "amb": amb})
                    print(f"  {T} {mode:>6} L2={L2} a2={a2:<5} correct={cor:.1%} entity={ent:.1%}", flush=True)
        rows.sort(key=lambda r: (r["correct"], r["entity"]), reverse=True)
        best[T] = rows[0]
        print(f"  {T} BEST -> {best[T]['mode']} L2={best[T]['L2']} a2={best[T]['a2']}", flush=True)

    print(f"\n[mrre] ===== TEST (disjoint n={len(evalset)}) =====", flush=True)
    res = {"model": a.model, "protocol": {"floor_max_new": a.max_new, "entity_free_floor": True,
           "l1": l1, "alpha1": a.alpha1, "adaptation": "v_r from prompt-level diff (see docstring)",
           "vr_chains": vr_chains},
           "best": best, "n_calib": len(calib), "n_eval": len(evalset),
           "calib_chains": calib, "eval_chains": evalset, "floors": floors, "eval": {}}
    print(f"  {'T':>4} {'config':>20} | {'render':>7} {'out=en':>7} {'entity':>7} {'es-fix':>7} | {'rand:rend':>9} {'rand:ent':>8}", flush=True)
    for T in a.targets:
        b = best[T]
        rs = run_arm(evalset, T, b["mode"], b["L2"], b["a2"])
        rr = run_arm(evalset, T, b["mode"], b["L2"], b["a2"], vec=rv)
        n = max(1, len(evalset))
        res["eval"][T] = {"steer_items": rs, "rand_items": rr,
                          "amb_steer": sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rs),
                          "amb_rand": sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rr)}
        print(f"  {T:>4} {b['mode']:>7} L2={b['L2']} a2={b['a2']:<4} | {sum(r['correct'] for r in rs)/n:>6.1%} "
              f"{sum(r['out']=='en' for r in rs)/n:>6.1%} {sum(r['entity'] for r in rs)/n:>6.1%} "
              f"{sum(r['es_repair'] for r in rs)/n:>6.1%} | {sum(r['correct'] for r in rr)/n:>8.1%} "
              f"{sum(r['entity'] for r in rr)/n:>7.1%}", flush=True)
    (out_dir / f"mrre_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[mrre] DONE", flush=True)

if __name__ == "__main__":
    main()
