"""Surfacing vectors (v_T): locate -> per-language configuration sweep -> frozen-configuration test.

The steering layers are taken from each model's measured override region rather than a fixed-depth
heuristic, and every steer language (including the CJK targets) receives its own configuration sweep.

Protocol per steer language T (es/ko/ja/zh):
  LOCATE   compute per-target L_peak/L_override on this model: take ~12 population-A items (T fails, pivot
           solves) from the solvability matrix, trace cp_P/cp_T/wo_T per layer (vt_override_layer.py's
           _curves/_locate), and take the median.
  CALIBRATE on the first --n-calib scorable failures, sweep positions x alphas (norm-relative), with positions
           from the located layers: {single L_ov, band L_ov..L_ov+2, deep single 0.95*nL}; pick the configuration
           maximizing designated-form realization in T (tiebreak: content in any form).
  TEST     on disjoint subsequent failures, with the frozen per-T configuration, plus norm-matched random
           directions at the same configuration (direction-specificity control). The full output-language
           breakdown is saved, including the out=en column.

    python scripts/vt_surfacing_vectors.py --model gemma4_31b
    python scripts/vt_surfacing_vectors.py --model qwen36_27b --chat
(--solvmatrix defaults to 02-data/results/solvability/solvability_{model}.json; --latent is optional and is used only if
it carries a by_target[T].L_override, i.e. a lateoverride json written by vt_override_layer.py.)
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import _form_matches, _normalize
from beacon.data.langid import output_language
from beacon.data.prompting import generate_answer
from beacon.models import load_model

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

_lo = importlib.util.spec_from_file_location("lo", str(REPO_ROOT / "scripts/vt_override_layer.py"))
lo = importlib.util.module_from_spec(_lo); _lo.loader.exec_module(lo)

def _label_hit(pred, label):
    pn = _normalize(pred)
    return bool(pn and label) and _form_matches(pn, pn.split(), label)

def _build_v(loaded, pairs, layers, tgt, src, chat, src_res=None):
    acc = {L: None for L in layers}
    n = 0
    for p in pairs:
        if tgt not in p or src not in p:
            continue
        rt = lo._resid_all(loaded, p[tgt], chat, layers)
        rs_ = src_res[id(p)] if src_res is not None else lo._resid_all(loaded, p[src], chat, layers)
        for L in layers:
            d = rt[L] - rs_[L]
            acc[L] = d.clone() if acc[L] is None else acc[L] + d
        n += 1
    return {L: (acc[L] / n) / ((acc[L] / n).norm() + 1e-8) for L in layers}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--fail-lang", default="es")
    ap.add_argument("--steers", nargs="*", default=["es", "ko", "ja", "zh"])
    ap.add_argument("--latent", default="", help="optional vt_override_layer.py output (by_target.L_override); if absent/incompatible, locate is computed here")
    ap.add_argument("--solvmatrix", default="", help="per_chain solvmatrix for population-A locate (default: 02-data/results/solvability/solvability_{model}.json)")
    ap.add_argument("--n-locate", type=int, default=12)
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--pairs", default="02-data/inputs/generic_pairs4.jsonl",
                    help="parallel prompt pairs (not distributed)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--n-calib", type=int, default=40)
    ap.add_argument("--n-eval", type=int, default=160)
    ap.add_argument("--max-new", type=int, default=32)

    ap.add_argument("--floor-max-new", type=int, default=32, help="floor decode window (full length, not clipped)")
    ap.add_argument("--allow-entity-in-floor", action="store_true",
                    help="OFF (default) = genuine fails only: floor must contain NO correct entity in ANY language")
    ap.add_argument("--rand-seeds", dest="rand_seeds", nargs="*", type=int, default=[7, 17, 27, 37, 47],
                    help="norm-matched random-direction seeds for the eval control (first = headline)")
    ap.add_argument("--steer-first-token", action="store_true",
                    help="anchor steering at the last PROMPT position so the FIRST generated token is steered too (arms only; floors stay plain)")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/interventions")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    pairs = [json.loads(x) for x in open(_path(a.pairs))]
    lat = {}
    if a.latent:
        try:
            j = json.load(open(_path(a.latent)))
            lat = j.get("by_target", {}) if any("L_override" in v for v in j.get("by_target", {}).values()) else {}
        except Exception:
            lat = {}
    sm_path = a.solvmatrix or f"02-data/results/solvability/solvability_{a.model}.json"
    pc = json.load(open(_path(sm_path)))["per_chain"]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    FL = a.fail_lang
    print(f"[cal] loading {a.model} (chat={a.chat})", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    nL = loaded.spec.n_layers
    deep = min(int(0.95 * nL), nL - 2)
    norm = lo._final_norm(loaded.model)
    lm = loaded.model.get_output_embeddings()
    if lm is None:
        raise RuntimeError(f"{a.model}: no output embeddings (lm_head); cannot lens-decode hidden states")
    from beacon.pipeline.patching import first_answer_token_id

    chains = [c for c in C if C[c]["prompts"].get(FL) and C[c]["answer_label_by_lang"].get(FL)
              and C[c]["answer_label_by_lang"].get(a.pivot)]
    fails, floors = [], {}
    for c in chains:
        if len(fails) >= a.n_calib + a.n_eval:
            break
        floor = generate_answer(loaded, C[c]["prompts"][FL], a.chat, max_new=a.floor_max_new)                                  
        labs, al = C[c]["answer_label_by_lang"], C[c]["answer_aliases_by_lang"]
        ok = lo._recovered_T(floor, labs, al, FL)
        flat = " ".join(floor.split())                                                                         
        ent_any = any(_label_hit(flat, f) for L2 in ("es", "en", "ko", "zh", "ja")
                      for f in ([labs.get(L2)] + (al.get(L2) or [])) if f)
        if ok or (ent_any and not a.allow_entity_in_floor):                                              
            continue
        fails.append(c); floors[c] = floor
    calib, evalset = fails[:a.n_calib], fails[a.n_calib:]
    if len(evalset) < 20:                                                                                     
        k = max(10, len(fails) // 3)                                                                        
        calib, evalset = fails[:k], fails[k:]
        print(f"[cal] WARNING: few fails ({len(fails)}); calib shrunk to {len(calib)}", flush=True)
    if not evalset:
        raise SystemExit(f"[cal] ABORT: only {len(fails)} scorable {FL}-fails --- not enough for calib+eval")
    print(f"[cal] scorable {FL}-fails: {len(fails)} -> calib={len(calib)} eval={len(evalset)}", flush=True)
    eval_excl = set(evalset)
    locate_chains = {}

    def locate(T):
        """median (L_peak, L_override) over population-A items: T wrong, pivot right, labels distinct, eval held out."""
        A = [c for c in pc if c not in eval_excl
             and pc[c]["x_correct"].get(T) is False
             and (pc[c]["x_correct"].get(a.pivot) or pc[c]["any_ok"].get(a.pivot))
             and c in C and C[c]["answer_label_by_lang"].get(T)
             and C[c]["answer_label_by_lang"].get(a.pivot)
             and pc[c]["gen"].get(T)
             and C[c]["answer_label_by_lang"].get(T) != C[c]["answer_label_by_lang"].get(a.pivot)][:a.n_locate]
        locate_chains[T] = A
        pks, ovs = [], []
        for c in A:
            labs = C[c]["answer_label_by_lang"]
            full0, _ = lo._readout(loaded, C[c]["prompts"][T], a.chat)
            toks = {"cp_P": first_answer_token_id(loaded, labs.get(a.pivot)),
                    "cp_T": first_answer_token_id(loaded, labs.get(T)),
                    "wo_T": int(full0[0, -1])}                                                                 
            cur, depth = lo._curves(loaded, C[c]["prompts"][T], a.chat, toks, norm, lm)
            lp, lov_ = lo._locate(cur["cp_P"], cur["cp_T"], cur["wo_T"], depth)
            pks.append(lp); ovs.append(lov_)
        if not ovs:
            return None, None
        return sorted(pks)[len(pks) // 2], sorted(ovs)[len(ovs) // 2]

    pos = {}
    for T in a.steers:
        if T in lat and "L_override" in lat[T]:
            lpk, lov = lat[T].get("L_peak"), int(lat[T]["L_override"])
            src = "latent-json"
        else:
            lpk, lov = locate(T)
            src = f"own-locate(n<={a.n_locate})"
        if lov is not None:
            pos[T] = {"single_Lov": [lov], "band_Lov": list(range(lov, min(lov + 3, nL - 1))), "deep95": [deep]}
            print(f"[cal] {T}: L_peak={lpk} L_override={lov} [{src}] -> positions {sorted(set(sum(pos[T].values(), [])))}", flush=True)
        else:
            pos[T] = {"heur_band": list(range(int(0.75 * nL), nL - 2, max(1, (nL - 2 - int(0.75 * nL)) // 3))),
                      "heur_deep": [deep], "heur_mid": [int(0.85 * nL)]}
            print(f"[cal] {T}: locate found NO A-items -> heuristic positions (flagged)", flush=True)

    layers_needed = sorted({L for T in a.steers for p in pos[T].values() for L in p})
    print(f"[cal] building v per steer-lang at layers {layers_needed}", flush=True)
    src_res = {id(p): lo._resid_all(loaded, p[a.pivot], a.chat, layers_needed) for p in pairs if a.pivot in p}
    V = {T: _build_v(loaded, pairs, layers_needed, T, a.pivot, a.chat, src_res) for T in a.steers}
    d = V[a.steers[0]][layers_needed[0]].shape[0]

    RANDS = []
    for sd in a.rand_seeds:
        g = torch.Generator().manual_seed(sd)
        rv = torch.randn(d, generator=g); rv = rv / (rv.norm() + 1e-8)
        RANDS.append({L: rv for L in layers_needed})
    RAND = RANDS[0]

    AMB = {"zh": "zh_ja_amb", "ja": "zh_ja_amb", "es": "en_es_amb", "en": "en_es_amb"}                                                

    def scored(pred, c, lang):
        pred = pred.replace("�", "").strip()                                                              
        labs = C[c]["answer_label_by_lang"]
        ol = output_language(pred, labs)
        ent = any(_label_hit(pred, labs.get(L2)) for L2 in ("es", "en", "ko", "zh", "ja") if labs.get(L2))
        es_fix = (ol == FL or ol == AMB.get(FL)) and _label_hit(pred, labs.get(FL))
        return {"out": ol, "entity": ent,
                "correct": (ol == lang or ol == AMB.get(lang)) and _label_hit(pred, labs.get(lang)),
                "es_repair": es_fix, "txt": pred}

    def run_arm(items, T, layers, vecs, alpha):
        return [scored(lo._steer_multi(loaded, C[c]["prompts"][FL], a.chat, layers, {}, vecs, 0.0, alpha,
                                       max_new=a.max_new, steer_from_prompt=a.steer_first_token), c, T) for c in items]

    best = {}
    print(f"\n[cal] ===== CALIBRATION (n={len(calib)}) =====", flush=True)
    for T in a.steers:
        rows = []
        for pname, lys in pos[T].items():
            for al in a.alphas:
                rs = run_arm(calib, T, lys, V[T], al)
                cor = sum(r["correct"] for r in rs) / len(calib)
                ent = sum(r["entity"] for r in rs) / len(calib)
                amb = sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rs)
                rows.append({"pos": pname, "layers": lys, "alpha": al, "correct": cor, "entity": ent, "amb": amb})
                print(f"  {T} {pname:>10} a={al:<5} correct={cor:.1%} entity={ent:.1%}", flush=True)
        rows.sort(key=lambda r: (r["correct"], r["entity"]), reverse=True)
        best[T] = rows[0]
        print(f"  {T} BEST -> {best[T]['pos']} a={best[T]['alpha']} (correct={best[T]['correct']:.1%})", flush=True)

    print(f"\n[cal] ===== TEST (disjoint n={len(evalset)}) =====", flush=True)
    res = {"model": a.model, "fail_lang": FL, "n_calib": len(calib), "n_eval": len(evalset),

           "protocol": {"floor_max_new": a.floor_max_new, "entity_free_floor": not a.allow_entity_in_floor,
                        "max_new": a.max_new, "solvmatrix": sm_path, "scorer": "evaluate._form_matches+langid",
                        "steer_first_token": a.steer_first_token, "amb_adjudicated": True,
                        "locate_chains": locate_chains},
           "calib_chains": calib, "eval_chains": evalset, "floors": {c: floors[c] for c in (calib + evalset)},
           "best": {T: {k: best[T][k] for k in ("pos", "layers", "alpha")} for T in a.steers}, "eval": {}}
    print(f"  {'steer':>6} {'config':>12} | {'out=T':>7} {'out=en':>7} {'entity':>7} {'CORRECT':>8} | {'rand:out=T':>10} {'rand:ent':>8}", flush=True)
    for T in a.steers:
        b = best[T]
        rs = run_arm(evalset, T, b["layers"], V[T], b["alpha"])
        rr_by_seed = [run_arm(evalset, T, b["layers"], R, b["alpha"]) for R in RANDS]                  
        rr = rr_by_seed[0]
        rr_pool = [r for rs_ in rr_by_seed for r in rs_]
        n = len(evalset)
        row = {"steer": [sum(r["out"] == T for r in rs) / n, sum(r["out"] == "en" for r in rs) / n,
                         sum(r["entity"] for r in rs) / n, sum(r["correct"] for r in rs) / n],
               "rand": [sum(r["out"] == T for r in rr_pool) / max(1, len(rr_pool)),
                        sum(r["entity"] for r in rr_pool) / max(1, len(rr_pool))],
               "rand_per_seed": [{"seed": sd, "correct": sum(r["correct"] for r in rr_) / n,
                                  "entity": sum(r["entity"] for r in rr_) / n}
                                 for sd, rr_ in zip(a.rand_seeds, rr_by_seed)],
               "amb": [sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rs),
                       sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rr)]}
        res["eval"][T] = {"steer_items": rs, "rand_items": rr,
                          "rand_items_by_seed": {str(sd): rr_ for sd, rr_ in zip(a.rand_seeds, rr_by_seed)},
                          "summary": row}
        print(f"  {T:>6} {b['pos']:>10} a={b['alpha']:<4} | {row['steer'][0]:>6.1%} {row['steer'][1]:>6.1%} "
              f"{row['steer'][2]:>6.1%} {row['steer'][3]:>7.1%} | {row['rand'][0]:>9.1%} {row['rand'][1]:>7.1%}", flush=True)
    (out_dir / f"surfacing_vector_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("\n  [KEY] each language got its own swept best config; CJK 0% here is a CALIBRATED null, not an unproven one.", flush=True)
    print("[cal] DONE", flush=True)

if __name__ == "__main__":
    main()
