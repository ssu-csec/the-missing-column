"""TRANSLATION-DIFFERENCE-VECTOR (Paths Not Taken, 2505.20546) on the two-hop render wall --- the 5th intervention family.

2505.20546 (EMNLP-25 main) mends SINGLE-hop multilingual fact recall training-free: Delta^(l) = mean(h at explicit
TRANSLATION prompts) - mean(h at fact-RECALL prompts), averaged across languages/relations (language-agnostic),
added raw (no scaling) to the residual stream at ~0.75 depth (their L21/28) at the final token position. Non-Latin
conversion correctness 39.6% -> 67.7%. They are single-hop, 3B, no controls. We run their faithful construction on
our locked TWO-hop wall population (en-solvable AND T-fail) at 70B with the full control battery.

  HIT  = TDV lifts correct-in-T render on two-hop over free AND over a norm-matched random vector (cluster CI)
         -> the published key composes with two-hop; our wall adjudication completes ("opened by the frame intervention").
  MISS = single-hop positive control passes but two-hop does not move
         -> "the translation-vector key that mends single-hop recall does NOT compose with two-hop retrieval";
         the wall becomes 5-intervention-family-robust INCLUDING the published mend.

PRE-REGISTERED (before any GPU): primary config = 1x Delta @ 0.75 depth (L60 for 80-layer llama31_70_it),
continuous injection (final prompt token + every generated position = the strong version; a MISS must be
unrebuttable); robustness = {1x @L68, 2x @L60} on ko only; RTV (their recall vector) SKIPPED because our
population is en-solvable by construction (recall is controlled); STOP-1 = single-hop positive control must show
TDV recovery > rand recovery on ko single-hop failures (else intervention-dead-at-scale, report as untested); per-item
Arm0 translation gate (en_label -> T label+alias) reported alongside; primary metric = correct answer rendered
in T (output_language==T AND T-label hit, the locked render metric); cluster bootstrap by answer label.

  python scripts/translation_vector.py --model llama31_70_it --chat --targets ko ja \
      --solvmatrix 02-data/results/solvability/solvability_llama31_70_it.json
  python scripts/translation_vector.py --verify     # $0 CPU plumbing check, no model
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import entity_match, _normalize
from beacon.data.langid import output_language
from beacon.data.prompting import encode
from beacon.models import load_model
from beacon.pipeline.patching import _decoder_layers, _unwrap_layer_out

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

LANGS = ["en", "es", "ko", "zh", "ja"]
TRANS_INSTR = {                                                                         
    "ko": "Instruction: Translate the phrase into Korean.\nPhrase: {x}\nTranslation:",
    "ja": "Instruction: Translate the phrase into Japanese.\nPhrase: {x}\nTranslation:",
    "zh": "Instruction: Translate the phrase into Chinese.\nPhrase: {x}\nTranslation:",
    "es": "Instruction: Translate the phrase into Spanish.\nPhrase: {x}\nTranslation:",
}

def _label_hit(pred, label):
    pn = _normalize(pred)
    from beacon.data.evaluate import _form_matches
    return bool(pn and label) and _form_matches(pn, pn.split(), label)

def _correct_T(pred, meta, T):
    """The locked render metric: correct answer emitted in target-language form."""
    pred = (pred or "").replace("�", "").strip()
    labs = meta["answer_label_by_lang"]
    ol = output_language(pred, labs)
    amb = {"zh": "zh_ja_amb", "ja": "zh_ja_amb", "es": "en_es_amb", "en": "en_es_amb"}.get(T)
    hit = any(_label_hit(pred, f) for f in ([labs.get(T)] + (meta.get("answer_aliases_by_lang", {}).get(T) or [])) if f)
    return (ol == T or ol == amb) and hit

def _boot_cluster(rows, delta, keyf, gen, B=3000):
    """cluster bootstrap: resample answer-label clusters."""
    from collections import defaultdict
    cl = defaultdict(list)
    for r in rows:
        cl[keyf(r)].append(r)
    keys = list(cl.keys()); ds = []
    for _ in range(B):
        idx = torch.randint(0, len(keys), (len(keys),), generator=gen).tolist()
        picked = [r for j in idx for r in cl[keys[j]]]
        ds.append(sum(delta(r) for r in picked) / max(1, len(picked)))
    ds.sort()
    return [ds[int(0.025 * B)], ds[int(0.975 * B)]]

class AddVec:
    """residual-stream hook: h[:, -1] += scale * v at one decoder layer (prefill final token + each new token)."""
    def __init__(self, v, scale):
        self.v = v; self.scale = scale

    def __call__(self, _m, _i, o):
        h, rest = _unwrap_layer_out(o)
        h = h.clone()
        h[:, -1, :] = h[:, -1, :] + self.scale * self.v.to(h.device, h.dtype)
        return (h,) + tuple(rest) if rest is not None else h

@torch.no_grad()
def gen_with(loaded, prompt, chat, max_new, layer=None, vec=None, scale=1.0, processor=None):
    from transformers import LogitsProcessorList
    ids = encode(loaded, prompt, chat)
    hd = None
    if vec is not None:
        hd = _decoder_layers(loaded.model)[int(layer)].register_forward_hook(AddVec(vec, scale))
    try:
        kw = dict(input_ids=ids, max_new_tokens=max_new, min_new_tokens=2, do_sample=False,
                  pad_token_id=loaded.tokenizer.eos_token_id,
                  repetition_penalty=1.3, no_repeat_ngram_size=3)                                                    
        if processor is not None:
            kw["logits_processor"] = LogitsProcessorList([processor(ids.shape[1])])
        g = loaded.model.generate(**kw)
    finally:
        if hd is not None:
            hd.remove()
    return loaded.tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True).replace("�", "").strip()

@torch.no_grad()
def _final_h(loaded, prompt, chat, layers):
    ids = encode(loaded, prompt, chat)
    hs = loaded.model(input_ids=ids, use_cache=False, output_hidden_states=True).hidden_states
    return {L: hs[L + 1][0, -1].detach().float().cpu() for L in layers}

def _verify():
    """$0 CPU: metric, hook shape-math, cluster bootstrap."""
    meta = {"answer_label_by_lang": {"en": "Bank of Korea", "ko": "한국은행", "ja": "韓国銀行", "zh": "韩国银行", "es": "Banco de Corea"},
            "answer_aliases_by_lang": {"ko": ["한국 은행"]}}
    assert _correct_T("한국은행이다", meta, "ko") and not _correct_T("Bank of Korea", meta, "ko")
    assert not _correct_T("韓国銀行", meta, "ko")                                                          
    h = torch.zeros(1, 5, 8); v = torch.ones(8)
    av = AddVec(v, 2.0); out = av(None, None, (h.clone(),))
    assert torch.allclose(out[0][0, -1], torch.full((8,), 2.0)) and out[0][0, 0].abs().sum() == 0
    gen = torch.Generator().manual_seed(0)
    rows = [{"a": (i % 2 == 0), "b": (i % 6 == 0), "k": f"c{i%17}"} for i in range(120)]
    ci = _boot_cluster(rows, lambda r: int(r["a"]) - int(r["b"]), lambda r: r["k"], gen)
    assert ci[0] > 0
    print(f"[tv] VERIFY OK: render metric + AddVec last-pos hook + cluster bootstrap (toy CI={[round(x,3) for x in ci]}).", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--pc-only", action="store_true")                                                             
    ap.add_argument("--model"); ap.add_argument("--chat", action="store_true")
    ap.add_argument("--targets", nargs="+", default=["ko", "ja"])
    ap.add_argument("--solvmatrix")
    ap.add_argument("--layers", nargs="+", type=int, default=[60, 68])                                        
    ap.add_argument("--primary-layer", type=int, default=60)
    ap.add_argument("--scales", nargs="+", type=float, default=[1.0, 2.0])
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--n-construct", type=int, default=60)                                          
    ap.add_argument("--pc-n", type=int, default=200)                                                              
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--dataset", default="02-data/inputs/single_hop_prompts.jsonl",
                    help="second-hop (single-hop) prompts per chain and language")
    ap.add_argument("--out", default="02-data/results/tdv")
    a = ap.parse_args()
    if a.verify:
        _verify(); return
    assert a.model and (a.solvmatrix or a.pc_only)
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    SM = json.load(open(_path(a.solvmatrix)))["per_chain"] if a.solvmatrix else {}

    SH = {}
    with open(_path(a.dataset)) as f:
        for line in f:
            d = json.loads(line)
            if d.get("variant", "second_hop") == "second_hop" and d["chain_id"] in C:
                SH.setdefault(d["chain_id"], {})[d["lang"]] = d["prompt"]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    print(f"[tv] loading {a.model} chat={a.chat} | SH chains={len(SH)}", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    sd_spec = importlib.util.spec_from_file_location("sd", str(REPO_ROOT / "scripts/script_decode.py"))
    sd = importlib.util.module_from_spec(sd_spec); sd_spec.loader.exec_module(sd)

    all_ch = [c for c in C if all(C[c]["prompts"].get(L) and C[c]["answer_label_by_lang"].get(L) for L in LANGS)]

    pools = {T: ([c for c in all_ch if c in SM and SM[c]["x_correct"].get("en") and not SM[c]["x_correct"].get(T)][:a.n]
                 if not a.pc_only else [])
             for T in a.targets}
    used = set().union(*[set(p) for p in pools.values()])
    constr = [c for c in all_ch if c not in used][:a.n_construct]
    print(f"[tv] pools: " + " ".join(f"{T}={len(pools[T])}" for T in a.targets) + f" | construct={len(constr)}", flush=True)

    print("[tv] building translation-difference vectors...", flush=True)
    sums_t = {L: torch.zeros(1) for L in a.layers}; sums_r = {L: torch.zeros(1) for L in a.layers}
    nt = nr = 0
    for c in constr:
        en_lab = C[c]["answer_label_by_lang"]["en"]
        for T2 in ("ko", "ja", "zh", "es"):
            hs = _final_h(loaded, TRANS_INSTR[T2].format(x=en_lab), a.chat, a.layers)
            for L in a.layers:
                sums_t[L] = sums_t[L] + hs[L] if sums_t[L].numel() > 1 else hs[L].clone()
            nt += 1
        for L2 in LANGS:
            if c in SH and SH[c].get(L2):
                hs = _final_h(loaded, SH[c][L2], a.chat, a.layers)
                for L in a.layers:
                    sums_r[L] = sums_r[L] + hs[L] if sums_r[L].numel() > 1 else hs[L].clone()
                nr += 1
    DELTA = {L: sums_t[L] / max(1, nt) - sums_r[L] / max(1, nr) for L in a.layers}
    RESID = {L: float((sums_r[L] / max(1, nr)).norm()) for L in a.layers}                                                                
    RAND = {}
    for L in a.layers:
        rv = torch.randn(DELTA[L].shape[0], generator=gen)
        RAND[L] = rv / rv.norm() * DELTA[L].norm()
    print(f"[tv] Delta built from {nt} translation + {nr} recall forwards | " +
          " ".join(f"L{L}: |D|={float(DELTA[L].norm()):.1f} |h|={RESID[L]:.1f} rel={float(DELTA[L].norm())/max(1e-6,RESID[L]):.2%}" for L in a.layers), flush=True)

    res = {"model": a.model, "protocol": {"primary_layer": a.primary_layer, "scales": a.scales,
           "construction": f"{nt}T/{nr}R forwards, langs-averaged, raw addition, continuous last-pos injection",
           "decode": "greedy max24 rep1.3 ngram3 (scripts/48 parity)", "rtv_skipped": "population en-solvable"},
           "positive_control": {}, "by_target": {}}
    PL = a.primary_layer

    print("[tv] ===== positive control: single-hop ko =====", flush=True)
    pc_items = [c for c in all_ch if c in SH and SH[c].get("ko") and SH[c].get("en") and c not in set(constr)][:a.pc_n]
    fails = []
    for i, c in enumerate(pc_items):
        ko_free = gen_with(loaded, SH[c]["ko"], a.chat, a.max_new)
        if _correct_T(ko_free, C[c], "ko"):
            continue
        en_free = gen_with(loaded, SH[c]["en"], a.chat, a.max_new)
        en_lab = C[c]["answer_label_by_lang"]["en"]
        if _label_hit(en_free, en_lab) or entity_match(en_free, en_lab, [en_lab] + (C[c].get("answer_aliases_by_lang", {}).get("en") or [])):
            fails.append(c)                                                                                   
        if (i + 1) % 50 == 0:
            print(f"    pc-screen [{i+1}/{len(pc_items)}] fails={len(fails)}", flush=True)
    print(f"[tv] pc: single-hop en-ok&ko-fail = {len(fails)}", flush=True)

                                                                                                            
    PC_SCALES = [1.0, 2.0, 4.0]
    pc_rows = []
    for c in fails:
        r = {"chain": c, "lab": C[c]["answer_label_by_lang"]["ko"]}
        for s in PC_SCALES:
            txt = gen_with(loaded, SH[c]["ko"], a.chat, a.max_new, layer=PL, vec=DELTA[PL], scale=s)
            r[f"tv{s:g}"] = _correct_T(txt, C[c], "ko")
            if s == 1.0:
                r["tv_txt"] = txt[:100]
        r["rand"] = _correct_T(gen_with(loaded, SH[c]["ko"], a.chat, a.max_new, layer=PL, vec=RAND[PL], scale=1.0), C[c], "ko")
        pc_rows.append(r)
    if pc_rows:
        rdr = sum(r["rand"] for r in pc_rows) / len(pc_rows)
        cis = {}
        for s in PC_SCALES:
            cis[f"tv{s:g}"] = _boot_cluster(pc_rows, lambda r, k=f"tv{s:g}": int(r[k]) - int(r["rand"]), lambda r: r["lab"], gen)
        pc_pass = any(ci[0] > 0 for ci in cis.values())
        rates = {f"tv{s:g}": sum(r[f"tv{s:g}"] for r in pc_rows) / len(pc_rows) for s in PC_SCALES}
        res["positive_control"] = {"n_fail": len(pc_rows), "rates": rates, "rand_recover": rdr,
                                   "ci_by_scale": cis, "PASS": bool(pc_pass), "items": pc_rows}
        print(f"[tv] pc: " + " ".join(f"TDVx{s:g}={rates[f'tv{s:g}']:.0%}" for s in PC_SCALES) +
              f" rand={rdr:.0%} | CIs " + " ".join(f"x{s:g}[{cis[f'tv{s:g}'][0]:+.2f},{cis[f'tv{s:g}'][1]:+.2f}]" for s in PC_SCALES) +
              f" -> {'PASS (intervention alive at 70B)' if pc_pass else 'STOP-1: intervention does not mend single-hop at any tested scale'}", flush=True)
    else:
        res["positive_control"] = {"n_fail": 0, "PASS": False, "note": "no mendable single-hop failures found"}
        print("[tv] pc: NO mendable single-hop failures -> STOP-1 (no headroom to validate intervention)", flush=True)
    (out_dir / f"transvec_{a.model}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))

    for T in a.targets:
        pool = pools[T]
        allowed, fmt, eos = sd.build_allowed(loaded.tokenizer, T)
        mk = lambda plen: sd.ScriptMask(allowed, eos, 2, plen)
        rows = []
        for i, c in enumerate(pool):
            meta = C[c]; pT = C[c]["prompts"][T]; pE = C[c]["prompts"]["en"]
            en_lab = meta["answer_label_by_lang"]["en"]
            r = {"chain": c, "lab": meta["answer_label_by_lang"][T]}

            tr = gen_with(loaded, TRANS_INSTR[T].format(x=en_lab), a.chat, a.max_new)
            r["arm0_ok"] = _correct_T(tr, meta, T)

            r["nat_free"] = _correct_T(gen_with(loaded, pT, a.chat, a.max_new), meta, T)
            nt_txt = gen_with(loaded, pT, a.chat, a.max_new, layer=PL, vec=DELTA[PL], scale=1.0)
            r["nat_tdv"] = _correct_T(nt_txt, meta, T); r["nat_tdv_txt"] = nt_txt[:100]
            r["nat_rand"] = _correct_T(gen_with(loaded, pT, a.chat, a.max_new, layer=PL, vec=RAND[PL], scale=1.0), meta, T)
            if T == a.targets[0]:

                r["en_mask"] = _correct_T(gen_with(loaded, pE, a.chat, a.max_new, processor=mk), meta, T)
                r["en_tdv_mask"] = _correct_T(gen_with(loaded, pE, a.chat, a.max_new, layer=PL, vec=DELTA[PL], scale=1.0, processor=mk), meta, T)

                                                                              
                L2 = [L for L in a.layers if L != PL][0] if len(a.layers) > 1 else PL
                r["nat_tdv_L2"] = _correct_T(gen_with(loaded, pT, a.chat, a.max_new, layer=L2, vec=DELTA[L2], scale=1.0), meta, T)
                for s in (2.0, 4.0, 8.0):
                    r[f"nat_tdv_{s:g}x"] = _correct_T(gen_with(loaded, pT, a.chat, a.max_new, layer=PL, vec=DELTA[PL], scale=s), meta, T)
            rows.append(r)
            if (i + 1) % 20 == 0:
                print(f"    {T} [{i+1}/{len(pool)}]", flush=True)
        n = max(1, len(rows))
        keys = [k for k in rows[0] if k not in ("chain", "lab", "nat_tdv_txt")]
        rate = {k: sum(int(bool(r.get(k, False))) for r in rows) / n for k in keys}
        ci_free = _boot_cluster(rows, lambda r: int(bool(r["nat_tdv"])) - int(bool(r["nat_free"])), lambda r: r["lab"], gen)
        ci_rand = _boot_cluster(rows, lambda r: int(bool(r["nat_tdv"])) - int(bool(r["nat_rand"])), lambda r: r["lab"], gen)
        hit = (ci_free[0] > 0) and (ci_rand[0] > 0) and res["positive_control"].get("PASS", False)
        res["by_target"][T] = {"n": n, "rate": rate, "ci_tdv_minus_free": ci_free, "ci_tdv_minus_rand": ci_rand,
                               "KEY_HIT": bool(hit), "items": rows}
        (out_dir / f"transvec_{a.model}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"[tv] {T}: arm0={rate['arm0_ok']:.0%} | nat: free={rate['nat_free']:.0%} TDV={rate['nat_tdv']:.0%} "
              f"rand={rate['nat_rand']:.0%} | dFree CI[{ci_free[0]:+.2f},{ci_free[1]:+.2f}] dRand CI[{ci_rand[0]:+.2f},{ci_rand[1]:+.2f}]"
              + (f" | en_mask={rate.get('en_mask',0):.0%} en_tdv_mask={rate.get('en_tdv_mask',0):.0%} "
                 f"L2={rate.get('nat_tdv_L2',0):.0%} 2x={rate.get('nat_tdv_2x',0):.0%} "
                 f"4x={rate.get('nat_tdv_4x',0):.0%} 8x={rate.get('nat_tdv_8x',0):.0%}" if T == a.targets[0] else "")
              + f" -> {'KEY-HIT' if hit else 'MISS'}", flush=True)
    print("[tv] DONE", flush=True)

if __name__ == "__main__":
    main()
