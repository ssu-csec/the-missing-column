"""Language-neuron amplification: output-language control via language-specific MLP neurons.

Reconstructs the intervention of Sun et al. (AAAI-26), who control output language by amplifying
language-specific neurons of the target language and suppressing those of the query language, and applies it
systematically across targets with calibration, a disjoint evaluation split, and a random-neuron control:

  IDENTIFY  language-specific MLP neurons per language L in {es,ko,ja,zh}: activation differential on the
            5-language parallel corpus (mean activation on L-prompts minus the max mean activation on other
            languages), deep layer band, with a vocabulary-projection filter (the neuron must promote L-script
            marker tokens: u_L . down_proj[:,i] > 0).
  INTERVENE during decoding at all positions: amplify the L-group x(1+amp) and optionally suppress the
            prompt-language (es) group x supp. amp/supp are swept on a calibration split.
  SCORE     on untreated floors decoded with max_new=32 (joined decode, entity-free failures only):
            designated-form realization in L, content in any form (entity_any), es-repair, and the full
            output-language distribution; a random-neuron control (size- and layer-matched) at the same
            factors; disjoint evaluation split.

    python scripts/language_neurons.py --model gemma4_31b
    python scripts/language_neurons.py --model qwen36_27b --chat
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

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

def _label_hit(pred, label):
    pn = _normalize(pred)
    return bool(pn and label) and _form_matches(pn, pn.split(), label)

def _script_bucket(tok: str) -> str:
    """First strong script in the decoded token: hangul/kana/han/latin_es/other."""
    for ch in tok:
        if "가" <= ch <= "힣":
            return "hangul"
        if "぀" <= ch <= "ヿ":
            return "kana"
        if "一" <= ch <= "鿿":
            return "han"
        if ch in "ñáéíóúü¿¡ÑÁÉÍÓÚÜ":
            return "latin_es"
    return "other"

LANG_BUCKET = {"ko": "hangul", "ja": "kana", "zh": "han", "es": "latin_es"}

@torch.no_grad()
def _marker_token_ids(loaded, lang: str, cap: int = 4000):
    """Vocab token ids whose surface form carries the language's script marker."""
    tk = loaded.tokenizer
    want = LANG_BUCKET[lang]
    ids = []
    for t in range(tk.vocab_size):
        s = tk.convert_ids_to_tokens(t)
        if s is None:
            continue
        try:
            s = tk.convert_tokens_to_string([s])
        except Exception:
            continue
        if _script_bucket(s) == want:
            ids.append(t)
            if len(ids) >= cap:
                break
    return ids

@torch.no_grad()
def _lang_activation_means(loaded, pairs, band, langs, chat):
    """Mean down_proj-INPUT activation per (layer, neuron) per language over the parallel corpus."""
    dl = _decoder_layers(loaded.model)
    sums = {L: {lg: None for lg in langs} for L in band}
    cnt = {lg: 0 for lg in langs}
    store = {}

    def mk(L):
        def hook(_m, args):
            store[L] = args[0][0].float().mean(0).cpu()                                       
        return hook

    for p in pairs:
        for lg in langs:
            if lg not in p:
                continue
            hs = [dl[int(L)].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in band]
            try:
                ids = encode(loaded, p[lg], chat)
                loaded.model(input_ids=ids, use_cache=False)
            finally:
                for h in hs:
                    h.remove()
            for L in band:
                sums[L][lg] = store[L].clone() if sums[L][lg] is None else sums[L][lg] + store[L]
            cnt[lg] += 1
    return {L: {lg: sums[L][lg] / max(1, cnt[lg]) for lg in langs} for L in band}, cnt

def _select_lang_neurons(loaded, means, band, lang, langs, K, marker_ids):
    """Top-K (layer, neuron) by activation differential, filtered to neurons that PROMOTE the language's script."""
    dl = _decoder_layers(loaded.model)
    unemb = (loaded.model.get_output_embeddings() or loaded.model.get_input_embeddings()).weight
    u = unemb[torch.tensor(marker_ids, device=unemb.device)].float().mean(0)
    cand = []
    for L in band:
        others = torch.stack([means[L][lg] for lg in langs if lg != lang]).amax(0)
        diff = means[L][lang] - others
        W = dl[int(L)].mlp.down_proj.weight
        proj = (u.to(W.device) @ W.float()).cpu()                                                   
        score = diff * (proj > 0)                                                                              
        topv, topi = torch.topk(score, min(4 * K, score.shape[0]))
        cand += [(float(v), int(L), int(i)) for v, i in zip(topv.tolist(), topi.tolist()) if v > 0]
    cand.sort(reverse=True)
    by = {}
    for _v, L, i in cand[:K]:
        by.setdefault(L, []).append(i)
    return by

def _mk_prehook_all(neurons, factor):
    idx = torch.tensor(neurons, dtype=torch.long)

    def hook(_m, args):
        a = args[0].clone()
        a[:, :, idx.to(a.device)] *= factor
        return (a,) + tuple(args[1:])
    return hook

@torch.no_grad()
def _decode_with_groups(loaded, prompt, chat, groups, max_new=32):
    """Greedy CACHED decode with (group -> factor) MLP prehooks applied at ALL positions; joined decode.

    KV-cache equivalence: the prehook is a deterministic position-uniform multiplier, so each position passes
    through the hooked MLP exactly once (at its own step) --- identical to the full re-forward. generate()'s
    stop set is generation_config.eos_token_id, which covers gemma's <end_of_turn> (tokenizer.eos misses it)."""
    dl = _decoder_layers(loaded.model)
    ids = encode(loaded, prompt, chat)
    hs = []
    for by, factor in groups:
        for L, ns in by.items():
            hs.append(dl[int(L)].mlp.down_proj.register_forward_pre_hook(_mk_prehook_all(ns, factor)))
    try:
        out = loaded.model.generate(input_ids=ids, max_new_tokens=max_new, do_sample=False,
                                    pad_token_id=loaded.tokenizer.eos_token_id)
    finally:
        for h in hs:
            h.remove()
    return loaded.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pivot", default="en")
    ap.add_argument("--fail-lang", default="es")
    ap.add_argument("--targets", nargs="*", default=["ko", "ja", "zh", "es"])
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--pairs", default="02-data/inputs/generic_pairs4.jsonl",
                    help="parallel prompt pairs (not distributed)")
    ap.add_argument("--k-neurons", type=int, default=400)
    ap.add_argument("--amps", nargs="*", type=float, default=[4.0, 9.0], help="amplify factor = 1+amp (Sun ~x10)")
    ap.add_argument("--suppress", type=float, default=0.1, help="prompt-language group multiplier (1.0 = off)")
    ap.add_argument("--n-calib", type=int, default=30)
    ap.add_argument("--n-eval", type=int, default=120)
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/interventions")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    pairs = [json.loads(x) for x in open(_path(a.pairs))]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    FL = a.fail_lang
    print(f"[language_neurons] loading {a.model} (chat={a.chat})", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    nL = loaded.spec.n_layers
    band = list(range(int(0.5 * nL), nL - 1))                                                                   
    langs = sorted(set(a.targets + [FL, a.pivot]))                                                                   

    print(f"[language_neurons] activation means over {len(pairs)}x{len(langs)} corpus, band {band[0]}..{band[-1]}", flush=True)
    means, cnt = _lang_activation_means(loaded, pairs, band, langs, a.chat)
    print(f"[language_neurons] corpus counts: {cnt}", flush=True)

    groups, marker_counts = {}, {}
    g = torch.Generator().manual_seed(7)
    for L in a.targets:
        mk_ids = _marker_token_ids(loaded, L)
        marker_counts[L] = len(mk_ids)
        groups[L] = _select_lang_neurons(loaded, means, band, L, langs, a.k_neurons, mk_ids)
        nsel = sum(len(v) for v in groups[L].values())
        print(f"[language_neurons] {L}: markers={len(mk_ids)} neurons={nsel} layers={sorted(groups[L])[:6]}...", flush=True)

    rand_groups = {}
    for L in a.targets:
        rg = {}
        for lay, ns in groups[L].items():
            inter = _decoder_layers(loaded.model)[int(lay)].mlp.down_proj.weight.shape[1]
            rg[lay] = torch.randperm(inter, generator=g)[:len(ns)].tolist()
        rand_groups[L] = rg
    es_group = groups.get(FL) or _select_lang_neurons(loaded, means, band, FL, langs, a.k_neurons,
                                                      _marker_token_ids(loaded, FL))

    chains = [c for c in C if C[c]["prompts"].get(FL) and C[c]["answer_label_by_lang"].get(FL)
              and C[c]["answer_label_by_lang"].get(a.pivot)]
    fails, floors = [], {}
    for c in chains:
        if len(fails) >= a.n_calib + a.n_eval:
            break
        floor = _decode_with_groups(loaded, C[c]["prompts"][FL], a.chat, [], max_new=a.max_new)
        labs, al = C[c]["answer_label_by_lang"], C[c]["answer_aliases_by_lang"]
        ok = output_language(floor, labs) == FL and any(
            _form_matches(_normalize(floor), _normalize(floor).split(), f)
            for f in [labs.get(FL), *(al.get(FL) or [])] if f)
        ent_any = any(_label_hit(floor, labs.get(L2)) for L2 in ("es", "en", "ko", "zh", "ja") if labs.get(L2))
        if not ok and not ent_any:                                                                            
            fails.append(c); floors[c] = floor
    calib, evalset = fails[:a.n_calib], fails[a.n_calib:]
    if len(evalset) < 20:                                                                                     
        k = max(10, len(fails) // 3)                                                                        
        calib, evalset = fails[:k], fails[k:]
        print(f"[language_neurons] WARNING: few fails ({len(fails)}); calib shrunk to {len(calib)}", flush=True)
    if not evalset:
        raise SystemExit(f"[language_neurons] ABORT: only {len(fails)} genuine {FL}-fails --- not enough for calib+eval")
    print(f"[language_neurons] genuine {FL}-fails (32-tok floors, entity-free): {len(fails)} -> calib={len(calib)} eval={len(evalset)}", flush=True)

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

    def run_arm(items, T, by, amp, supp):
        gs = [(by, 1.0 + amp)]
        if supp != 1.0 and T != FL:
            gs.append((es_group, supp))
        return [scored(_decode_with_groups(loaded, C[c]["prompts"][FL], a.chat, gs, max_new=a.max_new), c, T)
                for c in items]

    print(f"\n[language_neurons] ===== CALIBRATION (n={len(calib)}) =====", flush=True)
    best = {}
    for T in a.targets:
        rows = []
        for amp in a.amps:
            for supp in ([a.suppress, 1.0] if T != FL else [1.0]):
                rs = run_arm(calib, T, groups[T], amp, supp)
                cor = sum(r["correct"] for r in rs) / max(1, len(calib))
                ent = sum(r["entity"] for r in rs) / max(1, len(calib))
                rows.append({"amp": amp, "supp": supp, "correct": cor, "entity": ent})
                print(f"  {T} amp=x{1+amp:<4} supp={supp:<4} correct={cor:.1%} entity={ent:.1%}", flush=True)
        rows.sort(key=lambda r: (r["correct"], r["entity"]), reverse=True)
        best[T] = rows[0]
        print(f"  {T} BEST -> amp=x{1+best[T]['amp']} supp={best[T]['supp']}", flush=True)

    print(f"\n[language_neurons] ===== TEST (disjoint n={len(evalset)}) =====", flush=True)
    res = {"model": a.model, "k": a.k_neurons, "band": [band[0], band[-1]], "marker_counts": marker_counts,
           "best": best, "n_calib": len(calib), "n_eval": len(evalset),

           "protocol": {"floor_max_new": a.max_new, "max_new": a.max_new, "entity_free_floor": True,
                        "scorer": "evaluate._form_matches+langid", "amps": a.amps, "suppress": a.suppress,
                        "pivot": a.pivot, "fail_lang": FL, "chains": a.chains, "pairs": a.pairs, "rand_seed": 7},
           "calib_chains": calib, "eval_chains": evalset,
           "floors": {c: floors[c] for c in (calib + evalset)}, "eval": {}}
    print(f"  {'target':>6} {'config':>14} | {'render':>7} {'out=en':>7} {'entity':>7} {'es-fix':>7} | {'rand:render':>11} {'rand:ent':>8}", flush=True)
    for T in a.targets:
        b = best[T]
        rs = run_arm(evalset, T, groups[T], b["amp"], b["supp"])
        rr = run_arm(evalset, T, rand_groups[T], b["amp"], b["supp"])
        n = max(1, len(evalset))
        res["eval"][T] = {"steer_items": rs, "rand_items": rr,
                          "amb_steer": sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rs),
                          "amb_rand": sum(r["out"] in ("zh_ja_amb", "en_es_amb") for r in rr)}
        print(f"  {T:>6} x{1+b['amp']:<5} s={b['supp']:<4} | {sum(r['correct'] for r in rs)/n:>6.1%} "
              f"{sum(r['out']=='en' for r in rs)/n:>6.1%} {sum(r['entity'] for r in rs)/n:>6.1%} "
              f"{sum(r['es_repair'] for r in rs)/n:>6.1%} | {sum(r['correct'] for r in rr)/n:>10.1%} "
              f"{sum(r['entity'] for r in rr)/n:>7.1%}", flush=True)
    (out_dir / f"language_neurons_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[language_neurons] DONE", flush=True)

if __name__ == "__main__":
    main()
