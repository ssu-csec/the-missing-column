"""SCRIPT-CONSTRAINED DECODE + correctness gate --- the one un-tried intervention on the render wall.

The render wall's dead fixes (es-steer, ablate-English) all acted on the MID-STACK content/language direction ---
but the paper's thesis is that CONTENT is retrieved, so content-side intervention was doomed. This test acts on the
LATE/unembedding step instead: on cross-lingual failures (en-solvable AND T-fail), FORCE the decode to emit only
target-SCRIPT tokens (Hangul for ko; Kana|Han for ja) and ask whether the CORRECT entity emerges.

  HIT  = script-forced decode surfaces the correct target-language answer >> a random-other-entity control
         -> the content->script map EXISTS but was suppressed by a soft decode prior = an OVERRIDABLE wall = key.
  MISS = script-forced decode yields transliteration/garbage (correct ~= control ~= 0)
         -> a genuine content->script mapping gap = a SHARPER negative.

Either outcome strengthens the paper; the asymmetry is the whole point.

  python scripts/script_decode.py --model llama31_70_it --chat --targets ko ja --n 150 \
      --solvmatrix 02-data/results/solvability/solvability_llama31_70_it.json
  python scripts/script_decode.py --verify      # $0 CPU: mask + scoring plumbing, no model
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import entity_match
from beacon.data.langid import output_language
from beacon.data.prompting import encode
from beacon.models import load_model

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

LANGS = ["en", "es", "ko", "zh", "ja"]

def _has_hangul(s): return any(("가" <= c <= "힣") or ("ᄀ" <= c <= "ᇿ") or ("ㄱ" <= c <= "ㆎ") for c in s)
def _has_kana(s): return any(("぀" <= c <= "ゟ") or ("゠" <= c <= "ヿ") or c == "ー" for c in s)
def _has_han(s): return any("一" <= c <= "鿿" for c in s)
def _has_latin(s): return any(("A" <= c <= "Z") or ("a" <= c <= "z") for c in s)

def _in_target_script(s, T):
    if T == "ko":
        return _has_hangul(s)
    if T == "ja":
        return _has_kana(s) or _has_han(s)                                  
    if T == "zh":
        return _has_han(s) and not _has_kana(s) and not _has_hangul(s)
    raise ValueError(T)

def _is_target_letter(c, T):
    if T == "ko":
        return _has_hangul(c)
    if T == "ja":
        return _has_kana(c) or _has_han(c)
    if T == "zh":
        return _has_han(c) and not _has_kana(c) and not _has_hangul(c)
    raise ValueError(T)

def build_allowed(tokenizer, T):
    """Allowed token ids under the T-script constraint. A token is CONTENT-allowed iff every LETTER it contains is
    a target-script letter (so Latin/Cyrillic/Arabic/Greek content is all forbidden --- the earlier bug allowed any
    non-Latin script); a token is FMT-allowed iff it contains NO letter of any script (pure space/punct/digit/symbol).
    Plus EOS. This forces the model to render in the target script or emit nothing but formatting."""
    import unicodedata
    allowed, fmt = set(), set()
    for tid in range(len(tokenizer)):
        s = tokenizer.decode([tid])
        if not s:
            continue
        letters = [c for c in s if unicodedata.category(c).startswith("L")]
        if not letters:
            fmt.add(tid)                                                                                         
        elif all(_is_target_letter(c, T) for c in letters):
            allowed.add(tid)                                                                  
    eos = tokenizer.eos_token_id
    allowed |= fmt
    if eos is not None:
        allowed.add(eos)
    return allowed, fmt, eos

class ScriptMask:
    """transformers LogitsProcessor: keep only allowed ids; block EOS until min_new script tokens are produced."""
    def __init__(self, allowed_ids, eos_id, min_new, prompt_len):
        self.allowed = torch.tensor(sorted(allowed_ids), dtype=torch.long)
        self.eos_id = eos_id
        self.min_new = min_new
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        keep = torch.full_like(scores, float("-inf"))
        keep[:, self.allowed.to(scores.device)] = 0.0
        scores = scores + keep
        if self.eos_id is not None and (input_ids.shape[1] - self.prompt_len) < self.min_new:
            scores[:, self.eos_id] = float("-inf")
        return scores

def _aliases(meta, L):
    out = []
    lab = meta["answer_label_by_lang"].get(L)
    if lab:
        out.append(lab)
    out += (meta.get("answer_aliases_by_lang", {}).get(L) or [])
    return out

@torch.no_grad()
def _generate(loaded, ids, max_new, min_new, processor=None):
    from transformers import LogitsProcessorList
    kw = dict(input_ids=ids, max_new_tokens=max_new, min_new_tokens=min_new, do_sample=False,
              pad_token_id=loaded.tokenizer.eos_token_id,

                                                                                                   
              repetition_penalty=1.3, no_repeat_ngram_size=3)
    if processor is not None:
        kw["logits_processor"] = LogitsProcessorList([processor])
    g = loaded.model.generate(**kw)
    return loaded.tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True).replace("�", "").strip()

def _boot(rows, delta, gen, B=3000):
    n = len(rows); ds = []
    for _ in range(B):
        idx = torch.randint(0, n, (n,), generator=gen).tolist()
        ds.append(sum(delta(rows[j]) for j in idx) / n)
    ds.sort()
    return [ds[int(0.025 * B)], ds[int(0.975 * B)]]

def _verify():
    """$0 CPU check: script classifiers, mask plumbing, scoring, bootstrap --- no model."""
    assert _has_hangul("서울") and not _has_hangul("Seoul")
    assert _has_kana("トヨタ") and _has_han("日本") and not _has_latin("日本")
    assert _in_target_script("은행", "ko") and not _in_target_script("banco", "ko")
    assert _in_target_script("日本銀行", "ja") and _in_target_script("東京", "zh") is True

    assert _is_target_letter("한", "ko") and not _is_target_letter("Б", "ko") and not _is_target_letter("ا", "ko") and not _is_target_letter("a", "ko")
    assert _is_target_letter("日", "ja") and _is_target_letter("ト", "ja") and not _is_target_letter("Б", "ja")

    sm = ScriptMask({2, 5, 7}, eos_id=2, min_new=3, prompt_len=4)
    scores = torch.zeros(1, 10)
    ids = torch.zeros(1, 5, dtype=torch.long)                                          
    out = sm(ids, scores.clone())
    assert out[0, 5] == 0 and out[0, 7] == 0 and out[0, 0] == float("-inf") and out[0, 2] == float("-inf")
    ids2 = torch.zeros(1, 8, dtype=torch.long)                                   
    out2 = sm(ids2, scores.clone())
    assert out2[0, 2] == 0

    assert entity_match("서울은행이다", "서울은행", ["서울은행"]) and not entity_match("서울은행이다", "부산", ["부산"])
    gen = torch.Generator().manual_seed(0)
    rows = [{"hit": (i % 2 == 0), "ctrl": (i % 7 == 0)} for i in range(120)]
    ci = _boot(rows, lambda r: int(r["hit"]) - int(r["ctrl"]), gen)
    assert ci[0] > 0, ci
    print(f"[sd] VERIFY OK: script classifiers + mask (eos-gate) + scoring + bootstrap plumbed (toy hit-ctrl CI={[round(x,3) for x in ci]}).", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--targets", nargs="+", default=["ko", "ja"])
    ap.add_argument("--solvmatrix", help="default: 02-data/results/solvability/solvability_{model}.json")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--min-new", type=int, default=2)
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/rivals")
    a = ap.parse_args()
    if a.verify:
        _verify(); return
    assert a.model and a.solvmatrix, "--model and --solvmatrix required"
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    SM = json.load(open(_path(a.solvmatrix or f"02-data/results/solvability/solvability_{a.model}.json")))["per_chain"]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    print(f"[sd] loading {a.model} chat={a.chat}", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")

    res = {"model": a.model, "n_req": a.n, "max_new": a.max_new, "by_target": {}}
    for T in a.targets:

        pool = [c for c in C
                if all(C[c]["prompts"].get(L) and C[c]["answer_label_by_lang"].get(L) for L in LANGS)
                and c in SM and SM[c]["x_correct"].get("en") and not SM[c]["x_correct"].get(T)]
        pool = pool[:a.n]
        allowed, fmt, eos = build_allowed(loaded.tokenizer, T)
        print(f"[sd] T={T}: population={len(pool)} | allowed-script-toks={len(allowed)-len(fmt)-(1 if eos is not None else 0)} fmt-toks={len(fmt)}", flush=True)
        rows = []
        for i, c in enumerate(pool):
            meta = C[c]
            ids_T = encode(loaded, C[c]["prompts"][T], a.chat)                                             
            ids_en = encode(loaded, C[c]["prompts"]["en"], a.chat)                                                       
            nat_txt = _generate(loaded, ids_T, a.max_new, a.min_new, processor=None)
            sm = ScriptMask(allowed, eos, a.min_new, ids_en.shape[1])
            key_txt = _generate(loaded, ids_en, a.max_new, a.min_new, processor=sm)                                
            cd = pool[(i + 1) % len(pool)]                                                                        
            ctrl_aliases = _aliases(C[cd], T)
            gold_T = _aliases(meta, T)
            any_alias = sum((_aliases(meta, L) for L in LANGS), [])
            rows.append({
                "chain": c,

                "nat_render_T": output_language(nat_txt, meta["answer_label_by_lang"]) == T,
                "nat_correct_T": bool(gold_T and entity_match(nat_txt, gold_T[0], gold_T)),
                "nat_entity_any": bool(entity_match(nat_txt, any_alias[0] if any_alias else "", any_alias)),

                "key_correct_T": bool(gold_T and entity_match(key_txt, gold_T[0], gold_T)),
                "key_ctrl": bool(ctrl_aliases and entity_match(key_txt, ctrl_aliases[0], ctrl_aliases)),
                "key_render_lang": output_language(key_txt, meta["answer_label_by_lang"]) == T,
                "nat_txt": nat_txt[:120], "key_txt": key_txt[:120],
            })
            if (i + 1) % 20 == 0:
                print(f"    T={T} [{i+1}/{len(pool)}]", flush=True)
        n = max(1, len(rows))
        rate = {k: sum(int(bool(r[k])) for r in rows) / n for k in
                ("nat_render_T", "nat_correct_T", "nat_entity_any", "key_correct_T", "key_ctrl", "key_render_lang")}
        ci_key = _boot(rows, lambda r: int(bool(r["key_correct_T"])) - int(bool(r["key_ctrl"])), gen)
        hit = (rate["key_correct_T"] > 0.10) and (ci_key[0] > 0)
        res["by_target"][T] = {"n": n, "rate": rate, "ci_key_correct_minus_ctrl": ci_key,
                               "KEY_HIT": bool(hit), "items": rows}
        (out_dir / f"scriptdecode_{a.model}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
        verdict = ("KEY-HIT (EN-content renders CORRECTLY in forced T-script >> control -> render map exists, wall = soft prior)" if hit
                   else "MISS (forced T-script does not surface the correct answer -> genuine content->script gap)")
        print(f"[sd] T={T}: NAT(prompt-{T}): render_T={rate['nat_render_T']:.0%} correct_T={rate['nat_correct_T']:.0%} entity_any={rate['nat_entity_any']:.0%} | "
              f"KEY(prompt-en+force-{T}): correct={rate['key_correct_T']:.0%} ctrl={rate['key_ctrl']:.0%} "
              f"(Δ CI[{ci_key[0]:+.2f},{ci_key[1]:+.2f}]) render_lang={rate['key_render_lang']:.0%} -> {verdict}", flush=True)
    print("[sd] DONE", flush=True)

if __name__ == "__main__":
    main()
