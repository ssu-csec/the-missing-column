"""Strict re-scoring of the solvability matrices. The alias union contains generic single-token currency-type
words ('shilling', 'chelin', 'franc', ...), so entity_match counts a wrong-country, right-type answer ('Kenyan
shilling' for gold 'Ugandan shilling') as correct. Strict forms keep the full primary labels plus only specific
aliases (>=2 Latin tokens, all-caps codes >=3, or CJK >=2 chars), dropping bare generic type words. Every
solvability_*.json is re-scored offline from the saved generation text (no model is loaded), and the
lenient-vs-strict gap-by-pivot table is printed.

    python scripts/solvability_rescore.py 02-data/results/solvability/solvability_*.json   # from the repository root
"""
from __future__ import annotations

import glob
import json
import sys

from beacon.config import REPO_ROOT
from beacon.data.evaluate import _form_matches, _has_cjk, _normalize, entity_match

LANGS = ["en", "es", "ko", "zh", "ja"]
C = json.load(open(REPO_ROOT.parent / "02-data/inputs/chain_lang_prompts.json"))

def strict_forms(chain):
    prim = [C[chain]["answer_label_by_lang"].get(k) for k in LANGS]
    aliases = [a for al in C[chain]["answer_aliases_by_lang"].values() for a in (al or [])]
    keep = [f for f in prim if f]
    for a in aliases:
        if not a:
            continue
        if _has_cjk(a):
            if len(a) >= 2:
                keep.append(a)
        elif len(a.split()) >= 2:                                                                           
            keep.append(a)
        elif a.isupper() and len(a) >= 3:                         
            keep.append(a)

    return list(dict.fromkeys(keep))                          

def strict_x_correct(pred, chain, lang):
    """target-surface success under STRICT forms: CJK target needs a strict CJK form; es/en any strict form."""
    p = _normalize(pred)
    if not p:
        return False
    toks = p.split()
    if lang in ("ko", "zh", "ja"):

                                                                        
        lab = C[chain]["answer_label_by_lang"].get(lang) or ""
        return bool(lab) and _form_matches(p, toks, lab)
    forms = strict_forms(chain)
    return any(_form_matches(p, toks, f) for f in forms)

def rescore(pc):
    """recompute any_ok[lang] AND x_correct[lang] strictly from saved gen text (num AND denom consistent)."""
    out = {}
    for c, rec in pc.items():
        forms = strict_forms(c)
        ok, xc = {}, {}
        for lang, g in rec.get("gen", {}).items():
            ok[lang] = bool(entity_match(g, forms[0] if forms else "", forms))
            xc[lang] = strict_x_correct(g, c, lang)
        out[c] = {"any_ok": ok, "x_correct": xc}
    return out

def gap_by_pivot(pc, pivots=("en", "es", "zh")):
    rows = {}
    for T in LANGS:
        fails = [c for c in pc if pc[c]["x_correct"].get(T) is False]
        row = {"n": len(fails)}
        for P in pivots:
            if P == T:
                continue
            row[P] = sum(1 for c in fails if pc[c]["any_ok"].get(P)) / max(1, len(fails))
        rows[T] = row
    return rows

def main():
    files = []
    for a in sys.argv[1:]:
        files += glob.glob(a)
    for f in sorted(set(files)):
        d = json.load(open(f))
        pc = d.get("per_chain")
        if not pc:
            continue
        tag = d.get("tag") or d.get("model", f.split("/")[-1])
        lenient = {c: {"any_ok": pc[c].get("any_ok", {}), "x_correct": pc[c].get("x_correct", {})} for c in pc}
        strict = rescore(pc)
        gl, gs = gap_by_pivot(lenient), gap_by_pivot(strict)
        print(f"\n===== {tag}  (lenient -> STRICT, non-en targets) =====")
        print(f"  {'tgt':>4} {'nfail':>5} | {'en L->S':>12} | {'es L->S':>12} | {'zh L->S':>12}")
        for T in ["es", "ko", "zh", "ja"]:
            def cell(P):
                lv, sv = gl[T].get(P), gs[T].get(P)
                return f"{lv:.0%}->{sv:.0%}" if lv is not None else "   --"
            print(f"  {T:>4} {gl[T]['n']:>5} | {cell('en'):>12} | {cell('es'):>12} | {cell('zh'):>12}")

if __name__ == "__main__":
    main()
