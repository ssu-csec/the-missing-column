"""UNCLIPPED wall-pool membership RE-SCREEN (verification batch item 5; kills the 12-token-screen attack).

The campaign's eval pools (en-solvable AND T-fail) inherited membership from a solvmatrix whose generations were
decoded at 12 tokens (documented clip class). This re-screens the ACTUAL wall-pool chains per model with clean
32-token greedy decodes and the locked render metric (correct answer in language-L form, script-gated), then
reports membership churn: of the chains in the original pools, how many remain (a) en-solvable and (b) T-fail
for T in {ko, ja} --- i.e., does the wall population survive an unclipped screen.

  python scripts/rescreen.py --model gemma4_31b --pool <pool chain-ID list>
  python scripts/rescreen.py --verify
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.models import load_model

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

LANGS = ["en", "ko", "ja"]

_tv = importlib.util.spec_from_file_location("tv", str(REPO_ROOT / "scripts/translation_vector.py"))
tv = importlib.util.module_from_spec(_tv); _tv.loader.exec_module(tv)

def _verify():
    meta = {"answer_label_by_lang": {"en": "Bank of Korea", "ko": "한국은행", "ja": "韓国銀行", "zh": "韩国银行", "es": "Banco de Corea"},
            "answer_aliases_by_lang": {}}
    assert tv._correct_T("The answer is Bank of Korea.", meta, "en")
    assert not tv._correct_T("한국은행", meta, "en") and tv._correct_T("한국은행", meta, "ko")
    print("[rs] VERIFY OK: locked metric reused from scripts/translation_vector.py.", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--model"); ap.add_argument("--chat", action="store_true")
    ap.add_argument("--pool", help="JSON list of the original pool chain IDs (not distributed)")                                                                    
    ap.add_argument("--solvmatrix", help="default: 02-data/results/solvability/solvability_{model}.json")                                                                  
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--out", default="02-data/results/rescreen")
    a = ap.parse_args()
    if a.verify:
        _verify(); return
    assert a.model and a.pool and a.solvmatrix
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    pool = [c for c in json.load(open(_path(a.pool))) if c in C]
    SM = json.load(open(_path(a.solvmatrix or f"02-data/results/solvability/solvability_{a.model}.json")))["per_chain"]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[rs] {a.model} chat={a.chat} | pool={len(pool)} chains x {LANGS}", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")

    res = {"model": a.model, "max_new": a.max_new, "per_chain": {}}
    for i, c in enumerate(pool):
        row = {}
        for L in LANGS:
            txt = tv.gen_with(loaded, C[c]["prompts"][L], a.chat, a.max_new)
            row[L] = {"correct": bool(tv._correct_T(txt, C[c], L)), "txt": txt[:160]}
        row["old"] = SM.get(c, {}).get("x_correct", {})
        res["per_chain"][c] = row
        if (i + 1) % 25 == 0:
            print(f"    [{i+1}/{len(pool)}]", flush=True)
            (out_dir / f"rescreen_{a.model}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out_dir / f"rescreen_{a.model}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    for T in ("ko", "ja"):
        orig = [c for c in pool if SM.get(c, {}).get("x_correct", {}).get("en") and not SM.get(c, {}).get("x_correct", {}).get(T)]
        surv = [c for c in orig if res["per_chain"][c]["en"]["correct"] and not res["per_chain"][c][T]["correct"]]
        lost_en = [c for c in orig if not res["per_chain"][c]["en"]["correct"]]
        now_T_ok = [c for c in orig if res["per_chain"][c][T]["correct"]]
        print(f"[rs] {a.model} T={T}: original-pool {len(orig)} -> survive-unclipped {len(surv)} ({len(surv)/max(1,len(orig)):.0%}) | "
              f"lost en-solvable {len(lost_en)} | now T-correct {len(now_T_ok)}", flush=True)
    print("[rs] DONE", flush=True)

if __name__ == "__main__":
    main()
