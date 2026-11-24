"""Pivot-language solvability matrix. A cross-lingual gap measured only against English counts
target-language failures that are English-solvable; if a model's pivot is not English, that count undercounts
failures that are solvable in another language. This script therefore measures solvability in every language,
per chain, so the gap can be computed against any pivot. For each chain the two-hop question is asked in all of
{en,es,ko,zh,ja} and, per prompt language P, the script records: (a) any_ok[P] = the answer names the correct
entity in any registered form, in any language (content; the 'P-solvable' probe), and (b) x_correct[P] = the
answer is in P's own registered target-language form (used to define a strict target-language failure).

Derived quantity (offline, from the saved matrix): for target T, among chains with x_correct[T]==False, the
fraction that are solvable from each candidate pivot language.

    python scripts/solvability_matrix.py --model qwen3_32b --chat
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import entity_match, entity_outcome
from beacon.data.prompting import generate_answer
from beacon.models import load_model

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

ALL = ["en", "es", "ko", "zh", "ja"]

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3_32b")
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--langs", nargs="*", default=ALL, help="prompt languages to generate for")
    ap.add_argument("--targets", nargs="*", default=["en", "es", "ko", "zh", "ja"], help="target langs (incl en: are English failures solvable in other langs?)")
    ap.add_argument("--pivots", nargs="*", default=["en", "es", "zh"],
                    help="candidate pivot langs to score solvability. es is the CONTROL: if English-solvable >> "
                         "Spanish-solvable the effect is English-specific; if ~equal it is a high-resource/Latin "
                         "effect, not latent-English.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/solvability")
    args = ap.parse_args()

    load_env()
    cfg = load_configs()
    C = json.load(open(_path(args.chains)))
    chains = list(C)
    if args.limit:
        chains = chains[:args.limit]
    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[matrix] loading {args.model} | chains={len(chains)} langs={args.langs} ...", flush=True)
    loaded = load_model(args.model, cfg, framework="hf_hooks", attn_implementation="eager")

    per_chain = {}
    for i, c in enumerate(chains):
        forms = [C[c]["answer_label_by_lang"].get(k) for k in ALL]
        forms += [a for al in C[c]["answer_aliases_by_lang"].values() for a in (al or [])]
        forms = [f for f in forms if f]
        rec = {"any_ok": {}, "x_correct": {}, "gen": {}}
        for lang in args.langs:
            p = C[c]["prompts"].get(lang)
            if not p:
                continue
            g = generate_answer(loaded, p, args.chat, max_new=32)
            item = {"lang": lang, "answer_label": C[c]["answer_label_by_lang"].get(lang),
                    "answer_aliases": C[c]["answer_aliases_by_lang"].get(lang) or []}
            rec["gen"][lang] = g
            rec["any_ok"][lang] = bool(entity_match(g, forms[0] if forms else "", forms))
            rec["x_correct"][lang] = entity_outcome(g, item) == "x_correct"
        per_chain[c] = rec
        if (i + 1) % 50 == 0:
            print(f"    [{i + 1}/{len(chains)}]", flush=True)

    gap = {}
    for T in args.targets:
        fails = [c for c in per_chain if T in per_chain[c]["x_correct"] and not per_chain[c]["x_correct"][T]]
        row = {"n_fail": len(fails)}
        for P in args.pivots:
            if P == T:
                continue
            sol = sum(1 for c in fails if per_chain[c]["any_ok"].get(P))
            row[f"{P}_solvable"] = sol
            row[f"frac_{P}_solvable"] = sol / max(1, len(fails))

        anyp = sum(1 for c in fails if any(per_chain[c]["any_ok"].get(P) for P in args.pivots if P != T))
        row["any_pivot_solvable"] = anyp
        row["frac_any_pivot_solvable"] = anyp / max(1, len(fails))
        gap[T] = row

    out = {"model": args.model, "chat": args.chat, "chains": args.chains, "limit": args.limit,
           "max_new": 32, "targets": args.targets, "pivots": args.pivots,
           "gap_by_pivot": gap, "per_chain": per_chain}
    (out_dir / f"solvability_{args.model}.json").write_text(json.dumps(out, indent=2))

    print(f"\n[matrix] ===== {args.model}: of TARGET failures, fraction solvable per PIVOT language =====", flush=True)
    hdr = "  target  n_fail " + " ".join(f"{P}-solv" for P in args.pivots) + "  any-pivot"
    print(hdr, flush=True)
    for T, r in gap.items():
        cells = " ".join(f"{r.get(f'frac_{P}_solvable', float('nan')):6.0%}" for P in args.pivots)
        print(f"  {T:>6} {r['n_fail']:>6}  {cells}   {r['frac_any_pivot_solvable']:5.0%}", flush=True)

if __name__ == "__main__":
    main()
