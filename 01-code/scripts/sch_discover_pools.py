"""Discover a cross-model general SCH pool and record the ablation dNLL diagnostic.

    python scripts/sch_discover_pools.py --model qwen25_3b --limit 12   # small smoke test
    python scripts/sch_discover_pools.py --model llama31_70             # full discovery

Pipeline: stage1 localize (attention vs MLP) -> stage2 LAHIS per language -> k-of-five
selection -> stage3 mean-ablation diagnostic. Stage 3 does not perform head-level
pruning, so the selected candidate is the intervention pool.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.models import load_model
from beacon.pipeline.stage1_patching import localize_signal
from beacon.pipeline.stage2_lahis import lahis_scores, select_general_pool
from beacon.pipeline.stage3_validate import finalize_pool

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


SCORE_VARIANTS = ("first_hop", "two_hop", "shortcut_no_r1")


def stable_sample(rows, n, salt):
    """Order-independent deterministic sample, avoiding dataset-order/chain-family bias."""
    if n is None or n <= 0 or len(rows) <= n:
        return list(rows)
    return sorted(rows, key=lambda row: hashlib.sha256(
        f"{salt}:{row['chain_id']}:{row['lang']}:{row['variant']}".encode()).digest())[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_70")
    ap.add_argument("--dataset", default="02-data/inputs/dataset.jsonl", help="full evaluation set (not distributed)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--items-per-condition", type=int, default=32,
                    help="deterministic items per language x LAHIS condition (0 = all)")
    ap.add_argument("--limit", type=int, default=None,
                    help="deprecated smoke alias for --items-per-condition")
    ap.add_argument("--validation-items-per-language", type=int, default=32,
                    help="two-hop items per language for the Stage-3 dNLL diagnostic")
    ap.add_argument("--stage1-items", type=int, default=16,
                    help="items for the stage1 localizer (it costs n_layers*2 forwards PER item --- keep small)")
    ap.add_argument("--gradient-checkpointing", action="store_true", help="needed for 70B backward on 2x80GB")
    ap.add_argument("--skip-stage1", action="store_true",
                    help="skip the costly coarse localizer; does not change Stage-2 head eligibility")
    ap.add_argument("--cache-dir", default="sch_cache",
                    help="language-level LAHIS cache; safe resume after completed languages")
    ap.add_argument("--chat", action="store_true",
                    help="use the checkpoint chat template (Qwen: enable_thinking=False)")
    args = ap.parse_args()
    if args.limit is not None:
        args.items_per_condition = args.limit

    load_env()
    cfg = load_configs()
    items = [json.loads(line) for line in open(_path(args.dataset))]
    langs = cfg.data["languages"]
    print(f"[sch_discover_pools] {len(items)} items | model={args.model} | langs={langs}")

    loaded = load_model(args.model, cfg, framework="hf_hooks")

    loc = None
    two_hop = [row for row in items if row["variant"] == "two_hop"]
    stage1_items = stable_sample(two_hop, args.stage1_items, "stage1")
    if not args.skip_stage1:
        print(f"[stage1] localizing signal (attention vs MLP) on {args.stage1_items} items ...")
        loc = localize_signal(loaded, stage1_items, chat=args.chat)
        print(f"  attention dominates at {len(loc['attn_dominates_layers'])}/"
              f"{len(loc['eligible_attention_layers'])} eligible layers "
              f"(base two_hop NLL {loc['base_nll']:.3f})")

    print("[stage2] LAHIS per language ...")
    scores_by_lang = {}
    chain_rows = {}
    for row in two_hop:
        chain_rows.setdefault(row["chain_id"], row)
    scoring_seed_rows = stable_sample(
        list(chain_rows.values()), args.items_per_condition, "lahis:parallel-chains")
    scoring_chain_ids = {row["chain_id"] for row in scoring_seed_rows}
    cache_key = hashlib.sha256(
        (args.model + f"\nchat={args.chat}\n" + "\n".join(sorted(scoring_chain_ids))).encode()
    ).hexdigest()[:16]
    cache_dir = _path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for lang in langs:
        lang_items = []
        for variant in SCORE_VARIANTS:
            cell = [it for it in items if it["chain_id"] in scoring_chain_ids
                    and it["lang"] == lang and it["variant"] == variant]
            if len(cell) != len(scoring_chain_ids):
                raise RuntimeError(f"incomplete parallel LAHIS cell: {lang}/{variant}")
            lang_items.extend(cell)
        cache_file = cache_dir / f"{args.model}_{lang}_n{args.items_per_condition}_{cache_key}.pt"
        if cache_file.exists():
            scores_by_lang[lang] = torch.load(cache_file, map_location="cpu")
            print(f"  {lang}: loaded cache {cache_file}", flush=True)
        else:
            scores_by_lang[lang] = lahis_scores(
                loaded, lang_items, gradient_checkpointing=args.gradient_checkpointing,
                chat=args.chat)
            torch.save(scores_by_lang[lang], cache_file)
            print(f"  {lang}: saved cache {cache_file}", flush=True)
        c = scores_by_lang[lang]["_counts"]
        print(f"  {lang}: scored FH={c['I_FH']} TH={c['I_TH']} SH={c['I_SH']}")

    candidate, selection_diagnostics = select_general_pool(
        scores_by_lang, cfg, return_diagnostics=True)
    min_languages = cfg.pools["general_pool"].get("min_languages") or len(langs)
    print(f"[selection] {min_languages}-of-{len(langs)} general pool: {len(candidate)} heads")
    if not candidate:
        raise RuntimeError(
            "the preregistered top-4%/4-of-5 rule produced an empty pool; "
            "report discovery failure rather than relaxing the rule post hoc")

    validation_seed_rows = stable_sample(
        [row for chain, row in chain_rows.items() if chain not in scoring_chain_ids],
        args.validation_items_per_language, "validation:parallel-chains")
    validation_chain_ids = {row["chain_id"] for row in validation_seed_rows}
    validation_items = [row for row in two_hop
                        if row["chain_id"] in validation_chain_ids and row["lang"] in langs]
    if len(validation_items) != len(validation_chain_ids) * len(langs):
        raise RuntimeError("incomplete parallel ablation-validation set")
    if {row["chain_id"] for row in validation_items} & scoring_chain_ids:
        raise RuntimeError("LAHIS scoring and ablation-validation chains overlap")
    pool, validation = finalize_pool(
        loaded, candidate, validation_items, cfg, chat=args.chat
    )
    if validation.get("random_control_status") == "infeasible":
        pool_status = "control_infeasible"
    elif (validation.get("selected_minus_random_dnll") or 0.0) > 0:
        pool_status = "positive_control_supported"
    else:
        pool_status = "positive_control_failed"

    out = _path(args.out or f"02-data/inputs/pool_{args.model}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["layer", "head", "head_id", "set_type"])
        for (L, H, hid) in pool:
            w.writerow([L, H, hid, "general"])
    manifest = {
        "model": args.model,
        "hf_id": loaded.spec.hf_id,
        "checkpoint_revision": getattr(loaded.model.config, "_commit_hash", None),
        "dataset": str(args.dataset),
        "languages": list(langs),
        "chat": args.chat,
        "enable_thinking": False if args.chat else None,
        "selection": {
            "per_language_top_n": cfg.pools["general_pool"]["per_language_top_n"],
            "per_language_top_fraction": cfg.pools["general_pool"].get("per_language_top_fraction"),
            "min_languages": cfg.pools["general_pool"].get("min_languages") or len(langs),
            "control_feasibility": {
                "enforced_during_selection": True,
                "rule": "selected heads per layer <= floor(n_query_heads / 2)",
                "purpose": "equal-sized distinct disjoint layer-matched random control",
                **selection_diagnostics,
            },
            "items_per_language_condition": args.items_per_condition,
            "score_variants": list(SCORE_VARIANTS),
            "validation_items_per_language": args.validation_items_per_language,
            "scoring_validation_chain_disjoint": True,
            "sampling": "SHA-256 deterministic order, keyed by chain/language/variant",
            "parallel_chains_across_languages": True,
            "lahis_cache_key": cache_key,
        },
        "eligible_attention_layers": scores_by_lang[next(iter(scores_by_lang))]["_eligible_layers"],
        "architecture": {
            "n_layers": loaded.spec.n_layers,
            "n_heads": loaded.spec.n_heads,
            "head_dim": loaded.spec.d_head,
            "arch": loaded.spec.arch,
            "attn_layout": loaded.spec.attn_layout,
        },
        "validation": validation,
        "pool_status": pool_status,
        "software": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
        "candidate_count": len(candidate),
        "selected_count": len(pool),
        "heads": [{"model": args.model, "layer": L, "head": H,
                   "head_id": hid, "set_type": "general"} for L, H, hid in pool],
    }
    Path(str(out) + ".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"[sch_discover_pools] candidate pool ({len(pool)} heads), status={pool_status} -> {out}")


if __name__ == "__main__":
    main()
