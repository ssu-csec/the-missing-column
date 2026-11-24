#!/usr/bin/env python3
"""Held-out causal validation for model- and language-specific SCH pools.

The language-specific pools are selected from the same LAHIS discovery split as
the general pool, but they are different head sets and therefore cannot inherit
the general pool's positive-control result.  This script reconstructs the
general discovery pipeline's disjoint validation chains and compares each
language-specific pool with an equal-sized, distinct, disjoint, layer-matched
random pool under mean ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.models import load_model
from beacon.pipeline.stage3_validate import finalize_pool

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


TARGETS = ("ko", "ja", "zh", "es")


def stable_sample(rows, n, salt):
    if n is None or n <= 0 or len(rows) <= n:
        return list(rows)
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(
            f"{salt}:{row['chain_id']}:{row['lang']}:{row['variant']}".encode()
        ).digest(),
    )[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--heads", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", nargs="+", default=list(TARGETS))
    ap.add_argument("--items-per-condition", type=int, default=32)
    ap.add_argument("--validation-items-per-language", type=int, default=32)
    ap.add_argument("--chat", action="store_true")
    args = ap.parse_args()

    load_env()
    cfg = load_configs()
    items = [json.loads(line) for line in open(_path(args.dataset))]
    two_hop = [row for row in items if row["variant"] == "two_hop"]
    by_chain = {}
    for row in two_hop:
        by_chain.setdefault(row["chain_id"], row)

    scoring_seed_rows = stable_sample(
        list(by_chain.values()), args.items_per_condition, "lahis:parallel-chains"
    )
    scoring_ids = {row["chain_id"] for row in scoring_seed_rows}
    validation_seed_rows = stable_sample(
        [row for chain, row in by_chain.items() if chain not in scoring_ids],
        args.validation_items_per_language,
        "validation:parallel-chains",
    )
    validation_ids = {row["chain_id"] for row in validation_seed_rows}
    if scoring_ids & validation_ids:
        raise RuntimeError("specific-pool scoring and validation chains overlap")

    head_doc = json.load(open(_path(args.heads)))
    heads = head_doc.get("heads", []) if isinstance(head_doc, dict) else head_doc
    pools = {
        target: sorted({
            (int(row["layer"]), int(row["head"]))
            for row in heads
            if row.get("model") == args.model
            and row.get("set_type") != "general"
            and row.get("lang") == target
        })
        for target in args.targets
    }
    missing = [target for target, pool in pools.items() if not pool]
    if missing:
        raise RuntimeError(f"missing specific pools for {missing}")

    print(f"[specific-gate] loading {args.model} chat={args.chat}", flush=True)
    loaded = load_model(args.model, cfg, framework="hf_hooks")
    results = {}
    for target in args.targets:
        validation_items = [
            row for row in two_hop
            if row["chain_id"] in validation_ids and row["lang"] == target
        ]
        if len(validation_items) != len(validation_ids):
            raise RuntimeError(
                f"incomplete validation cell {target}: "
                f"{len(validation_items)}/{len(validation_ids)}"
            )
        print(
            f"[specific-gate] {target}: heads={len(pools[target])} "
            f"items={len(validation_items)}",
            flush=True,
        )
        _pool, validation = finalize_pool(
            loaded, pools[target], validation_items, cfg, chat=args.chat
        )
        contrast = validation.get("selected_minus_random_dnll")
        supported = (
            validation.get("random_control_status") == "ok"
            and contrast is not None
            and contrast > 0
        )
        results[target] = {
            "n_heads": len(pools[target]),
            "n_validation_chains": len(validation_ids),
            "positive_control_supported": bool(supported),
            "validation": validation,
        }

    output = {
        "model": args.model,
        "set_type": "specific",
        "dataset": args.dataset,
        "heads": args.heads,
        "chat": args.chat,
        "selection_items_per_condition": args.items_per_condition,
        "validation_items_per_language": args.validation_items_per_language,
        "scoring_validation_chain_disjoint": True,
        "scoring_chain_ids_sha256": hashlib.sha256(
            "\n".join(sorted(scoring_ids)).encode()
        ).hexdigest(),
        "validation_chain_ids": sorted(validation_ids),
        "by_lang": results,
    }
    out = _path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(f"[specific-gate] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
