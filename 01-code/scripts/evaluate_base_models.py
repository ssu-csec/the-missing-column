"""Evaluate a base model over the dataset, then build the failure / matched-success subsets.

    python scripts/evaluate_base_models.py --model qwen25_3b --limit 200   # smoke test on a small model
    python scripts/evaluate_base_models.py --model llama31_70               # full evaluation

Reads 02-data/inputs/dataset.jsonl (the full evaluation set, not distributed), writes evaluated.jsonl + per-language subset jsonls, and
cross-checks subset sizes against configs/beacon.yaml. The subset partition is on two_hop correctness:
failure = EN-success and X-incorrect; matched_success = EN and X both correct.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import platform

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.models import load_model
from beacon.data.evaluate import evaluate_items
from beacon.data.subsets import build_subsets, crosscheck_sizes

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_70")
    ap.add_argument("--dataset", default="02-data/inputs/dataset.jsonl")
    ap.add_argument("--out-dir", default="02-data/inputs")
    ap.add_argument("--limit", type=int, default=None, help="subsample items (smoke test)")
    ap.add_argument("--variants", nargs="*", default=None,
                    help="restrict eval to these query variants (e.g. two_hop) --- the failure/success "
                         "subset build only needs two_hop, so this is ~5x less generation")
    ap.add_argument("--target-langs", nargs="*", default=["ko", "zh", "ja", "es"])
    ap.add_argument("--chat", action="store_true",
                    help="use the checkpoint chat template (Qwen: enable_thinking=False)")
    args = ap.parse_args()

    load_env()
    cfg = load_configs()
    items = [json.loads(line) for line in open(_path(args.dataset))]
    if args.variants:
        items = [it for it in items if it["variant"] in args.variants]
    if args.limit:
        items = items[:args.limit]
    print(f"[eval] {len(items)} items | model={args.model}")

    # eval only generates --- sdpa is much faster than eager and needs no attention weights
    loaded = load_model(args.model, cfg, framework="hf_hooks", attn_implementation="sdpa")
    evaluate_items(loaded, items, batch_report=max(1, len(items) // 10), chat=args.chat)

    out_dir = _path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"evaluated_{args.model}.jsonl", "w") as fh:
        for it in items:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    metadata = {
        "model": args.model,
        "hf_id": loaded.spec.hf_id,
        "checkpoint_revision": getattr(loaded.model.config, "_commit_hash", None),
        "dataset": args.dataset,
        "variants": args.variants,
        "max_new_tokens": 32,
        "decoding": "greedy",
        "chat": args.chat,
        "enable_thinking": False if args.chat else None,
        "software": {
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
            "transformers": __import__("transformers").__version__,
        },
    }
    (out_dir / f"evaluated_{args.model}.metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    # per (lang, variant) accuracy --- the premise check (is EN two_hop high? do the hops work?)
    import collections
    accmap = collections.defaultdict(lambda: [0, 0])
    for it in items:
        k = (it["lang"], it["variant"])
        accmap[k][0] += it["eval"]["correct"]
        accmap[k][1] += 1
    print("[eval] accuracy by lang x variant:")
    for (lang, variant), (c, n) in sorted(accmap.items()):
        print(f"    {lang:3s} {variant:12s} {c}/{n} = {c / n:.3f}")

    print("[subsets] failure = ROUTING failure (X two_hop wrong, X first+second hop correct)")
    for lang in args.target_langs:
        subsets = build_subsets(items, args.model, lang)
        raw, kept = subsets["raw_failure_count"], len(subsets["failure"])
        print(f"  [{lang}] routing-failures {kept} of {raw} raw two_hop-failures "
              f"(dropped {raw - kept} capability failures); matched_success {len(subsets['matched_success'])}")
        crosscheck_sizes(subsets, args.model, lang, cfg)
        for name in ("failure", "matched_success"):
            with open(out_dir / f"subset_{args.model}_{lang}_{name}.jsonl", "w") as fh:
                for it in subsets[name]:
                    fh.write(json.dumps(it, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
