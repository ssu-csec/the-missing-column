#!/usr/bin/env python3
"""Hop-level solvability of target-language failures.

Two-hop composition can fail without any cross-lingual cause. This script
reports, for each checkpoint and evaluated language, the share of items that
the model fails under a prompt in that language but answers under an English
prompt. A high share bounds how much of the observed gap can be attributed to
two-hop composition or missing knowledge rather than to language.

Source: 02-data/results/solvability/solvability_<model>.json, field
gap_by_pivot[lang]["frac_en_solvable"].

Usage:
    python3 hop_solvability.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ACCESS = os.path.abspath(os.path.join(os.environ.get("RESULTS_DIR") or os.path.join(HERE, "..", "..", "02-data", "results"), "solvability"))

CHECKPOINTS = [
    ("llama31_70", "Llama-3.1-70B"),
    ("llama31_70_it", "Llama-3.1-70B-It"),
    ("gemma4_31b", "Gemma-4-31B"),
    ("gemma4_31b_it", "Gemma-4-31B-It"),
    ("qwen36_27b", "Qwen3.6-27B"),
]
LANGS = ["es", "zh", "ja", "ko"]


def rows():
    out = {}
    for key, label in CHECKPOINTS:
        path = os.path.join(ACCESS, f"solvability_{key}.json")
        if not os.path.exists(path):
            continue
        gap = json.load(open(path)).get("gap_by_pivot", {})
        for lang in LANGS:
            g = gap.get(lang)
            if not g or "frac_en_solvable" not in g:
                continue
            out[(label, lang)] = (g["n_fail"], 100 * g["frac_en_solvable"])
    return out


def console(data):
    print(f"{'checkpoint':20s}" + "".join(f"{l:>16s}" for l in LANGS))
    for _, label in CHECKPOINTS:
        cells = []
        for lang in LANGS:
            v = data.get((label, lang))
            cells.append("--" if v is None else f"{v[1]:.1f}% (n={v[0]})")
        print(f"{label:20s}" + "".join(f"{c:>16s}" for c in cells))


if __name__ == "__main__":
    console(rows())
