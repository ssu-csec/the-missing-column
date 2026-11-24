#!/usr/bin/env python3
"""Checkpoint-exclusion sensitivity: drop Gemma-4-31B, where neuron degeneration is concentrated.

Diagnostic only; it does not redefine the primary estimand. Computed from preserved generations; no new
generation is involved. Intervals are descriptive, unadjusted answer-cluster bootstrap intervals with
within-checkpoint resampling and equal checkpoint weights (no multiplicity correction).

1. Main grid (the paired-effects table cells): Delta_C and Delta_DFR over the four remaining checkpoints.
2. Language-neuron comparators: delta against construction-matched (R), energy-matched (M), and
   output-matched (L) random groups over the same four checkpoints (designated form is zero in every arm,
   so delta equals the content effect).

Inputs: preserved generations in 02-data/results/interventions/ (via paired_effects.py), and the comparator
items in energy_matched_control/ and output_matched_control/. Writes
02-data/results/checkpoint_exclusion/checkpoint_exclusion.json.

Usage:
    python3 checkpoint_exclusion_sensitivity.py [--out PATH]
"""
import argparse
import json
import os
import random

import output_matched_summary as LM
import paired_effects as S
import energy_matched_summary as SR
from result_paths import result_path

HERE = os.path.dirname(os.path.abspath(__file__))
EXCLUDE = "gemma4_31b"
CELLS = [("neuron", "zh"), ("neuron", "ko"), ("vT", "ko"), ("vT", "es"), ("vT", "zh")]
DRAWS = 10000


def ci(draws):
    draws = sorted(draws)
    return [round(draws[int(.025 * len(draws))], 1), round(draws[int(.975 * len(draws))], 1)]


def comparator_macro(items_by_model, n_cols, seed):
    """items: (cluster, d_1, ..., d_k) per item; returns per-column macro point and interval."""
    grouped = []
    for items in items_by_model:
        g = {}
        for it in items:
            g.setdefault(it[0], []).append(it[1:])
        grouped.append((list(g), g))
    points = [sum(100 * sum(x[j] for x in it) / len(it) for it in
                  [[i[1:] for i in items] for items in items_by_model]) / len(items_by_model) for j in range(n_cols)]
    rng = random.Random(seed)
    draws = [[] for _ in range(n_cols)]
    for _ in range(DRAWS):
        vals = [[] for _ in range(n_cols)]
        for keys, g in grouped:
            sample = [x for _ in keys for x in g[keys[rng.randrange(len(keys))]]]
            for j in range(n_cols):
                vals[j].append(100 * sum(x[j] for x in sample) / len(sample))
        for j in range(n_cols):
            draws[j].append(sum(vals[j]) / len(vals[j]))
    return [(round(p, 1), ci(d)) for p, d in zip(points, draws)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=result_path("checkpoint_exclusion", "checkpoint_exclusion.json"))
    a = ap.parse_args()
    keep = [m for m in S.ROSTER if m != EXCLUDE]
    out = {"excluded": EXCLUDE, "checkpoints": keep,
           "scope": "diagnostic checkpoint-exclusion sensitivity; not a redefined primary estimand; "
                    "descriptive unadjusted intervals", "main_grid": [], "neuron_comparators": {}}
    for k, (family, lang) in enumerate(CELLS):
        rows = [r for m, r in zip(S.ROSTER, S.cell_rows(family, lang)) if m != EXCLUDE]
        mc, mf, _ = S.point(rows)
        cc, cf, _ = S.paired_ci(rows, seed=777 + k)
        out["main_grid"].append({"family": family, "lang": lang, "delta_C": round(mc, 1), "delta_C_ci": cc,
                                 "delta_DFR": round(mf, 1), "delta_DFR_ci": cf})
    strength = {m: SR.load(m) for m in keep}
    logit = {m: LM.load(m) for m in keep}
    for t in ("zh", "ko"):
        sr = [[(c, r, mm) for c, r, mm in strength[m][t]] for m in keep]
        rm = comparator_macro(sr, 2, 20261007)
        lm = [[(it[0], it[2]) for it in logit[m][t]["items"]] for m in keep]
        (l_point, l_ci), = comparator_macro(lm, 1, 20261008)
        out["neuron_comparators"][t] = {"delta_R": rm[0], "delta_M": rm[1], "delta_L": (l_point, l_ci)}
    for c in out["main_grid"]:
        print(c)
    print(json.dumps(out["neuron_comparators"]))
    json.dump(out, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
