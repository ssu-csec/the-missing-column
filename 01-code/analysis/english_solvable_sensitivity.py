#!/usr/bin/env python3
"""English-solvable restriction and item-weighted pooling for the Spanish-prompt audit.

Two sensitivities, computed from the preserved generations
with the locked scorer of paired_effects.py; no new generation is involved.

1. English-solvable restriction: keep only evaluation items that the same
   checkpoint answers correctly under an English prompt
   (02-data/results/solvability/solvability_<model>.json, per_chain[chain].x_correct.en). This
   separates cross-lingual access failures from general composition failures.
2. Item-weighted pooling: weight item--checkpoint observations equally instead
   of checkpoints, as an alternative to the checkpoint-equal macro.

Both are descriptive, unadjusted paired answer-cluster bootstrap intervals
(within-checkpoint resampling, 4,000 draws, seed 900 + cell index as in paired_effects.analyze)
and do not inherit the multiplicity-controlled verdict. Writes
02-data/results/english_solvable/english_solvable_and_item_weighted.json (App. A.7).

Usage:
    python3 english_solvable_sensitivity.py [--out PATH]
"""
import argparse
import json
import os
import random

import paired_effects as S
from result_paths import result_path

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = [(f, l) for f in S.SPANISH_FAMILIES for l in S.LANGS]


def en_solvable(model):
    d = json.load(open(result_path("solvability", f"solvability_{model}.json")))
    return {c for c, v in d["per_chain"].items() if v.get("x_correct", {}).get("en")}


def rows_for(family, model, lang, keep=None):
    """spanish_rows with an optional chain filter (same scoring and exclusions)."""
    V = S.V
    d = V.load("interventions/" + V.FAMILIES[family].format(model=model))
    chains, floors = d["eval_chains"], d["floors"]
    steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
    rows = []
    for i, chain in enumerate(chains):
        if (chain not in V.CHAINS or chain not in floors or i >= len(steer)
                or i >= len(rand) or not V.scoreable(chain, lang)):
            continue
        if keep is not None and chain not in keep:
            continue
        si, ri = steer[i].get("txt", ""), rand[i].get("txt", "")
        rows.append((S._cluster(chain), V.content(si, chain), V.content(ri, chain),
                     V.ilr(si, chain, lang), V.ilr(ri, chain, lang)))
    return rows


def pooled_ci(rows_by_model, seed, draws=S.B):
    """Item-weighted (Delta_C, Delta_F, delta) with within-checkpoint cluster resampling."""
    def est(samples):
        n = sum(len(s) for s in samples)
        dc = 100 * sum(int(r[0]) - int(r[1]) for s in samples for r in s) / n
        df = 100 * sum(int(r[2]) - int(r[3]) for s in samples for r in s) / n
        return dc, df, dc - df
    grouped = []
    for rows in rows_by_model:
        g = {}
        for r in rows:
            g.setdefault(r[0], []).append(r[1:])
        grouped.append((list(g), g))
    point = est([[r[1:] for r in rows] for rows in rows_by_model])
    rng = random.Random(seed)
    out = [[], [], []]
    for _ in range(draws):
        samples = [[it for _ in keys for it in g[keys[rng.randrange(len(keys))]]]
                   for keys, g in grouped]
        for k, v in enumerate(est(samples)):
            out[k].append(v)
    cis = []
    for d in out:
        d.sort()
        cis.append((round(d[int(.025 * draws)], 1), round(d[int(.975 * draws)], 1)))
    return [round(x, 1) for x in point], cis


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=result_path("english_solvable", "english_solvable_and_item_weighted.json"))
    a = p.parse_args()
    keep = {m: en_solvable(m) for m in S.ROSTER}
    result = []
    for k, (family, lang) in enumerate(CELLS):
        full = [rows_for(family, m, lang) for m in S.ROSTER]
        solv = [rows_for(family, m, lang, keep[m]) for m in S.ROSTER]
        cell = {"family": family, "lang": lang,
                "n_full": [len(r) for r in full], "n_en_solvable": [len(r) for r in solv]}
        if all(solv):
            mc, mf, d = S.point(solv)
            ci_c, ci_f, ci_d = S.paired_ci(solv, 900 + k)
            cell["en_solvable_macro"] = dict(content=round(mc, 1), content_ci95=ci_c,
                                             form=round(mf, 1), form_ci95=ci_f,
                                             delta=round(d, 1), delta_ci95=ci_d)
        pt, cis = pooled_ci(full, 900 + k)
        cell["item_weighted_full"] = dict(content=pt[0], content_ci95=cis[0], form=pt[1],
                                          form_ci95=cis[1], delta=pt[2], delta_ci95=cis[2])
        mc, mf, d = S.point(full)
        cell["macro_full_check"] = dict(content=round(mc, 1), form=round(mf, 1), delta=round(d, 1))
        result.append(cell)
        print(json.dumps(cell))
    with open(a.out, "w") as fh:
        json.dump(dict(cells=result, draws=S.B,
                       scope="Descriptive unadjusted paired answer-cluster intervals; no multiplicity claim"),
                  fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
