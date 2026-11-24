#!/usr/bin/env python3
"""Selection-aware status of every Spanish-prompt cell under leave-one-checkpoint-out rosters.

For each of the twelve family--language cells and each of the five checkpoints, drops
that checkpoint and recomputes the separation contrast delta over the remaining four
with the two selection-aware criteria of paired_effects.multiplicity: the Bonferroni
percentile interval over the twelve cells (B = 40,000) and the cluster sign-flip test
(20,000 draws, passed when p < .05/12).  Seeds follow paired_effects.multiplicity
(3000 + cell index, and +77 for the sign-flip test), so the full-roster values
reproduce Table 1.  Diagnostic only: it does not redefine the primary estimand.
The excluded-Gemma-4-31B entries for the Chinese and Korean language-neuron cells
are also computed, with that script's own seeds, by neuron_robustness.py; the
two differ only by Monte Carlo error, and Section 5.3 quotes the latter.

Writes 02-data/results/checkpoint_exclusion/leave_one_out_selection.json.

    python3 leave_one_out_selection.py [--out PATH]
"""
import argparse
import json

import paired_effects as S
from result_paths import result_path

FAMILIES = ("vT", "neuron", "anchor")
LANGS = ("es", "zh", "ja", "ko")


def criteria(rows, seed):
    alpha = 0.05 / S.MULT_FAMILY_SIZE
    draws = S.delta_draws(rows, seed, S.B_MULT)
    n = len(draws)
    lo, hi = draws[int(alpha / 2 * n)], draws[int((1 - alpha / 2) * n)]
    p = S.signflip_p(rows, seed + 77)
    bonferroni = bool(lo > 0 or hi < 0)
    signflip = bool(p < alpha)
    return {"delta": round(S.point(rows)[2], 1), "bonferroni_ci": [round(lo, 1), round(hi, 1)],
            "p_signflip": round(p, 4), "bonferroni_pass": bonferroni, "signflip_pass": signflip,
            "joint_pass": bonferroni and signflip}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=result_path("checkpoint_exclusion", "leave_one_out_selection.json"))
    args = ap.parse_args()
    cells = []
    for f, family in enumerate(FAMILIES):
        for l, lang in enumerate(LANGS):
            seed = 3000 + f * 4 + l
            rows = S.cell_rows(family, lang)
            drops = {}
            for k, model in enumerate(S.ROSTER):
                kept = [r for j, r in enumerate(rows) if j != k]
                drops[model] = criteria(kept, seed)
            cells.append({"family": family, "lang": lang,
                          "joint_pass_count": sum(d["joint_pass"] for d in drops.values()),
                          "drop": drops})
            print(f"{family:6s} {lang}: joint pass in {cells[-1]['joint_pass_count']}/5 leave-one-out rosters",
                  flush=True)
    out = {"roster": list(S.ROSTER), "bonferroni_draws": S.B_MULT, "signflip_draws": S.N_PERM,
           "threshold": round(0.05 / S.MULT_FAMILY_SIZE, 4),
           "scope": "diagnostic leave-one-checkpoint-out; not a redefined primary estimand",
           "cells": cells}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
