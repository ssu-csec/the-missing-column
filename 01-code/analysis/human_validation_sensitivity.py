#!/usr/bin/env python3
"""Effect of the stage-1 human judgments on the Spanish-prompt cells (Appendix B.2).

Reads the returned stage-1 judgments and their item key from
02-data/results/human_form_audit/grid_validation/ and recomputes Delta_C, Delta_F, and
delta for every cell that contains a judged item, under two sensitivities:

  forms    credit the designated form of every output judged content-correct with an
           acceptable designated form;
  all      additionally remove the content (and form) credit of every output judged
           content-incorrect.

Point estimates and paired answer-cluster intervals use paired_effects (seed 900 + cell
index, as in Table 1).  Writes grid_validation/sensitivity.json.

    python3 human_validation_sensitivity.py
"""
import csv
import json
import os

import paired_effects as S
from result_paths import result_path

BASE = result_path("human_form_audit", "grid_validation")
FAMILIES = ("vT", "neuron", "anchor")
LANGS = ("es", "zh", "ja", "ko")


def judgments():
    key = {}
    with open(os.path.join(BASE, "item_key.jsonl")) as fh:
        for line in fh:
            k = json.loads(line)
            key[k["item_id"]] = k
    out = {}
    with open(os.path.join(BASE, "annotations_all_result.csv"), encoding="utf-8-sig", newline="") as fh:
        for r in csv.DictReader(fh):
            k = key[r["item_id"]]
            out[(k["family"], k["model"], k["lang"], k["arm"], k["chain"])] = r
    return out


def rows(family, model, lang, judged, mode):
    V = S.V
    d = V.load("interventions/" + V.FAMILIES[family].format(model=model))
    chains, floors = d["eval_chains"], d["floors"]
    steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
    out = []
    for i, chain in enumerate(chains):
        if (chain not in V.CHAINS or chain not in floors or i >= len(steer) or i >= len(rand)
                or not V.scoreable(chain, lang)):
            continue
        arms = []
        for arm, item in (("I", steer[i]), ("R", rand[i])):
            text = item.get("txt", "")
            c, f = V.content(text, chain), V.ilr(text, chain, lang)
            h = judged.get((family, model, lang, arm, chain))
            if h and mode in ("forms", "all") and h["content_status"] == "correct" \
                    and h["target_form_status"] == "acceptable":
                f = True
            if h and mode == "all" and h["content_status"] == "incorrect":
                c, f = False, False
            arms.append((c, f))
        (ci, fi), (cr, fr) = arms
        out.append((V.CHAINS[chain]["answer_label_by_lang"].get("en") or chain, ci, cr, fi, fr))
    return out


def main():
    judged = judgments()
    cells = sorted({(k[0], k[2]) for k in judged})
    result = {"scope": "stage-1 human judgments applied to the locked scores; descriptive paired intervals",
              "cells": []}
    for family, lang in cells:
        seed = 900 + FAMILIES.index(family) * 4 + LANGS.index(lang)
        entry = {"family": family, "lang": lang,
                 "judged_items": sum(1 for k in judged if (k[0], k[2]) == (family, lang))}
        for mode in ("locked", "forms", "all"):
            by_model = [rows(family, m, lang, judged, mode) for m in S.ROSTER]
            mc, mf, d = S.point(by_model)
            cc, cf, cd = S.paired_ci(by_model, seed)
            entry[mode] = {"delta_C": round(mc, 1), "delta_C_ci95": list(cc), "delta_F": round(mf, 1),
                           "delta_F_ci95": list(cf), "delta": round(d, 1), "delta_ci95": list(cd)}
        result["cells"].append(entry)
        print(f"{family:6s} {lang}: " + "  ".join(
            f"{m} dF {entry[m]['delta_F']:+.1f} delta {entry[m]['delta']:+.1f} {entry[m]['delta_ci95']}"
            for m in ("locked", "forms", "all")))
    with open(os.path.join(BASE, "sensitivity.json"), "w") as fh:
        json.dump(result, fh, indent=2)
        fh.write("\n")


if __name__ == "__main__":
    main()
