#!/usr/bin/env python3
"""Realized-form composition of intervention-only content gains (descriptive diagnostic).

Population: paired items in each main-audit cell (pooled over checkpoints) where the
intervention output is a locked content match and the control output is not (C_I=1, C_R=0). For each such
intervention output, the scored first line is classified by which registered answer form it contains, in order:
  designated   - credited designated form (F_T = 1)
  spanish      - the Spanish canonical label (prompt language), not also the English label
  english      - the English canonical label, not also the Spanish label
  es_en_shared - a label identical in Spanish and English (attributable to neither)
  other        - a registered form of another evaluated language (Korean, Japanese, Chinese)
  multiple     - forms of more than one of the above non-designated categories
Outputs with none of these forms are counted as unclassified. The locked matcher and scoring rules are reused
unchanged; this does not alter any estimand. Reads the preserved generations in 02-data/results/interventions/ and
writes 02-data/results/realized_form_composition/realized_form_composition.json.

Usage:
    python3 realized_form_composition.py
"""
import collections
import json
import os

import paired_effects as S
from result_paths import result_path

V = S.V
OUT = result_path("realized_form_composition", "realized_form_composition.json")
CATS = ("designated", "spanish", "english", "es_en_shared", "other", "multiple")


def first(text):
    t = (text or "").strip()
    return t.splitlines()[0] if t else ""


def classify(text, chain, lang):
    if V.ilr(text, chain, lang):
        return "designated"
    line = first(text)
    labs = V.CHAINS[chain]["answer_label_by_lang"]
    es, en = labs.get("es"), labs.get("en")
    found = set()
    if es and en and V.norm(es) == V.norm(en):
        if V.matches(line, [es]):
            found.add("es_en_shared")
    else:
        if es and V.matches(line, [es]):
            found.add("spanish")
        if en and V.matches(line, [en]):
            found.add("english")
    for other in ("ko", "ja", "zh"):
        if other != lang and V.matches(line, V.inventory(chain, other)):
            found.add("other")
    if not found:
        return "unclassified"
    return found.pop() if len(found) == 1 else "multiple"


def main():
    out = {}
    for family in ("vT", "neuron", "anchor"):
        for lang in ("es", "zh", "ja", "ko"):
            counts = collections.Counter()
            for model in S.ROSTER:
                d = V.load("interventions/" + V.FAMILIES[family].format(model=model))
                chains, floors = d["eval_chains"], d["floors"]
                steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
                for i, chain in enumerate(chains):
                    if (chain not in V.CHAINS or chain not in floors or i >= len(steer) or i >= len(rand)
                            or not V.scoreable(chain, lang)):
                        continue
                    si, ri = steer[i].get("txt", ""), rand[i].get("txt", "")
                    if V.content(si, chain) and not V.content(ri, chain):
                        counts[classify(si, chain, lang)] += 1
            out[f"{family}:{lang}"] = dict(gained=sum(counts.values()), **{k: counts.get(k, 0) for k in
                                                                         CATS + ("unclassified",)})
    json.dump(dict(population="intervention-only content gains (C_I=1, C_R=0), pooled over checkpoints",
                   categories=list(CATS), cells=out), open(OUT, "w"), indent=2)
    for k, v in out.items():
        print(k, v)


if __name__ == "__main__":
    main()
