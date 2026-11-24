#!/usr/bin/env python3
"""Item-level control-to-intervention state transitions for every audited cell.

Each scored output is in one of three states: F (content in the designated form), c (content in another
form), or 0 (no content). For every item we record the control state and the intervention state, pooled
over checkpoints, using the locked attribution-strict rows of paired_effects.py.

Pooled over items, Delta_F - Delta_C is proportional to (c->F - F->c) + (c->0 - 0->c), so the form effect
exceeds the content effect exactly when the intervention removes more correct answers in another form than
it adds, net of within-content form changes. Writes 02-data/results/state_transitions/summary.json and,
if a path is given, the manuscript table.

Usage:
    python3 state_transitions.py [TABLE_TEX_PATH]
"""
import importlib.util
import json
import os
from collections import Counter
from result_paths import result_path

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("sep", os.path.join(HERE, "paired_effects.py"))
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

KINDS = ("0->F", "0->c", "F->0", "c->0", "c->F", "F->c")
FAMILIES = ("vT", "neuron", "anchor")


def state(content, form):
    return "F" if form else ("c" if content else "0")


def cell(family, lang):
    rows_by_model = S.cell_rows(family, lang)
    counts, n, pooled = Counter(), 0, 0
    for rows in rows_by_model:
        for _, ci, cr, fi, fr in rows:
            counts[f"{state(cr, fr)}->{state(ci, fi)}"] += 1
            pooled += (int(fi) - int(fr)) - (int(ci) - int(cr))
            n += 1
    # Pooled identity: sum(dF - dC) = (c->F - F->c) + (c->0 - 0->c).
    assert pooled == (counts["c->F"] - counts["F->c"]) + (counts["c->0"] - counts["0->c"]), (family, lang)
    mc, mf, d = S.point(rows_by_model)
    return {"n": n, **{k: counts.get(k, 0) for k in KINDS},
            "delta_C": round(mc, 1), "delta_F": round(mf, 1), "delta": round(d, 1)}


def main():
    cells = {(f, l): cell(f, l) for f in FAMILIES for l in S.LANGS}
    out_dir = result_path("state_transitions")
    os.makedirs(out_dir, exist_ok=True)
    json.dump({f"{f}:{l}": c for (f, l), c in cells.items()},
              open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
    for (f, l), c in cells.items():
        print(f, l, {k: c[k] for k in KINDS}, c["delta_C"], c["delta_F"], c["delta"])


if __name__ == "__main__":
    main()
