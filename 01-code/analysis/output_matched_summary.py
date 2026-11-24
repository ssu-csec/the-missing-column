#!/usr/bin/env python3
"""Five-checkpoint language-neuron comparison against output-matched (L) random groups.

Reads 02-data/results/output_matched_control/<checkpoint>/{plan.json,validation.json,generations.jsonl}, produced
by 01-code/scripts/output_matched_control.py. For each checkpoint and target,
delta_R = I - mean(R seeds) (construction-matched random groups) and delta_L = I - mean(L seeds) (random groups whose
increment was fitted so that last-prompt-position next-token KL from the suppression-only arm S matches the
selected group's on calibration clusters). Designated form is zero in every arm, so delta equals the content effect.

Reports per-checkpoint and equal-checkpoint macros with within-checkpoint English-answer-cluster resampling
(10,000 draws), L minus R, held-out matching (sqrt KL ratio in [0.9, 1.1]) per seed, a sensitivity restricted to
seeds that pass held-out matching, and arm degeneration rates. Descriptive, unadjusted intervals.
Writes output_matched_control/summary.json (App. C.2).

Usage:
    python3 output_matched_summary.py
"""
import json
import os
import random

from result_paths import result_path

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.environ.get("LOGIT_MATCHED_DIR", result_path("output_matched_control"))
ROSTER = ["gemma4_31b", "gemma4_31b_it", "llama31_70", "llama31_70_it", "qwen36_27b"]
SEEDS = (7, 17, 27, 37, 47)
DRAWS = 10000


def r4(x):
    return round(x, 4)


def ci(values):
    values = sorted(values)
    return [r4(values[int(.025 * len(values))]), r4(values[int(.975 * len(values))])]


def load(model):
    d = os.path.join(DIR, model)
    plan = json.load(open(os.path.join(d, "plan.json")))
    val = json.load(open(os.path.join(d, "validation.json")))
    by, deg, cluster = {}, {}, {}
    for line in open(os.path.join(d, "generations.jsonl")):
        r = json.loads(line)
        s = r["score"]
        by[(r["chain"], r["target"], r["arm"])] = int(s["content"]) - int(s["form"])
        deg[(r["chain"], r["target"], r["arm"])] = int(s["degenerate"])
        cluster[r["chain"]] = r["answer_cluster"]
    out = {}
    for t in ("zh", "ko"):
        ok = [s for s in SEEDS if val[t]["seeds"][str(s)]["achieved"]]
        items = []
        for c in plan["evaluation"]:
            i = by[(c, t, "I")]
            rm = sum(by[(c, t, f"R{s}")] for s in SEEDS) / len(SEEDS)
            lm = sum(by[(c, t, f"L{s}")] for s in SEEDS) / len(SEEDS)
            lo = sum(by[(c, t, f"L{s}")] for s in ok) / len(ok) if ok else None
            items.append((cluster[c], i - rm, i - lm, None if lo is None else i - lo))
        n = len(plan["evaluation"])
        rates = {a: r4(100 * sum(deg[(c, t, f"{a}{s}")] for c in plan["evaluation"] for s in SEEDS) / (n * len(SEEDS)))
                 for a in ("R", "L")}
        rates["I"] = r4(100 * sum(deg[(c, t, "I")] for c in plan["evaluation"]) / n)
        out[t] = dict(items=items, matched_seeds=ok, degenerate=rates,
                      sqrt_ratio_L={str(s): r4(val[t]["seeds"][str(s)]["sqrt_ratio_L"]) for s in SEEDS},
                      sqrt_ratio_R={str(s): r4(val[t]["seeds"][str(s)]["sqrt_ratio_R"]) for s in SEEDS})
    return out


def macro(grouped, idx, rng):
    draws = []
    for _ in range(DRAWS):
        vals = []
        for keys, g in grouped:
            sample = [x for _ in keys for x in g[keys[rng.randrange(len(keys))]]]
            vals.append(idx(sample))
        draws.append(sum(vals) / len(vals))
    return draws


def main():
    data = {m: load(m) for m in ROSTER}
    result = {}
    for t in ("zh", "ko"):
        per, grouped = {}, []
        for m in ROSTER:
            d = data[m][t]
            items = d["items"]
            n = len(items)
            per[m] = dict(n=n, delta_R=r4(100 * sum(x[1] for x in items) / n),
                          delta_L=r4(100 * sum(x[2] for x in items) / n),
                          matched_seeds=d["matched_seeds"], degenerate=d["degenerate"],
                          sqrt_ratio_L=d["sqrt_ratio_L"], sqrt_ratio_R=d["sqrt_ratio_R"])
            if d["matched_seeds"]:
                per[m]["delta_L_matched_only"] = r4(100 * sum(x[3] for x in items) / n)
            g = {}
            for x in items:
                g.setdefault(x[0], []).append(x[1:])
            grouped.append((list(g), g))
        rng = random.Random(f"logit{t}20261001")
        dR = macro(grouped, lambda s: 100 * sum(x[0] for x in s) / len(s), rng)
        rng = random.Random(f"logit{t}20261001")
        dL = macro(grouped, lambda s: 100 * sum(x[1] for x in s) / len(s), rng)
        k = len(ROSTER)
        mR = sum(v["delta_R"] for v in per.values()) / k
        mL = sum(v["delta_L"] for v in per.values()) / k
        with_matched = [m for m in ROSTER if data[m][t]["matched_seeds"]]
        result[t] = dict(
            per_checkpoint=per,
            macro=dict(delta_R=r4(mR), delta_R_ci95=ci(dR), delta_L=r4(mL), delta_L_ci95=ci(dL),
                       L_minus_R=r4(mL - mR), L_minus_R_ci95=ci([b - a for a, b in zip(dR, dL)])),
            matched_only=dict(checkpoints=with_matched,
                              delta_L=r4(sum(per[m]["delta_L_matched_only"] for m in with_matched) / len(with_matched))))
    seeds_ok = sum(len(data[m][t]["matched_seeds"]) for m in ROSTER for t in ("zh", "ko"))
    out = dict(draws=DRAWS, held_out_matched_seeds=seeds_ok, seeds_total=len(ROSTER) * 2 * len(SEEDS),
               results=result,
               scope="Equal-checkpoint descriptive intervals (paired answer-cluster resampling shared by R and L); "
                     "not multiplicity-controlled")
    json.dump(out, open(os.path.join(DIR, "summary.json"), "w"), indent=2)
    print(json.dumps({t: dict(macro=result[t]["macro"], matched_only=result[t]["matched_only"]) for t in result},
                     indent=1), "\nheld-out matched seeds", seeds_ok, "/", out["seeds_total"])


if __name__ == "__main__":
    main()
