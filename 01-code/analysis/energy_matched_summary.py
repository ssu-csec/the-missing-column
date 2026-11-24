#!/usr/bin/env python3
"""Five-checkpoint language-neuron comparison against construction-matched (R) and energy-matched (M) controls.

Reads 02-data/results/energy_matched_control/<checkpoint>/{plan.json,generations.jsonl}, produced by
01-code/scripts/energy_matched_control.py (energy_matched_control_gemma.py for Gemma-4-31B). Energy matching is
fitted on all prompt positions for base checkpoints (Gemma-4-31B with S=U) and on the user-message span for chat
checkpoints (App. C.2).

For each checkpoint and target (zh, ko), delta_R = I - mean(R seeds) and delta_M = I - mean(M seeds) over the five
random-group seeds, on paired items (designated form is zero in every arm, so delta equals the content effect).
Equal-checkpoint macros use within-checkpoint English-answer-cluster resampling (10,000 draws),
matching the prompt-comparison pooling. Descriptive, unadjusted intervals.
Writes energy_matched_control/summary/macro.json.

Usage:
    python3 energy_matched_summary.py
"""
import json
import os
import random

from result_paths import DATA, result_path

HERE = os.path.dirname(os.path.abspath(__file__))
A = os.environ.get('NEURON_STRENGTH_DATA', result_path('energy_matched_control'))
RUNS = {m: m for m in ('gemma4_31b', 'gemma4_31b_it', 'llama31_70', 'llama31_70_it', 'qwen36_27b')}
SEEDS = (7, 17, 27, 37, 47)
DRAWS = 10000


def load(model):
    chains = json.load(open(os.environ.get('CHAIN_PROMPTS', os.path.join(DATA, 'inputs', 'chain_lang_prompts.json'))))
    d = os.path.join(A, RUNS[model])
    rows = [json.loads(l) for l in open(os.path.join(d, 'generations.jsonl'))]
    plan = json.load(open(os.path.join(d, 'plan.json')))
    by = {}
    for r in rows:
        s = r['score']
        by[(r['chain'], r['target'], r['arm'])] = int(s['content']) - int(s['form'])
    out = {}
    for t in ('zh', 'ko'):
        items = []
        for c in plan['evaluation']:
            i = by[(c, t, 'I')]
            rm = sum(by[(c, t, f'R{s}')] for s in SEEDS) / 5
            mm = sum(by[(c, t, f'M{s}')] for s in SEEDS) / 5
            items.append((chains[c]['answer_label_by_lang']['en'], i - rm, i - mm))
        out[t] = items
    return out


def ci(values):
    values = sorted(values)
    return [round(values[int(.025 * len(values))], 2), round(values[int(.975 * len(values))], 2)]


def main():
    data = {m: load(m) for m in RUNS}
    result = {}
    for t in ('zh', 'ko'):
        per = {}
        grouped = []
        for m in RUNS:
            items = data[m][t]
            n = len(items)
            per[m] = dict(n=n, delta_R=round(100 * sum(x[1] for x in items) / n, 2),
                          delta_M=round(100 * sum(x[2] for x in items) / n, 2))
            g = {}
            for cl, r, mm in items:
                g.setdefault(cl, []).append((r, mm))
            grouped.append((list(g), g))
        rng = random.Random(20260930)
        draws = {'R': [], 'M': [], 'M_minus_R': []}
        for _ in range(DRAWS):
            rs, ms = [], []
            for keys, g in grouped:
                sample = [x for _ in keys for x in g[keys[rng.randrange(len(keys))]]]
                rs.append(100 * sum(x[0] for x in sample) / len(sample))
                ms.append(100 * sum(x[1] for x in sample) / len(sample))
            r, mm = sum(rs) / len(rs), sum(ms) / len(ms)
            draws['R'].append(r); draws['M'].append(mm); draws['M_minus_R'].append(mm - r)
        k = len(RUNS)
        mR = sum(v['delta_R'] for v in per.values()) / k
        mM = sum(v['delta_M'] for v in per.values()) / k
        result[t] = dict(per_checkpoint=per,
                         macro=dict(delta_R=round(mR, 2), delta_R_ci95=ci(draws['R']),
                                    delta_M=round(mM, 2), delta_M_ci95=ci(draws['M']),
                                    M_minus_R=round(mM - mR, 2), M_minus_R_ci95=ci(draws['M_minus_R'])))
    out = os.path.join(A, 'summary', 'macro.json')
    json.dump(dict(runs=RUNS, draws=DRAWS, results=result,
                   scope='Equal-checkpoint descriptive intervals; not multiplicity-controlled'),
              open(out, 'w'), indent=2)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
