#!/usr/bin/env python3
"""Seed-as-random-factor and checkpoint random-effects summaries for the language-neuron energy-matched control.

Uses the five checkpoints listed in energy_matched_summary.RUNS (five construction-matched R and
five energy-matched M random groups per checkpoint and target), read from
02-data/results/energy_matched_control/<checkpoint>/.

1. Seed-inclusive intervals: a two-stage bootstrap resamples English-answer clusters within checkpoint and,
   independently, the five random-group seeds within checkpoint (with replacement), so intervals reflect
   control-draw variability rather than conditioning on the five fixed groups.
2. Checkpoint random effects: DerSimonian-Laird pooling of checkpoint-level delta with seed-inclusive
   bootstrap standard errors, reporting the pooled mean, between-checkpoint SD (tau), I^2, and a 95%
   prediction interval for a new checkpoint (t distribution, k-2 df).
3. Seed spread: per-checkpoint range of single-seed delta.
Descriptive; no multiplicity claim. Writes
energy_matched_control/summary/seed_and_checkpoint_random_effects.json.

Usage:
    python3 energy_matched_seed_effects.py
"""
import json
import math
import os
import random

import energy_matched_summary as S
from result_paths import DATA

DRAWS = 4000
T3 = 3.182446305284263  # t_{0.975, df=3} for k=5 checkpoints


def rows(model):
    chains = json.load(open(os.path.join(DATA, 'inputs', 'chain_lang_prompts.json')))
    d = os.path.join(S.A, S.RUNS[model])
    by = {}
    for l in open(os.path.join(d, 'generations.jsonl')):
        r = json.loads(l)
        by[(r['chain'], r['target'], r['arm'])] = int(r['score']['content']) - int(r['score']['form'])
    plan = json.load(open(os.path.join(d, 'plan.json')))
    out = {}
    for t in ('zh', 'ko'):
        out[t] = [(chains[c]['answer_label_by_lang']['en'], by[(c, t, 'I')],
                   {k: [by[(c, t, f'{k}{s}')] for s in S.SEEDS] for k in ('R', 'M')})
                  for c in plan['evaluation']]
    return out


def delta(items, kind, seeds):
    return 100 * sum(i - sum(ctrl[kind][s] for s in seeds) / len(seeds) for _, i, ctrl in items) / len(items)


def boot(items, kind, rng):
    g = {}
    for it in items:
        g.setdefault(it[0], []).append(it)
    keys = list(g)
    sample = [x for _ in keys for x in g[keys[rng.randrange(len(keys))]]]
    seeds = [rng.randrange(len(S.SEEDS)) for _ in S.SEEDS]
    return delta(sample, kind, seeds)


def pct(v):
    v = sorted(v)
    return [round(v[int(.025 * len(v))], 4), round(v[int(.975 * len(v))], 4)]


def dersimonian_laird(est, se):
    w = [1 / s ** 2 for s in se]
    fixed = sum(wi * e for wi, e in zip(w, est)) / sum(w)
    q = sum(wi * (e - fixed) ** 2 for wi, e in zip(w, est))
    k = len(est)
    c = sum(w) - sum(wi ** 2 for wi in w) / sum(w)
    tau2 = max(0., (q - (k - 1)) / c)
    ws = [1 / (s ** 2 + tau2) for s in se]
    mu = sum(wi * e for wi, e in zip(ws, est)) / sum(ws)
    se_mu = math.sqrt(1 / sum(ws))
    i2 = max(0., (q - (k - 1)) / q) if q > 0 else 0.
    half = T3 * math.sqrt(tau2 + se_mu ** 2)
    return dict(mu=round(mu, 4), ci95=[round(mu - 1.96 * se_mu, 4), round(mu + 1.96 * se_mu, 4)],
                tau=round(math.sqrt(tau2), 4), I2=round(100 * i2, 1), Q=round(q, 4),
                prediction95=[round(mu - half, 4), round(mu + half, 4)])


def main():
    data = {m: rows(m) for m in S.RUNS}
    result = {}
    for t in ('zh', 'ko'):
        result[t] = {}
        for kind in ('R', 'M'):
            per = {};est = [];se = []
            rng = random.Random(f'{t}{kind}20260930')
            macro_draws = []
            ck_draws = {m: [] for m in S.RUNS}
            for _ in range(DRAWS):
                vals = [boot(data[m][t], kind, rng) for m in S.RUNS]
                for m, v in zip(S.RUNS, vals):
                    ck_draws[m].append(v)
                macro_draws.append(sum(vals) / len(vals))
            for m in S.RUNS:
                items = data[m][t]
                single = [delta(items, kind, [s]) for s in range(len(S.SEEDS))]
                d = delta(items, kind, range(len(S.SEEDS)))
                sd = math.sqrt(sum((x - sum(ck_draws[m]) / DRAWS) ** 2 for x in ck_draws[m]) / (DRAWS - 1))
                per[m] = dict(delta=round(d, 4), seed_inclusive_ci95=pct(ck_draws[m]), boot_se=round(sd, 4),
                              single_seed_range=[round(min(single), 4), round(max(single), 4)])
                est.append(d);se.append(sd)
            result[t][kind] = dict(per_checkpoint=per,
                                   equal_checkpoint_macro=dict(delta=round(sum(est) / len(est), 4),
                                                               seed_inclusive_ci95=pct(macro_draws)),
                                   random_effects=dersimonian_laird(est, se))
    out = os.path.join(S.A, 'summary', 'seed_and_checkpoint_random_effects.json')
    json.dump(dict(runs=S.RUNS, draws=DRAWS, results=result,
                   scope='Seed-and-cluster bootstrap; DerSimonian-Laird across five checkpoints; descriptive'),
              open(out, 'w'), indent=2)
    for t in result:
        for kind in ('R', 'M'):
            r = result[t][kind]
            print(t, kind, 'macro', r['equal_checkpoint_macro'], '| RE', r['random_effects'])
            for m, p in r['per_checkpoint'].items():
                print('    ', m, p)


if __name__ == '__main__':
    main()
