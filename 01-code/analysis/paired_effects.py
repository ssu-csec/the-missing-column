#!/usr/bin/env python3
"""Paired estimation of Delta_C, Delta_DFR and the separation contrast delta.

The primary estimand is

    delta = [P(C=1|I) - P(C=1|R)] - [P(F=1|I) - P(F=1|R)]
          = Delta_C - Delta_DFR,

where C is content match, F is designated-form realization (DFR), I is the
intervention arm and R its construction-matched control.  Because C and F are
scored on the *same* generation and F <= C, the two effects are strongly
dependent; separate confidence intervals do not test their difference.  This
script therefore resamples answer clusters **once** per bootstrap draw and
recomputes both effects on that same resample, so the covariance between them
is preserved and the reported interval is a paired interval for delta itself.

Effects are checkpoint-equal macros over the fixed roster; answer clusters are
resampled within checkpoint (B = 4,000 draws).  Cells cover the three
Spanish-prompt families (v_T, language neurons, MRRE) and the SCH general pool,
each for es/zh/ja/ko.  It also reports, for each cell, the item-level transition
decomposition: among generations that the intervention newly makes
content-correct (I content = 1, R content = 0), the share that realizes the
designated answer form; leave-one-checkpoint-out, permissive-policy and
cluster-disjoint sensitivities; the v_T random-seed range; and the
selection-aware reading (cluster sign-flip test and Bonferroni interval over
the twelve Spanish-prompt cells).  Scoring helpers are imported from
verify_numbers.py; generations are read from 02-data/results/interventions/.

Usage:

    python paired_effects.py                 # human-readable report
    python paired_effects.py --json          # machine-readable

Set ``AOL_MATCHER_POLICY=designated`` for the shared-form sensitivity.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _scoring():
    """Load verify_numbers.py for its scoring helpers, skipping its checks."""
    os.environ["AOL_SCORING_ONLY"] = "1"
    spec = importlib.util.spec_from_file_location(
        "aol_scoring", os.path.join(HERE, "verify_numbers.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        os.environ.pop("AOL_SCORING_ONLY", None)
    return mod


V = _scoring()

B = 4000
# The selection-aware pass needs a stable far tail, so it uses a larger B.
B_MULT = 40000
# Family of tests for the Bonferroni adjustment: the Spanish-prompt audit grid
# of three intervention families by four designated languages.
MULT_FAMILY_SIZE = 12
# Cluster-level sign-flip draws for the randomization test.
N_PERM = 20000
ROSTER = sorted(V.ROSTER)
SCH_MODELS = ["qwen36_27b", "llama31_70", "llama31_70_it", "gemma4_31b"]
SPANISH_FAMILIES = ("vT", "neuron", "anchor")
LANGS = ("es", "zh", "ja", "ko")
MODEL_LABEL = {"llama31_70": "L-70B", "llama31_70_it": "L-70B-It",
               "gemma4_31b": "G-31B", "gemma4_31b_it": "G-31B-It",
               "qwen36_27b": "Q-27B"}


# ----------------------------------------- deterministic permissive policy
# The permissive sensitivity uses only frozen formatting equivalences.  It
# never overlays human judgments, expands entity identity, or inspects arms.


# --------------------------------------------------------------- row builders
def _cluster(chain):
    return V.CHAINS[chain]["answer_label_by_lang"].get("en") or chain


def calibration_clusters(family, model):
    """English-answer clusters represented in this cell's calibration split."""
    d = V.load("interventions/" + V.FAMILIES[family].format(model=model))
    return {_cluster(c) for c in d.get("calib_chains", []) if c in V.CHAINS}


def spanish_rows(family, model, lang, seed=None, permissive=False,
                 cluster_disjoint=False):
    """Per-item (cluster, content_I, content_R, form_I, form_R) for one cell.

    ``cluster_disjoint`` additionally drops evaluation items whose English
    answer cluster also occurs in the calibration split, so that configuration
    selection and evaluation share no answer cluster.
    """
    d = V.load("interventions/" + V.FAMILIES[family].format(model=model))
    chains, floors = d["eval_chains"], d["floors"]
    steer = d["eval"][lang]["steer_items"]
    if seed is None:
        rand = d["eval"][lang]["rand_items"]
    else:
        rand = d["eval"][lang]["rand_items_by_seed"][str(seed)]
    blocked = calibration_clusters(family, model) if cluster_disjoint else set()
    rows = []
    for i, chain in enumerate(chains):
        if (chain not in V.CHAINS or chain not in floors or i >= len(steer)
                or i >= len(rand) or not V.scoreable(chain, lang)
                or _cluster(chain) in blocked):
            continue
        si, ri = steer[i].get("txt", ""), rand[i].get("txt", "")
        ci = V.content(si, chain, permissive=permissive)
        fi = V.ilr(si, chain, lang, permissive=permissive)
        cr = V.content(ri, chain, permissive=permissive)
        fr = V.ilr(ri, chain, lang, permissive=permissive)
        rows.append((V.CHAINS[chain]["answer_label_by_lang"].get("en") or chain,
                     ci, cr, fi, fr))
    return rows


def sch_rows(model, lang, pool="general", permissive=False):
    d = V.load(f"interventions/sch_amplification_{model}_{pool}.json")
    rows = []
    for i, item in enumerate(d["by_lang"][lang]["items"]):
        chain = item["chain"]
        if chain not in V.CHAINS or not V.scoreable(chain, lang):
            continue
        si = item["amp_a1.0"].get("txt", "")
        ri = item["rand_a1.0"].get("txt", "")
        ci = V.content(si, chain, permissive=permissive)
        fi = V.ilr(si, chain, lang, permissive=permissive)
        cr = V.content(ri, chain, permissive=permissive)
        fr = V.ilr(ri, chain, lang, permissive=permissive)
        rows.append((V.CHAINS[chain]["answer_label_by_lang"].get("en") or chain,
                     ci, cr, fi, fr))
    return rows


def cell_rows(family, lang, seed=None, permissive=False, cluster_disjoint=False):
    """List of per-checkpoint row lists for one family/language cell."""
    if family == "sch":
        return [sch_rows(m, lang, permissive=permissive) for m in SCH_MODELS]
    return [spanish_rows(family, m, lang, seed=seed, permissive=permissive,
                         cluster_disjoint=cluster_disjoint)
            for m in ROSTER]


# ------------------------------------------------------------------ estimation
def point(rows_by_model):
    """Checkpoint-equal macro (DeltaContent, DeltaForm, delta) in points."""
    dc, df = [], []
    for rows in rows_by_model:
        n = len(rows)
        dc.append(100 * sum(int(r[1]) - int(r[2]) for r in rows) / n)
        df.append(100 * sum(int(r[3]) - int(r[4]) for r in rows) / n)
    k = len(rows_by_model)
    mc, mf = sum(dc) / k, sum(df) / k
    return mc, mf, mc - mf


def paired_ci(rows_by_model, seed):
    """Paired answer-cluster bootstrap over DeltaContent, DeltaForm and delta.

    One cluster resample per draw feeds both outcomes, so the delta interval
    carries the within-generation covariance of the two effects.
    """
    grouped = []
    for rows in rows_by_model:
        groups = {}
        for row in rows:
            groups.setdefault(row[0], []).append(row[1:])
        grouped.append((list(groups), groups))
    rng = random.Random(seed)
    dc_draws, df_draws, d_draws = [], [], []
    for _ in range(B):
        dcs, dfs = [], []
        for keys, groups in grouped:
            sample = [item for _ in keys
                      for item in groups[keys[rng.randrange(len(keys))]]]
            n = len(sample)
            dcs.append(100 * sum(int(s[0]) - int(s[1]) for s in sample) / n)
            dfs.append(100 * sum(int(s[2]) - int(s[3]) for s in sample) / n)
        k = len(grouped)
        mc, mf = sum(dcs) / k, sum(dfs) / k
        dc_draws.append(mc)
        df_draws.append(mf)
        d_draws.append(mc - mf)
    out = []
    for draws in (dc_draws, df_draws, d_draws):
        draws.sort()
        out.append((round(draws[int(.025 * B)], 1), round(draws[int(.975 * B)], 1)))
    return out


def transitions(rows_by_model):
    """Pooled item-level decomposition of intervention-induced content gains.

    ``gained`` counts generations the intervention makes content-correct while
    the matched control is not; ``form`` is the subset also realizing the
    designated form; ``lost`` counts the reverse content transition.
    """
    gained = form = lost = 0
    for rows in rows_by_model:
        for _, ci, cr, fi, _ in rows:
            if ci and not cr:
                gained += 1
                form += int(bool(fi))
            elif cr and not ci:
                lost += 1
    return {"gained": gained, "gained_in_form": form, "lost": lost,
            "share_in_form": (round(100 * form / gained, 1) if gained else None)}


def analyze(family, lang, seed_offset):
    rows = cell_rows(family, lang)
    mc, mf, d = point(rows)
    ci_c, ci_f, ci_d = paired_ci(rows, 900 + seed_offset)
    per_ckpt = []
    models = SCH_MODELS if family == "sch" else ROSTER
    for model, r in zip(models, rows):
        n = len(r)
        c = 100 * sum(int(x[1]) - int(x[2]) for x in r) / n
        f = 100 * sum(int(x[3]) - int(x[4]) for x in r) / n
        per_ckpt.append({"model": model, "n": n, "content": round(c, 1),
                         "form": round(f, 1), "delta": round(c - f, 1)})
    return {
        "family": family, "lang": lang,
        "content": round(mc, 1), "content_ci95": list(ci_c),
        "form": round(mf, 1), "form_ci95": list(ci_f),
        "delta": round(d, 1), "delta_ci95": list(ci_d),
        "delta_excludes_zero": ci_d[0] > 0 or ci_d[1] < 0,
        "per_checkpoint": per_ckpt,
        "transitions": transitions(rows),
    }


def leave_one_out(family, lang):
    """delta recomputed with each evaluated checkpoint dropped in turn."""
    rows = cell_rows(family, lang)
    models = SCH_MODELS if family == "sch" else ROSTER
    out = {}
    for drop in range(len(rows)):
        kept = [r for j, r in enumerate(rows) if j != drop]
        out[models[drop]] = round(point(kept)[2], 1)
    return out


def permissive_delta(family, lang, seed_offset):
    """Delta under the deterministic permissive matcher policy."""
    rows = cell_rows(family, lang, permissive=True)
    mc, mf, d = point(rows)
    _, _, ci = paired_ci(rows, 1700 + seed_offset)
    return {"content": round(mc, 1), "form": round(mf, 1),
            "delta": round(d, 1), "delta_ci95": list(ci)}


def delta_draws(rows_by_model, seed, n_draws):
    """Sorted bootstrap draws of delta under the paired cluster resample."""
    grouped = []
    for rows in rows_by_model:
        groups = {}
        for row in rows:
            groups.setdefault(row[0], []).append(row[1:])
        grouped.append((list(groups), groups))
    rng = random.Random(seed)
    draws = []
    for _ in range(n_draws):
        total_c = total_f = 0.0
        for keys, groups in grouped:
            sample = [item for _ in keys
                      for item in groups[keys[rng.randrange(len(keys))]]]
            n = len(sample)
            total_c += 100 * sum(int(s[0]) - int(s[1]) for s in sample) / n
            total_f += 100 * sum(int(s[2]) - int(s[3]) for s in sample) / n
        k = len(grouped)
        draws.append(total_c / k - total_f / k)
    draws.sort()
    return draws


def signflip_p(rows_by_model, seed, n_perm=N_PERM):
    """Two-sided randomization p-value for delta from cluster-level sign flips.

    For item i, let d_i = (C_I - C_R) - (F_I - F_R).  Swapping the arm labels of
    that item maps d_i to -d_i, so under the null that the arm assignment is
    exchangeable within an answer cluster, flipping the sign of every item in a
    cluster produces an equally likely dataset.  We therefore draw a sign per
    cluster per checkpoint, recompute the checkpoint-equal macro, and compare
    the permutation distribution with the observed statistic.  Unlike a
    percentile-bootstrap tail proportion, this is centred under the null.
    """
    grouped = []
    for rows in rows_by_model:
        groups = {}
        for cluster, ci, cr, fi, fr in rows:
            d = (int(ci) - int(cr)) - (int(fi) - int(fr))
            groups.setdefault(cluster, []).append(d)
        grouped.append((list(groups.values()), sum(len(v) for v in groups.values())))
    observed = abs(sum(100 * sum(sum(g) for g in groups) / n
                       for groups, n in grouped) / len(grouped))
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n_perm):
        stat = 0.0
        for groups, n in grouped:
            total = 0
            for g in groups:
                total += sum(g) if rng.getrandbits(1) else -sum(g)
            stat += 100 * total / n
        if abs(stat / len(grouped)) >= observed - 1e-12:
            extreme += 1
    return (extreme + 1) / (n_perm + 1)


def multiplicity(family, lang, seed):
    """Selection-aware summary for one cell of the Spanish-prompt grid.

    The interval is a Bonferroni percentile interval at family-wise level 0.05
    over the twelve Spanish-prompt cells; because all twelve are reported and
    none was prespecified, that is the relevant correction for reading any one
    of them.  The accompanying p-value is the cluster-level sign-flip
    randomization test above, not a bootstrap tail proportion.
    """
    rows = cell_rows(family, lang)
    draws = delta_draws(rows, seed, B_MULT)
    n = len(draws)
    alpha = 0.05 / MULT_FAMILY_SIZE
    lo = draws[int(alpha / 2 * n)]
    hi = draws[int((1 - alpha / 2) * n)]
    return {"p_signflip": round(signflip_p(rows, seed + 77), 4),
            "bonferroni_ci": [round(lo, 1), round(hi, 1)],
            "survives": bool(lo > 0 or hi < 0)}


def cluster_disjoint_delta(family, lang, seed):
    """Delta after removing evaluation items sharing a calibration cluster."""
    rows = cell_rows(family, lang, cluster_disjoint=True)
    mc, mf, d = point(rows)
    _, _, ci = paired_ci(rows, seed)
    return {"n": sum(len(r) for r in rows), "content": round(mc, 1),
            "form": round(mf, 1), "delta": round(d, 1), "delta_ci95": list(ci)}


def verdict(effect, ci):
    """Prespecified success rule: positive effect whose paired CI excludes 0."""
    return effect > 0 and ci[0] > 0


# ---------------------------------------------------------------------- report
def collect():
    cells, i = [], 0
    for family in ("vT", "neuron", "anchor", "sch"):
        for lang in LANGS:
            cells.append(analyze(family, lang, i))
            i += 1
    for i, cell in enumerate(cells):
        cell["verdict_content"] = verdict(cell["content"], cell["content_ci95"])
        cell["verdict_form"] = verdict(cell["form"], cell["form_ci95"])
        cell["verdict_reversal"] = (cell["verdict_content"]
                                    and not cell["verdict_form"])
        cell["delta_leave_one_checkpoint_out"] = leave_one_out(
            cell["family"], cell["lang"])
        if cell["lang"] in ("ko", "ja"):
            cell["delta_permissive_policy"] = permissive_delta(
                cell["family"], cell["lang"], i)
        if cell["family"] in SPANISH_FAMILIES:
            cell["multiplicity"] = multiplicity(
                cell["family"], cell["lang"], 3000 + i)
            cell["delta_cluster_disjoint"] = cluster_disjoint_delta(
                cell["family"], cell["lang"], 2100 + i)
    return cells


def seed_sensitivity():
    """delta across the five preserved matched-random draws (v_T family only)."""
    out = {}
    for lang in LANGS:
        vals = []
        for seed in (7, 17, 27, 37, 47):
            vals.append(round(point(cell_rows("vT", lang, seed=seed))[2], 1))
        out[lang] = {"per_seed": vals, "min": min(vals), "max": max(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cells = collect()
    payload = {"policy": V.MATCHER_POLICY, "cells": cells,
               "vT_delta_seed_sensitivity": seed_sensitivity(),
               "verdict_reversals": [
                   f"{c['family']}/{c['lang']}" for c in cells
                   if c["verdict_reversal"]]}

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"policy = {V.MATCHER_POLICY}\n")
    head = f"{'cell':>18} {'dContent':>9} {'dForm':>9} {'delta':>9}  {'delta 95% CI':>18}  gained/in-form"
    print(head)
    print("-" * len(head))
    for c in cells:
        t = c["transitions"]
        lo, hi = c["delta_ci95"]
        mark = "*" if c["delta_excludes_zero"] else " "
        print(f"{c['family'] + '/' + c['lang']:>18} {c['content']:>9.1f} "
              f"{c['form']:>9.1f} {c['delta']:>8.1f}{mark} "
              f"[{lo:>+6.1f}, {hi:>+6.1f}]  {t['gained']:>3d}/{t['gained_in_form']:<3d}"
              f" ({t['share_in_form']}%)")
    print("\nverdict reversals (content success, form not):",
          ", ".join(payload["verdict_reversals"]) or "none")
    print("\nv_T delta across five matched-random draws:")
    for lang, v in payload["vT_delta_seed_sensitivity"].items():
        print(f"  {lang}: {v['per_seed']}  range [{v['min']}, {v['max']}]")
    print("\nleave-one-checkpoint-out delta (reversal cells):")
    for c in cells:
        if not c["verdict_reversal"]:
            continue
        print(f"  {c['family']}/{c['lang']}: "
              f"{sorted(c['delta_leave_one_checkpoint_out'].values())}")
    print("\npermissive-policy delta (ko/ja cells):")
    for c in cells:
        p = c.get("delta_permissive_policy")
        if p:
            print(f"  {c['family']}/{c['lang']}: locked {c['delta']:+.1f} "
                  f"-> permissive {p['delta']:+.1f} {p['delta_ci95']} "
                  f"(content {p['content']:+.1f}, form {p['form']:+.1f})")
    print(f"\nselection-aware (Bonferroni over {MULT_FAMILY_SIZE} cells, "
          f"alpha={0.05 / MULT_FAMILY_SIZE:.4f}; p = cluster sign-flip):")
    for c in cells:
        m = c.get("multiplicity")
        if m:
            print(f"  {c['family']}/{c['lang']:>2}: delta {c['delta']:+6.1f} "
                  f"p={m['p_signflip']:.4f} adj {m['bonferroni_ci']} "
                  f"{'SURVIVES' if m['survives'] else ''}")
    print("\ncluster-disjoint delta (calibration clusters removed):")
    for c in cells:
        cd = c.get("delta_cluster_disjoint")
        if cd:
            print(f"  {c['family']}/{c['lang']:>2}: n={cd['n']:>4} "
                  f"locked {c['delta']:+6.1f} -> disjoint {cd['delta']:+6.1f} "
                  f"{cd['delta_ci95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
