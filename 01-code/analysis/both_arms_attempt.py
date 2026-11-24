#!/usr/bin/env python3
"""Both-arms-attempt restriction for the Chinese and Korean language-neuron cells.

Restricts each paired item to those where both the intervention and the control
first line are answer attempts (not flagged by the degeneration rule of
attempt_decomposition.py) and re-estimates Delta_C, Delta_DFR and delta on that
subset, on all five checkpoints and on the four excluding Gemma-4-31B. It also
reports the exact decomposition of the unconditional macro delta over the four
attempt strata (11, 10, 01, 00).

Reuses the frozen scorer, population, degeneration rule and 4,000-draw paired
answer-cluster bootstrap. This is a post-generation subset analysis, not a
correction of the full-population effect. Reads the preserved language-neuron
generations in 02-data/results/interventions/ and writes results.json,
report.md and paired_items.csv to 02-data/results/both_arms_attempt/.

    python3 both_arms_attempt.py [--out-dir DIR]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import paired_effects as sep
from attempt_decomposition import first_line, is_degenerate, macro
from result_paths import DATA, REPO, RESULTS, result_path

HERE = Path(__file__).resolve().parent
STATES = ("11", "10", "01", "00")  # Intervention attempt, control attempt.
METRICS = ("content", "form", "delta")


def paired_records(model, lang):
    data = sep.V.load("interventions/" + sep.V.FAMILIES["neuron"].format(model=model))
    chains = data["eval_chains"]
    arms = data["eval"][lang]
    assert len(chains) == len(arms["steer_items"]) == len(arms["rand_items"])
    assert len(chains) == len(set(chains)), "Duplicate chain IDs"
    records = []
    for i, chain in enumerate(chains):
        if (chain not in sep.V.CHAINS or chain not in data["floors"]
                or not sep.V.scoreable(chain, lang)):
            continue
        texts = [arms[key][i].get("txt", "") for key in ("steer_items", "rand_items")]
        attempts = [not is_degenerate(first_line(t)) for t in texts]
        ci, cr = [int(sep.V.content(t, chain)) for t in texts]
        fi, fr = [int(sep.V.ilr(t, chain, lang)) for t in texts]
        assert fi <= ci and fr <= cr
        records.append(dict(model=model, lang=lang, chain=chain,
                            cluster=sep._cluster(chain),
                            attempt_I=int(attempts[0]), attempt_C=int(attempts[1]),
                            state="".join(str(int(a)) for a in attempts),
                            content_I=ci, content_C=cr, form_I=fi, form_C=fr))
    # Exact row-by-row agreement with the primary analysis, before filtering.
    assert score_rows(records) == sep.spanish_rows("neuron", model, lang)
    return records


def score_rows(records):
    return [(r["cluster"], r["content_I"], r["content_C"], r["form_I"], r["form_C"])
            for r in records]


def estimate(groups, seed):
    if any(not g for g in groups):
        return None  # Never silently drop a checkpoint or change the roster.
    point = sep.point(groups)
    ci = sep.paired_ci(groups, seed)
    return {name: {"estimate": value, "ci95": list(bounds)}
            for name, value, bounds in zip(METRICS, point, ci)}


def analyze(lang, seed):
    records_by_model = [paired_records(model, lang) for model in sep.ROSTER]
    all_rows = [score_rows(rs) for rs in records_by_model]
    both_rows = [score_rows([r for r in rs if r["state"] == "11"])
                 for rs in records_by_model]
    checkpoints = []
    macro_contributions = {state: [0., 0., 0.] for state in STATES}
    for j, (model, records, rows, both) in enumerate(
            zip(sep.ROSTER, records_by_model, all_rows, both_rows)):
        n = len(rows)
        strata = {}
        for state in STATES:
            subset = score_rows([r for r in records if r["state"] == state])
            pt = sep.point([subset]) if subset else (0., 0., 0.)
            contribution = [len(subset) / n * x for x in pt]
            strata[state] = dict(n=len(subset),
                                 within_stratum=dict(zip(METRICS, pt)) if subset else None,
                                 unconditional_contribution=dict(zip(METRICS, contribution)))
            for k, value in enumerate(contribution):
                macro_contributions[state][k] += value / len(sep.ROSTER)
        primary_point = sep.point([rows])
        for k, metric in enumerate(METRICS):
            assert abs(sum(s["unconditional_contribution"][metric]
                           for s in strata.values()) - primary_point[k]) < 1e-10
        checkpoints.append(dict(
            model=model, n=n, n_both=len(both), coverage_pct=100 * len(both) / n,
            both_clusters=len({r[0] for r in both}), strata=strata,
            all=dict(zip(METRICS, primary_point)),
            both=estimate([both], seed + 100 + j),
            degenerate_content_hits={tag: sum(r["content_" + tag]
                                               for r in records if not r["attempt_" + tag])
                                     for tag in ("I", "C")}))
    keep = [i for i, model in enumerate(sep.ROSTER) if model != "gemma4_31b"]
    result = dict(
        lang=lang, n=sum(map(len, all_rows)), n_both=sum(map(len, both_rows)),
        all=estimate(all_rows, seed), both=estimate(both_rows, seed + 1),
        without_gemma=dict(models=[sep.ROSTER[i] for i in keep],
                           all=estimate([all_rows[i] for i in keep], seed + 2),
                           both=estimate([both_rows[i] for i in keep], seed + 3)),
        checkpoints=checkpoints,
        stratum_contributions={state: dict(zip(METRICS, values))
                               for state, values in macro_contributions.items()},
        legacy_arm_conditioned=macro(vars(sep.V), "neuron", lang))
    # Check that the primary point effects reproduce and that the decomposition is exact.
    expected = {"zh": 7.3, "ko": 8.8}[lang]
    assert round(result["all"]["delta"]["estimate"], 1) == expected
    for metric in METRICS:
        assert abs(sum(s[metric] for s in result["stratum_contributions"].values())
                   - result["all"][metric]["estimate"]) < 1e-10
    return result, [r for rs in records_by_model for r in rs]


def fmt(est, metric="delta"):
    if est is None:
        return "undefined"
    x = est[metric]
    return f"{x['estimate']:+.1f} [{x['ci95'][0]:+.1f}, {x['ci95'][1]:+.1f}]"


def report(payload):
    out = ["# Paired common-attempt analysis", "",
           "Verification status: VERIFIED for primary point reproduction, paired row alignment, "
           "and exact stratum decomposition; new subset estimates are exploratory.", "",
           "Scope: Chinese and Korean language-neuron amplification, frozen five-checkpoint roster. "
           "An attempt means not flagged by the existing first-line degeneration heuristic; "
           "it does not guarantee a valid or on-task answer.", "",
           "All effects are percentage points. Brackets are unadjusted 95% paired answer-cluster "
           "bootstrap intervals (4,000 draws). Clusters are resampled independently within each "
           "checkpoint, as in paired_effects.py; checkpoint effects are equally weighted. "
           "The retained subset and checkpoint roster are held fixed, with no model resampling. "
           "Seeds and input hashes are in the JSON companion.", "",
           "## Macro effects", "",
           "| Language | Both-attempt pairs / all | All delta | Both content | Both form | Both delta |",
           "|---|---:|---:|---:|---:|---:|"]
    for r in payload["results"]:
        out.append(f"| {r['lang']} | {r['n_both']}/{r['n']} | {fmt(r['all'])} | "
                   f"{fmt(r['both'], 'content')} | {fmt(r['both'], 'form')} | {fmt(r['both'])} |")
    out += ["", "## Checkpoint effects and attempt states", "",
            "State order is intervention/control: 11 = both attempt, 10 = intervention only, "
            "01 = control only, 00 = neither. Counts are item–checkpoint observations.", "",
            "| Language | Checkpoint | N | 11 | 10 | 01 | 00 | Coverage | Clusters in 11 | All delta | Both delta |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in payload["results"]:
        for c in r["checkpoints"]:
            counts = " | ".join(str(c["strata"][s]["n"]) for s in STATES)
            out.append(f"| {r['lang']} | {c['model']} | {c['n']} | {counts} | "
                       f"{c['coverage_pct']:.1f}% | {c['both_clusters']} | "
                       f"{c['all']['delta']:+.1f} | {fmt(c['both'])} |")
    out += ["", "## Same four-checkpoint comparison excluding Gemma-4-31B", "",
            "This sensitivity removes Gemma from BOTH analyses; it does not compare macros "
            "with different checkpoint rosters.", "",
            "| Language | All delta | Both delta |", "|---|---:|---:|"]
    for r in payload["results"]:
        out.append(f"| {r['lang']} | {fmt(r['without_gemma']['all'])} | "
                   f"{fmt(r['without_gemma']['both'])} |")
    out += ["", "## Exact decomposition of the unconditional macro", "",
            "Each contribution is mean_checkpoint[(N_state/N_all) × delta_state]. "
            "Contributions sum to the primary macro. They are algebraic strata, not causal mediation effects.", "",
            "| Language | 11 contribution | 10 contribution | 01 contribution | 00 contribution | Total |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in payload["results"]:
        parts = " | ".join(f"{r['stratum_contributions'][s]['delta']:+.2f}" for s in STATES)
        out.append(f"| {r['lang']} | {parts} | {r['all']['delta']['estimate']:+.2f} |")
    out += ["", "## Interpretation limits", "",
            "- Both-attempt conditioning selects on observed generation outcomes. It estimates "
            "the contrast on these paired retained items, not a corrected full-population effect.",
            "- Gemma's tiny retained sample still receives 20% macro weight. Resampling these "
            "few clusters cannot establish generalization beyond the observed support.",
            "- The attempt-decomposition table (App. C.1, Table 13) conditions each arm separately; its denominators differ from this analysis. "
            "The JSON retains those per-arm estimates for comparison and counts content hits on "
            "degenerate outputs to check their numerator assumption.",
            "- These intervals have no multiplicity correction or new sign-flip test. "
            "Existing unconditional selection-aware results do not transfer to this subset.",
            "- Random control draw and generation are fixed. Control functional strength remains "
            "unmatched. The heuristic can miss fluent off-task output.", "",
            "## Reading these results", "",
            "The Chinese five-checkpoint common-attempt interval includes zero. Gemma changes "
            "from +15.8 to -16.7 points on its 18 retained pairs, while the four-checkpoint macro "
            "excluding Gemma remains positive. Thus the five-checkpoint claim is sensitive to "
            "attempt conditioning; these results do not show that all Chinese separation is an artifact.", "",
            "The Korean common-attempt macro remains positive, but Gemma's +33.3 points comes "
            "from one net content-match difference among only three retained pairs. At 20% "
            "checkpoint weight that single pair contributes +6.67 points to the macro. Its "
            "control first line is 'El banco central del país de origen de Matoke es el Banco de "
            "la República de Uganda.' and the intervention first line is 'El Banco de Uganda.' "
            "The locked matcher credits only the latter; no semantic re-adjudication was used. "
            "The paired_items.csv row P495_P1304_Q1626853 identifies this observation.", "",
            "The exact full-population decomposition assigns +3.67 of +7.30 Chinese points "
            "and +4.83 of +8.83 Korean points to intervention-only attempts. The 11-stratum "
            "contribution differs from the common-attempt macro because the former retains "
            "the full-population denominator within each checkpoint, whereas the latter "
            "renormalizes each retained checkpoint subset.", "",
            "The all-population CI endpoints can differ slightly from the manuscript because "
            "this analysis records new bootstrap seeds; the primary point estimates reproduce. "
            "The four-checkpoint Korean lower bound is close to zero and should not be treated "
            "as strong confirmatory evidence.", ""]
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(result_path("both_arms_attempt")))
    args = parser.parse_args()
    assert sep.V.MATCHER_POLICY == "attribution_strict"
    results, records = [], []
    for lang, seed in (("zh", 21000), ("ko", 22000)):
        result, rows = analyze(lang, seed)
        results.append(result)
        records.extend(rows)
        print(f"{lang}: all {fmt(result['all'])}; both {fmt(result['both'])}; "
              f"retained {result['n_both']}/{result['n']}", flush=True)
    paths = [Path(__file__), HERE / "paired_effects.py", HERE / "attempt_decomposition.py",
             HERE / "verify_numbers.py", Path(DATA) / "inputs/chain_lang_prompts.json"]
    paths += [Path(RESULTS) / "interventions" / sep.V.FAMILIES["neuron"].format(model=m)
              for m in sep.ROSTER]
    payload = dict(family="neuron", matcher=sep.V.MATCHER_POLICY, bootstrap_draws=sep.B,
                   base_seeds={"zh": 21000, "ko": 22000},
                   seed_offsets={"all": 0, "both": 1, "four_all": 2, "four_both": 3,
                                 "checkpoint_both": "100 + index in sorted roster"},
                   roster=sep.ROSTER,
                   sha256={str(p.relative_to(Path(REPO))): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in paths}, results=results)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    (args.out_dir / "report.md").write_text(report(payload))
    with (args.out_dir / "paired_items.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


if __name__ == "__main__":
    main()
