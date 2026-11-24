#!/usr/bin/env python3
"""Robustness checks for the Chinese and Korean language-neuron separations.

Reads the frozen result JSONs and computes three analyses for Chinese and Korean:

1. the primary paired estimand after excluding Gemma-4-31B base, with the two
   selection-aware criteria (12-cell Bonferroni interval and cluster sign-flip test);
2. a descriptive sensitivity restricted to items where both arms attempted an
   answer, using all five checkpoints; and
3. the same jointly-attempting sensitivity after excluding Gemma-4-31B base.

The excluded-Gemma-4-31B values are quoted in Section 5.3.
The jointly-attempting analyses condition on a post-intervention variable and
therefore remain diagnostics rather than replacements for the primary paired
estimand.  Seeds start at 6100 (adjusted inference: +1000; sign-flip: +77), so the
excluded-Gemma intervals differ by Monte Carlo error from the same quantities in
leave_one_out_selection.py, which uses the Table 1 seed convention; pass/fail agree.

Writes 02-data/results/checkpoint_exclusion/neuron_robustness.json.

    python3 neuron_robustness.py [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import paired_effects as sep
from result_paths import result_path


DROP_MODEL = "gemma4_31b"
LANGS = ("zh", "ko")
NEURON_FAMILY = "neuron"
ALPHA_FAMILYWISE = 0.05 / sep.MULT_FAMILY_SIZE

MARKER = re.compile(r"^(Instruction|Question|Answer|Pregunta|Respuesta)\s*[::]")


def first_line(text: str) -> str:
    for line in (text or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def is_degenerate(line: str) -> bool:
    if not line or MARKER.match(line):
        return True
    tokens = line.split()
    if len(tokens) >= 4:
        run = 1
        for left, right in zip(tokens, tokens[1:]):
            run = run + 1 if left == right else 1
            if run >= 4:
                return True
    squashed = re.sub(r"\s+", "", line)
    for ngram_size in (1, 2, 3):
        stop = max(0, len(squashed) - ngram_size * 4 + 1)
        for start in range(stop):
            gram = squashed[start : start + ngram_size]
            if gram and squashed[start : start + ngram_size * 4] == gram * 4:
                return True
    return False


def model_rows(model: str, lang: str, jointly_attempting: bool = False):
    v = sep.V
    data = v.load("interventions/" + v.FAMILIES[NEURON_FAMILY].format(model=model))
    chains = data["eval_chains"]
    floors = data["floors"]
    intervention = data["eval"][lang]["steer_items"]
    control = data["eval"][lang]["rand_items"]

    rows = []
    counts = Counter()
    for index, chain in enumerate(chains):
        if (
            chain not in v.CHAINS
            or chain not in floors
            or index >= len(intervention)
            or index >= len(control)
            or not v.scoreable(chain, lang)
        ):
            continue

        intervention_text = intervention[index].get("txt", "")
        control_text = control[index].get("txt", "")
        intervention_attempt = not is_degenerate(first_line(intervention_text))
        control_attempt = not is_degenerate(first_line(control_text))

        counts["eligible"] += 1
        counts["intervention_attempt"] += int(intervention_attempt)
        counts["control_attempt"] += int(control_attempt)
        counts["joint_attempt"] += int(intervention_attempt and control_attempt)

        if jointly_attempting and not (intervention_attempt and control_attempt):
            continue

        cluster = v.CHAINS[chain]["answer_label_by_lang"].get("en") or chain
        rows.append(
            (
                cluster,
                v.content(intervention_text, chain),
                v.content(control_text, chain),
                v.ilr(intervention_text, chain, lang),
                v.ilr(control_text, chain, lang),
            )
        )
    return rows, counts


def rows_for(models, lang: str, jointly_attempting: bool):
    rows_by_model = []
    counts_by_model = {}
    for model in models:
        rows, counts = model_rows(model, lang, jointly_attempting=jointly_attempting)
        if not rows:
            raise RuntimeError(f"No analysable rows for model={model}, lang={lang}, jointly_attempting={jointly_attempting}")
        rows_by_model.append(rows)
        counts_by_model[model] = dict(counts)
    return rows_by_model, counts_by_model


def adjusted_inference(rows_by_model, seed: int):
    draws = sep.delta_draws(rows_by_model, seed, sep.B_MULT)
    n_draws = len(draws)
    low = draws[int(ALPHA_FAMILYWISE / 2 * n_draws)]
    high = draws[int((1 - ALPHA_FAMILYWISE / 2) * n_draws)]
    p_value = sep.signflip_p(rows_by_model, seed + 77)
    return {
        "bonferroni_ci": [round(low, 1), round(high, 1)],
        "p_signflip": round(p_value, 4),
        "passes_interval": low > 0 or high < 0,
        "passes_signflip": p_value < ALPHA_FAMILYWISE,
        "passes_both": (low > 0 or high < 0) and p_value < ALPHA_FAMILYWISE,
    }


def analyze(models, lang: str, jointly_attempting: bool, seed: int):
    rows_by_model, counts = rows_for(models, lang, jointly_attempting)
    content, form, delta = sep.point(rows_by_model)
    content_ci, form_ci, delta_ci = sep.paired_ci(rows_by_model, seed)
    adjusted = adjusted_inference(rows_by_model, seed + 1000)
    checkpoints = []
    for model, rows in zip(models, rows_by_model):
        cc, cf, cd = sep.point([rows])
        checkpoints.append({"model": model, "n_analyzed": len(rows), "eligible": counts[model].get("eligible", 0),
                            "intervention_attempt": counts[model].get("intervention_attempt", 0),
                            "control_attempt": counts[model].get("control_attempt", 0),
                            "joint_attempt": counts[model].get("joint_attempt", 0),
                            "content_delta": round(cc, 1), "form_delta": round(cf, 1), "separation_delta": round(cd, 1)})
    return {"language": lang, "models": list(models), "jointly_attempting_only": jointly_attempting,
            "content_delta": round(content, 1), "content_ci95": list(content_ci),
            "form_delta": round(form, 1), "form_ci95": list(form_ci),
            "separation_delta": round(delta, 1), "separation_ci95": list(delta_ci),
            **adjusted, "checkpoints": checkpoints}


def collect():
    all_models = tuple(sep.ROSTER)
    models_without_gemma = tuple(model for model in all_models if model != DROP_MODEL)
    analyses = []
    analysis_specs = (
        ("drop_gemma_primary", models_without_gemma, False),
        ("joint_attempt_all_models_diagnostic", all_models, True),
        ("drop_gemma_joint_attempt_diagnostic", models_without_gemma, True),
    )
    seed = 6100
    for name, models, jointly_attempting in analysis_specs:
        for lang in LANGS:
            result = analyze(models, lang, jointly_attempting, seed)
            result["analysis"] = name
            analyses.append(result)
            seed += 1
    return {"excluded_checkpoint": DROP_MODEL, "familywise_alpha": ALPHA_FAMILYWISE, "analyses": analyses}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=result_path("checkpoint_exclusion", "neuron_robustness.json"))
    args = parser.parse_args()
    report = collect()
    with open(args.out, "w") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    for r in report["analyses"]:
        print(f"{r['analysis']:38s} {r['language']}: delta {r['separation_delta']:+.1f} CI {r['separation_ci95']} | "
              f"Bonferroni {r['bonferroni_ci']} p_flip {r['p_signflip']:.4f} both={r['passes_both']}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
