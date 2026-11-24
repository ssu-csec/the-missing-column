"""Dependency-free success/failure cohort collection for frozen language-neuron interventions.

Screens the untreated model, freezes a baseline-success cohort and the frozen failure (evaluation)
cohort on disk, then runs every arm on both cohorts with paired recording. The decoder and scorer
are injected so selection and recording can be tested without loading a model. Frozen failure IDs
are kept as given; they are not re-screened against the untreated output.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import random


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + "\n", encoding="utf-8")


def first_line(text):
    return next((s.strip() for s in text.splitlines() if s.strip()), "")


def build_arms(best, targets, seeds, doses, suppression_modes, prompt_lang="es"):
    """One identical arm specification list for both cohorts; no outcome input."""
    if not seeds or len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds):
        raise ValueError("Control seeds must be nonempty, unique and nonnegative")
    if (not doses or len(set(doses)) != len(doses) or 1.0 not in doses
            or any(not math.isfinite(d) or not 0 <= d <= 1 for d in doses)):
        raise ValueError("Dose fractions must be unique, in [0,1], and include 1")
    if not suppression_modes or set(suppression_modes) - {"original", "off"}:
        raise ValueError("Suppression modes must be original and/or off")
    result = {}
    for target in targets:
        a, s = float(best[target]["amp"]), float(best[target]["supp"])
        if not math.isfinite(a) or a < 0 or not math.isfinite(s) or not 0 <= s <= 1:
            raise ValueError(f"Invalid frozen configuration for {target}")
        levels = list(dict.fromkeys(1.0 if target == prompt_lang or mode == "off" else s
                                   for mode in suppression_modes))
        arms = [dict(arm_id="U", kind="untreated", seed=None, amp=0.,
                     dose=0., suppression=1.)]
        for supp in levels:
            if supp != 1:
                arms.append(dict(arm_id=f"S:s={supp:g}", kind="suppression", seed=None,
                                 amp=0., dose=0., suppression=supp))
            for dose in doses:
                if dose == 0:
                    continue  # U/S already records the physical zero-dose condition.
                for kind, seed in [("selected", None), *[("random", s) for s in seeds]]:
                    prefix = "I" if kind == "selected" else f"R:seed={seed}"
                    arms.append(dict(arm_id=f"{prefix}:d={dose:g}:s={supp:g}", kind=kind,
                                     seed=seed, amp=a * dose, dose=dose, suppression=supp))
        result[target] = arms
    return result


def select_cohorts(chains, frozen, baseline, n_success, sample_seed):
    calib, failure = frozen["calib_chains"], frozen["eval_chains"]
    if n_success < 0:
        raise ValueError("n_success must be >= 0 (0 means all eligible successes)")
    if (len(set(calib)) != len(calib) or len(set(failure)) != len(failure)
            or set(calib) & set(failure)):
        raise ValueError("Frozen calibration/evaluation lists must be unique and disjoint")
    if set(calib + failure) - set(baseline):
        raise ValueError("Frozen chains are missing from the baseline screen")

    def cluster(chain):
        return chains[chain]["answer_label_by_lang"].get("en") or chain

    calibration_clusters = {cluster(c) for c in calib}
    excluded = {}
    candidates = []
    for chain, row in baseline.items():
        if not row["baseline_content"]:
            continue
        if chain in calib:
            excluded[chain] = "calibration_item"
        elif chain in failure:
            excluded[chain] = "historical_failure_replay_drift"
        elif cluster(chain) in calibration_clusters:
            excluded[chain] = "calibration_answer_cluster"
        else:
            candidates.append(chain)
    random.Random(sample_seed).shuffle(candidates)
    successes = candidates if n_success == 0 else candidates[:n_success]
    return dict(
        frozen_failure=list(failure), baseline_success=successes,
        calibration_chains=list(calib), excluded_successes=excluded,
        success_candidate_count=len(candidates), requested_successes=n_success,
        selected_successes=len(successes), sample_seed=sample_seed,
        success_shortfall=max(0, n_success - len(successes)) if n_success else 0,
        unselected_success_candidates=[c for c in candidates if c not in set(successes)],
        failure_baseline_drift=[c for c in failure if baseline[c]["baseline_content"]],
        selection="Untreated content only; success excludes calibration items/answer clusters; "
                  "frozen_failure preserves historical IDs even if baseline replay changes")


def summarize(records):
    """Separate cohorts; score both outcomes on the same attribution-eligible rows."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["cohort"], row["target_language"], row["arm_id"])].append(row)
    output = []
    for (cohort, lang, arm_id), rows in sorted(groups.items()):
        eligible = [r for r in rows if r["score"]["eligible"]]
        success = [r for r in eligible if r["baseline_score"]["content"]]
        failure = [r for r in eligible if not r["baseline_score"]["content"]]

        def rate(items, predicate):
            return 100 * sum(bool(predicate(r)) for r in items) / len(items) if items else None

        counts = {state: sum(f"{int(not r['score']['degenerate'])}"
                             f"{int(not r['baseline_score']['degenerate'])}" == state
                             for r in eligible) for state in ("11", "10", "01", "00")}
        output.append(dict(
            cohort=cohort, target_language=lang, arm_id=arm_id,
            n_all=len(rows), n_eligible=len(eligible), n_excluded=len(rows)-len(eligible),
            n_baseline_content_correct=len(success), n_baseline_content_incorrect=len(failure),
            content_rate_pct=rate(eligible, lambda r: r["score"]["content"]),
            form_rate_pct=rate(eligible, lambda r: r["score"]["form"]),
            degeneration_rate_pct=rate(rows, lambda r: r["score"]["degenerate"]),
            degeneration_eligible_pct=rate(eligible, lambda r: r["score"]["degenerate"]),
            content_retention_pct=rate(success, lambda r: r["score"]["content"]),
            content_loss_pct=rate(success, lambda r: not r["score"]["content"]),
            content_recovery_pct=rate(failure, lambda r: r["score"]["content"]),
            form_gained=sum(not r["baseline_score"]["form"] and r["score"]["form"] for r in eligible),
            form_lost=sum(r["baseline_score"]["form"] and not r["score"]["form"] for r in eligible),
            attempt_states_arm_baseline=counts))
    return output


def run_audit(*, chains, frozen, arms, decode, score, output, n_success=120,
              sample_seed=53, progress=print):
    """Screen first, freeze cohorts on disk, then run ALL arms on BOTH cohorts.

    decode(chain, target_language_or_None, arm) -> {text, token_ids, ...}.
    score(text, chain, language) -> {content, form, eligible, degenerate}.
    The output directory must exist. JSONL is flushed after every generation.
    """
    output = Path(output)
    prompt_lang = frozen["protocol"].get("fail_lang", "es")
    pivot = frozen["protocol"].get("pivot", "en")
    targets = list(arms)
    candidates = [c for c, m in chains.items() if m["prompts"].get(prompt_lang)
                  and m["answer_label_by_lang"].get(prompt_lang)
                  and m["answer_label_by_lang"].get(pivot)]
    # Fail before any costly generation if the frozen populations cannot be re-generated.
    if set(frozen["calib_chains"] + frozen["eval_chains"]) - set(candidates):
        raise ValueError("Some frozen chains lack required prompts/labels")
    if set(frozen["calib_chains"]) & set(frozen["eval_chains"]):
        raise ValueError("Frozen calibration/evaluation overlap")
    untreated = dict(kind="untreated", arm_id="U", seed=None, amp=0., dose=0., suppression=1.)
    baseline = {}
    with (output / "screen.jsonl").open("x", encoding="utf-8") as fh:
        for i, chain in enumerate(candidates):
            generation = decode(chain, None, untreated)
            scores = {t: score(generation["text"], chain, t) for t in targets}
            contents = {s["content"] for s in scores.values()}
            if len(contents) != 1:
                raise ValueError("Content scorer must be target-language independent")
            row = dict(chain=chain, baseline_content=contents.pop(), generation=generation,
                       scores=scores)
            baseline[chain] = row
            fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            fh.flush()
            progress(f"[screen] {i+1}/{len(candidates)} {chain}")
    cohorts = select_cohorts(chains, frozen, baseline, n_success, sample_seed)
    save_json(output / "cohorts.json", cohorts)
    # This file exists before the first intervention call: selection cannot depend on it.
    save_json(output / "arms.json", arms)
    progress(f"[cohorts] historical failures={len(cohorts['frozen_failure'])}; "
             f"successes={cohorts['selected_successes']}; shortfall={cohorts['success_shortfall']}; "
             f"failure replay drift={len(cohorts['failure_baseline_drift'])}")
    records = []
    with (output / "generations.jsonl").open("x", encoding="utf-8") as fh:
        for cohort in ("frozen_failure", "baseline_success"):
            for chain in cohorts[cohort]:
                for target, specs in arms.items():
                    for arm in specs:
                        generation = (baseline[chain]["generation"] if arm["kind"] == "untreated"
                                      else decode(chain, target, arm))
                        scored = score(generation["text"], chain, target)
                        prior = None
                        if chain in frozen["eval_chains"]:
                            idx = frozen["eval_chains"].index(chain)
                            config = frozen["best"][target]
                            original_s = 1. if target == prompt_lang else config["supp"]
                            if arm["kind"] == "untreated":
                                prior = frozen.get("floors", {}).get(chain)
                            elif arm["dose"] == 1 and arm["suppression"] == original_s:
                                key = ("steer_items" if arm["kind"] == "selected" else
                                       "rand_items" if arm["kind"] == "random" and arm["seed"] == 7 else None)
                                if key:
                                    prior = frozen["eval"][target][key][idx]["txt"]
                        row = dict(model=frozen["model"], cohort=cohort, chain=chain,
                                   answer_cluster=chains[chain]["answer_label_by_lang"].get("en") or chain,
                                   prompt_language=prompt_lang, target_language=target, **arm,
                                   generation=generation, scored_line=first_line(generation["text"]),
                                   baseline_score=baseline[chain]["scores"][target], score=scored,
                                   historical_text_exact=(generation["text"] == prior) if prior is not None else None,
                                   historical_first_line_exact=(first_line(generation["text"]) == first_line(prior))
                                   if prior is not None else None)
                        records.append(row)
                        fh.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                        fh.flush()
                    progress(f"[eval] {cohort} {chain} {target}")
    save_json(output / "summary.json", summarize(records))
    compared = [r for r in records if r["historical_text_exact"] is not None]
    replay = dict(compared_records=len(compared),
                  text_mismatches=sum(not r["historical_text_exact"] for r in compared),
                  first_line_mismatches=sum(not r["historical_first_line_exact"] for r in compared),
                  failure_baseline_drift=cohorts["failure_baseline_drift"],
                  note="U repeats across languages; counts are records, not unique generations. "
                       "Mismatch is disclosed, never silently merged with historical scores.")
    save_json(output / "replay_validation.json", replay)
    return dict(cohorts=cohorts, n_records=len(records), replay=replay)
