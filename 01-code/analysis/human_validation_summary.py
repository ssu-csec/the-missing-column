#!/usr/bin/env python3
"""Validate and summarize returned sheets of the grid-wide human validation.

Returned judgments are read from annotations_<lang>_result.csv, or from one combined
annotations_all_result.csv (as in the stage-1 package). The annotation package (reference sheets,
schema, and guide) is not distributed; its returned judgments and item key are in
02-data/results/human_form_audit/grid_validation/.

    python3 01-code/analysis/human_validation_summary.py \
        --package <package directory> --returned <directory with annotations_<lang>_result.csv> \
        --output <new directory> [--allow-incomplete]

Checks immutable columns, labels and conditional rules against the package schema, joins
the admin key, and writes joined_admin.csv, issues.csv, missed_forms.csv and summary.json
following the analysis contract in admin/ADMIN.md.  Judgments are never changed:
uncertain and unassessable rows are counted separately and blank rows are unfinished.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

LANGS = ("zh", "ja", "ko", "es")
UNCREDITED = ("script_uncredited", "es_uncredited_content", "es_uncredited_other")


def clopper_pearson(k, n, level=0.95):
    """Exact two-sided binomial interval, by bisection on the binomial CDF."""
    if n == 0:
        return None
    a = (1 - level) / 2

    def cdf(x, p):
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(x + 1))

    def solve(f):
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: cdf(k - 1, p) > 1 - a)
    upper = 1.0 if k == n else solve(lambda p: cdf(k, p) > a)
    return [round(lower, 4), round(upper, 4)]


def human_form(r, specific=False):
    ok = r["content_status"] == "correct" and r["target_form_status"] == "acceptable"
    return ok and (not specific or r["attribution_status"] == "target_specific")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--package", required=True, type=Path)
    ap.add_argument("--returned", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; choose a new directory.")

    pkg, admin = args.package / "annotator_package", args.package / "admin"
    schema = json.loads((pkg / "annotation_schema.json").read_text())
    key = {r["item_id"]: r for r in map(json.loads, (admin / "item_key.jsonl").read_text().splitlines())}
    refs = {r["item_id"]: r for r in map(json.loads, (pkg / "reference_items.jsonl").read_text().splitlines())}

    joined, issues = [], []
    combined_path = args.returned / "annotations_all_result.csv"
    combined = None
    if combined_path.exists():
        with combined_path.open(encoding="utf-8-sig", newline="") as fh:
            combined = list(csv.DictReader(fh))
    for lang in LANGS:
        path = args.returned / f"annotations_{lang}_result.csv"
        if combined is not None:
            rows = [r for r in combined if r["target_language"] == lang]
            if not rows and not any(k["lang"] == lang for k in key.values()):
                continue
        elif not path.exists():
            if args.allow_incomplete:
                continue
            raise SystemExit(f"missing {path}")
        else:
            with path.open(encoding="utf-8-sig", newline="") as fh:
                rows = list(csv.DictReader(fh))
        label = f"{combined_path.name} ({lang})" if combined is not None else path.name
        ids = [r["item_id"] for r in rows]
        expected = {i for i, k in key.items() if k["lang"] == lang}
        if len(set(ids)) != len(ids):
            raise SystemExit(f"{label}: duplicate item IDs")
        if set(ids) != expected:
            raise SystemExit(f"{label}: item IDs differ from the package")
        for r in rows:
            i = r["item_id"]
            if any(r[c] != refs[i][c] for c in schema["immutable_columns"]):
                raise SystemExit(f"{label}: reference columns changed for {i}")
            c, f, a = r["content_status"], r["target_form_status"], r["attribution_status"]
            flags = []
            if not (c or f or a):
                flags.append("unfinished")
            else:
                for field, allowed in schema["allowed_labels"].items():
                    if r[field] not in allowed:
                        flags.append("invalid_" + field)
                if c == "correct" and f == "not_applicable":
                    flags.append("correct_content_requires_form_decision")
                if c != "correct" and f != "not_applicable":
                    flags.append("noncorrect_content_requires_not_applicable_form")
                if (f == "acceptable") == (a == "not_applicable"):
                    flags.append("attribution_conditional_rule")
                if f in ("acceptable", "uncertain") and r["form_span"].strip() not in r["scored_line"]:
                    flags.append("form_span_not_in_scored_line")
                if f in ("acceptable", "uncertain") and not r["form_span"].strip():
                    flags.append("missing_form_span")
                needs = schema.get("rationale_required", True) and ("uncertain" in (c, f, a) or f == "acceptable")
                if needs and not r["rationale"].strip():
                    flags.append("missing_rationale")
            k = key[i]
            row = {**r, **{x: k[x] for x in ("stratum", "family", "model", "lang", "arm", "chain", "index",
                                              "content_locked", "form_locked", "prior_audit_overlap")},
                   "issues": ";".join(flags)}
            joined.append(row)
            if flags:
                issues.append(row)
    if any("unfinished" in r["issues"] for r in joined) and not args.allow_incomplete:
        raise SystemExit("unfinished rows remain; rerun with --allow-incomplete to summarize anyway")

    done = [r for r in joined if "unfinished" not in r["issues"]]
    by = defaultdict(list)
    for r in done:
        by[(r["stratum"], r["lang"])].append(r)

    summary = dict(n_returned=len(joined), n_judged=len(done),
                   n_rows_with_issues=len(issues), strata={})
    for (s, lang), rs in sorted(by.items()):
        n = len(rs)
        content = Counter(r["content_status"] for r in rs)
        form = Counter(r["target_form_status"] for r in rs if r["content_status"] == "correct")
        attr = Counter(r["attribution_status"] for r in rs if r["target_form_status"] == "acceptable")
        k_form = sum(human_form(r) for r in rs)
        k_spec = sum(human_form(r, specific=True) for r in rs)
        entry = dict(n=n, content=dict(content), form_among_correct=dict(form),
                     attribution_among_acceptable=dict(attr),
                     human_form=k_form, human_form_ci95=clopper_pearson(k_form, n),
                     human_form_target_specific=k_spec,
                     human_form_target_specific_ci95=clopper_pearson(k_spec, n),
                     locked_content_vs_human=dict(Counter(
                         f"locked{r['content_locked']}/human_{r['content_status']}" for r in rs)))
        if s == "credited_check":
            entry["precision"] = round(k_form / n, 4) if n else None
        summary["strata"][f"{s}/{lang}"] = entry

    missed = [r for r in done if r["stratum"] in UNCREDITED and human_form(r)]
    cells = Counter((r["family"], r["lang"], r["arm"]) for r in missed)
    summary["missed_forms_by_cell"] = [dict(family=f, lang=l, arm=a, n=n) for (f, l, a), n in sorted(cells.items())]

    args.output.mkdir(parents=True)
    fields = list(joined[0]) if joined else []
    for name, rows in (("joined_admin.csv", joined), ("issues.csv", issues), ("missed_forms.csv", missed)):
        with (args.output / name).open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
