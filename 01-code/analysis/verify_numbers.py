#!/usr/bin/env python3
"""Verify the numbers reported in the paper from the released result files.

The first part recomputes the main-audit aggregates from the preserved generations in interventions/; the final
block compares the numbers printed in the paper, stored in PAPER at the precision printed there, with their
recomputation from the per-analysis result folders. Set RESULTS_DIR to check another copy of the results.

Run from any directory:

    python 01-code/analysis/verify_numbers.py

The checker uses only the Python standard library.  It exits non-zero on any
mismatch and checks the scoring/aggregation convention used in the manuscript:
equal-model family effects, gate-qualified general-pool SCH, attribution-strict
Spanish answer forms, and separate specific-pool sensitivity results. Set
``AOL_MATCHER_POLICY=designated`` to reproduce the shared-form sensitivity.
"""
from __future__ import annotations

import json
import glob
import math
import os
import random
import re
import sys
import unicodedata
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.environ.get("RESULTS_DIR") or os.path.join(HERE, "..", "..", "02-data", "results"))
DATA = os.path.dirname(RESULTS)
CHAINS = json.load(open(os.path.join(DATA, "inputs", "chain_lang_prompts.json")))
FAIL = []
PASS = 0

# Primary Spanish attribution excludes items whose normalized Spanish and
# English inventories overlap. Exclusion is item-level and symmetric.
MATCHER_POLICY = os.environ.get("AOL_MATCHER_POLICY", "attribution_strict")
if MATCHER_POLICY not in {"attribution_strict", "designated"}:
    raise ValueError(f"unknown AOL_MATCHER_POLICY={MATCHER_POLICY!r}")


def load(path):
    with open(os.path.join(RESULTS, path)) as fh:
        return json.load(fh)


def check(label, got, want):
    global PASS
    ok = got == want
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {got}")
    if ok:
        PASS += 1
    else:
        FAIL.append(f"{label}: got {got!r}, expected {want!r}")


def near(label, got, want, tol=0.06):
    global PASS
    ok = abs(got - want) <= tol
    print(f"[{'PASS' if ok else 'FAIL'}] {label}: {got:.2f}")
    if ok:
        PASS += 1
    else:
        FAIL.append(f"{label}: got {got:.4f}, expected {want:.4f} +/- {tol}")


# ---------------------------------------------------------------- scoring inventory
ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "un", "una"}


def fold(text):
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", text)


def norm(text):
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    toks = re.sub(r"[^\w\s]", " ", fold(line).lower(), flags=re.UNICODE).split()
    if toks and toks[0] in ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


def has_hangul(s):
    return any("\uac00" <= c <= "\ud7a3" for c in (s or ""))


def has_kana(s):
    return any("\u3040" <= c <= "\u30ff" for c in (s or ""))


def has_han(s):
    return any("\u3400" <= c <= "\u9fff" for c in (s or ""))


def cjk(s):
    return has_hangul(s) or has_kana(s) or has_han(s)


def native(form, lang):
    if lang == "ko":
        return has_hangul(form)
    if lang == "ja":
        return has_kana(form)
    if lang == "zh":
        return has_han(form) and not has_kana(form) and not has_hangul(form)
    return False


def inventory(chain, lang, permissive=False):
    """Return the designated-language inventory for one chain.

    The locked inventory contains the canonical label and script-native source
    aliases.  The permissive policy changes normalization, not entity identity.
    """
    meta = CHAINS[chain]
    forms = [meta["answer_label_by_lang"].get(lang)]
    forms += [x for x in meta.get("answer_aliases_by_lang", {}).get(lang, [])
              if x and native(x, lang)]
    return list(dict.fromkeys(x for x in forms if x))


def all_forms(chain, permissive=False):
    meta = CHAINS[chain]
    forms = []
    for lang in ("ko", "ja", "zh"):
        forms += inventory(chain, lang, permissive=permissive)
    for lang in ("es", "en"):
        forms.append(meta["answer_label_by_lang"].get(lang))
    return list(dict.fromkeys(x for x in forms if x))


def matches(text, forms, permissive=False):
    """Match frozen answer forms without semantic adjudication.

    Locked CJK matching preserves normalized token spacing.  Permissive CJK
    matching additionally ignores whitespace and punctuation separators, which
    covers Korean spacing and Japanese middle-dot/interpunct variation.  It
    does not admit abbreviations, alternative institutions, or contextual
    entity judgments.
    """
    p = norm(text)
    toks = p.split()
    p_compact = SQUASH.sub("", p) if permissive else ""
    for form in forms:
        f = norm(form)
        if not f:
            continue
        if cjk(f):
            candidate = SQUASH.sub("", f) if permissive else f
            haystack = p_compact if permissive else p
            if candidate in haystack:
                return True
        else:
            ft = f.split()
            if any(toks[i:i + len(ft)] == ft for i in range(len(toks) - len(ft) + 1)):
                return True
    return False


def scoreable(chain, lang):
    if lang == "ja":
        return has_kana(CHAINS[chain]["answer_label_by_lang"].get("ja"))
    if lang == "es" and MATCHER_POLICY == "attribution_strict":
        meta = CHAINS[chain]
        spanish = inventory(chain, "es")
        english = [meta["answer_label_by_lang"].get("en")]
        spanish_norm = {norm(x) for x in spanish if norm(x)}
        english_norm = {norm(x) for x in english if norm(x)}
        return spanish_norm.isdisjoint(english_norm)
    return True


def ilr(text, chain, lang, permissive=False):
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    gate = {
        "ko": has_hangul(line),
        "ja": has_kana(line),
        "zh": has_han(line) and not has_kana(line) and not has_hangul(line),
        "es": bool(re.search(r"[A-Za-z]", line)),
    }[lang]
    return gate and matches(
        line, inventory(chain, lang, permissive=permissive),
        permissive=permissive,
    )


def content(text, chain, permissive=False):
    line = (text or "").strip().splitlines()[0] if (text or "").strip() else ""
    return matches(
        line, all_forms(chain, permissive=permissive),
        permissive=permissive,
    )


SQUASH = re.compile(r"[\s·・‧゠\-]")




# ------------------------------- main-grid roster families (Table 1, tab:paired-roster)
ROSTER = ("llama31_70", "llama31_70_it", "gemma4_31b",
          "gemma4_31b_it", "qwen36_27b")
FAMILIES = {
    "vT": "surfacing_vector_{model}.json",
    "neuron": "language_neurons_{model}.json",
    "anchor": "mrre_{model}.json",
}
# Table 1 effects (Delta_C, Delta_F), recomputed item by item from the intervention JSONs.
TABLE1 = {
    "vT": {
        "es": (13.0, 15.6), "zh": (11.3, 8.4),
        "ja": (2.5, 0.2), "ko": (8.7, 0.0),
    },
    "neuron": {
        "es": (-0.7, -1.3), "zh": (7.3, 0.0),
        "ja": (-5.9, 0.0), "ko": (8.8, 0.0),
    },
    "anchor": {
        "es": (3.3, 4.5), "zh": (-2.0, 0.5),
        "ja": (-1.0, 0.7), "ko": (2.2, 1.2),
    },
}
PAIRED_INTERVALS = {
    "vT": {
        "es": ((6.2, 19.1), (10.0, 20.8)), "zh": ((5.9, 16.4), (6.0, 10.8)),
        "ja": ((-2.2, 7.5), (0.0, 0.6)), "ko": ((2.4, 15.3), (0.0, 0.0)),
    },
    "neuron": {
        "es": ((-4.5, 3.8), (-4.8, 2.7)), "zh": ((4.0, 11.0), (0.0, 0.0)),
        "ja": ((-9.9, -2.0), (0.0, 0.0)), "ko": ((4.4, 13.6), (0.0, 0.0)),
    },
    "anchor": {
        "es": ((-0.8, 7.6), (0.7, 8.5)), "zh": ((-5.6, 1.5), (0.0, 1.1)),
        "ja": ((-4.2, 2.0), (0.0, 1.7)), "ko": ((-1.3, 5.8), (0.3, 2.3)),
    },
}

if MATCHER_POLICY == "designated":
    # The paper prints only the designated-form side of this sensitivity (App. A.3).
    TABLE1["vT"]["es"] = (None, 14.2)
    TABLE1["neuron"]["es"] = (None, -0.3)
    TABLE1["anchor"]["es"] = (None, 6.1)
    PAIRED_INTERVALS["vT"]["es"] = (None, (8.9, 19.3))
    PAIRED_INTERVALS["neuron"]["es"] = (None, (-3.6, 3.2))
    PAIRED_INTERVALS["anchor"]["es"] = (None, (2.5, 10.1))


def roster_family(family, lang):
    """Equal-weight mean of within-model paired rates, directly from item JSONs."""
    cells = []
    for model in sorted(ROSTER):
        d = load("interventions/" + FAMILIES[family].format(model=model))
        chains, floors = d["eval_chains"], d["floors"]
        steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
        rows = []
        for i, chain in enumerate(chains):
            if (chain not in CHAINS or chain not in floors or i >= len(steer)
                    or i >= len(rand) or not scoreable(chain, lang)):
                continue
            rows.append((
                content(steer[i].get("txt", ""), chain),
                content(rand[i].get("txt", ""), chain),
                ilr(steer[i].get("txt", ""), chain, lang),
                ilr(rand[i].get("txt", ""), chain, lang),
            ))
        cells.append(tuple(sum(int(row[j]) for row in rows) / len(rows)
                           for j in range(4)))
    return (
        100 * sum(x[0] for x in cells) / len(cells),
        100 * sum(x[0] - x[1] for x in cells) / len(cells),
        100 * sum(x[2] for x in cells) / len(cells),
        100 * sum(x[2] - x[3] for x in cells) / len(cells),
    )


def family_cell(family, model, lang):
    """Per-checkpoint intervention/random content and ILR (designated-form) rates."""
    d = load("interventions/" + FAMILIES[family].format(model=model))
    chains, floors = d["eval_chains"], d["floors"]
    steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
    rows = []
    for i, chain in enumerate(chains):
        if (chain not in CHAINS or chain not in floors or i >= len(steer)
                or i >= len(rand) or not scoreable(chain, lang)):
            continue
        rows.append((
            content(steer[i].get("txt", ""), chain),
            content(rand[i].get("txt", ""), chain),
            ilr(steer[i].get("txt", ""), chain, lang),
            ilr(rand[i].get("txt", ""), chain, lang),
        ))
    return tuple(100 * sum(int(row[j]) for row in rows) / len(rows)
                 for j in range(4))


def family_rows(family, model, lang):
    d = load("interventions/" + FAMILIES[family].format(model=model))
    chains, floors = d["eval_chains"], d["floors"]
    steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
    rows = []
    for i, chain in enumerate(chains):
        if (chain not in CHAINS or chain not in floors or i >= len(steer)
                or i >= len(rand) or not scoreable(chain, lang)):
            continue
        rows.append({
            "cluster": CHAINS[chain]["answer_label_by_lang"].get("en") or chain,
            "sc": content(steer[i].get("txt", ""), chain),
            "rc": content(rand[i].get("txt", ""), chain),
            "si": ilr(steer[i].get("txt", ""), chain, lang),
            "ri": ilr(rand[i].get("txt", ""), chain, lang),
        })
    return rows


def roster_ci(family, lang, left, right, B=4000):
    grouped = []
    for model in sorted(ROSTER):
        groups = {}
        for row in family_rows(family, model, lang):
            groups.setdefault(row["cluster"], []).append((row[left], row[right]))
        grouped.append((list(groups), groups))
    rng, draws = random.Random(0), []
    for _ in range(B):
        effects = []
        for keys, groups in grouped:
            sample = [
                pair
                for _ in keys
                for pair in groups[keys[rng.randrange(len(keys))]]
            ]
            effects.append(100 * sum(int(a) - int(b) for a, b in sample) / len(sample))
        draws.append(sum(effects) / len(effects))
    draws.sort()
    return round(draws[int(.025 * B)], 1), round(draws[int(.975 * B)], 1)


# ``paired_effects.py`` reuses the scoring and bootstrap helpers above without
# executing the checks below.  It sets AOL_SCORING_ONLY, executes this module and
# swallows the resulting SystemExit.
if os.environ.get("AOL_SCORING_ONLY"):
    raise SystemExit(0)

for family, langs in TABLE1.items():
    for lang, want in langs.items():
        _rates = roster_family(family, lang)
        got = (round(_rates[1], 1), round(_rates[3], 1))
        _keep = [i for i, w in enumerate(want) if w is not None]
        check(f"Table 1 {family}/{lang} (Delta_C, Delta_F; item-level recomputation)",
              tuple(got[i] for i in _keep), tuple(want[i] for i in _keep))
        want_ci = PAIRED_INTERVALS[family][lang]
        got_ci = tuple(
            roster_ci(family, lang, a, b) if w is not None else None
            for (a, b), w in zip((("sc", "rc"), ("si", "ri")), want_ci)
        )
        check(f"paired intervals {family}/{lang}",
              tuple(x for x in got_ci if x is not None), tuple(w for w in want_ci if w is not None))

# Values printed in the main text or Appendix configuration/per-cell tables.
near("Appendix A.9 Gemma neuron/ja random content",
     family_cell("neuron", "gemma4_31b", "ja")[1], 39.7)
check("Appendix A.9 Korean vT/neuron designated-form floor",
      all(family_cell(family, model, "ko")[2:] == (0.0, 0.0)
          for family in ("vT", "neuron") for model in ROSTER), True)
check("Appendix selected Korean vT alpha values",
      [
          load("interventions/surfacing_vector_" + model + ".json")["best"]["ko"]["alpha"]
          for model in ("llama31_70_it", "gemma4_31b_it", "qwen36_27b")
      ],
      [1.0, 1.0, 1.0])
check("Appendix A.9 Gemma vT designated form (zh I/C; ja; ko)",
      (
          round(family_cell("vT", "gemma4_31b", "zh")[2], 1),
          round(family_cell("vT", "gemma4_31b", "zh")[3], 1),
          round(family_cell("vT", "gemma4_31b", "ja")[2], 1),
          round(family_cell("vT", "gemma4_31b", "ko")[2], 1),
      ),
      (39.4, 0.0, 0.0, 0.0))

# ---------------------------------------------------------- main-text general SCH
SCH_MODELS = ["qwen36_27b", "llama31_70", "llama31_70_it", "gemma4_31b"]
SCH_ALL = SCH_MODELS + ["gemma4_31b_it"]


def sch_rates(model, lang, pool):
    d = load(f"interventions/sch_amplification_{model}_{pool}.json")
    rows = []
    for item in d["by_lang"][lang]["items"]:
        chain = item["chain"]
        if chain not in CHAINS or not scoreable(chain, lang):
            continue
        amp, rand = item["amp_a1.0"], item["rand_a1.0"]
        rows.append((
            content(amp.get("txt", ""), chain),
            ilr(amp.get("txt", ""), chain, lang),
            ilr(rand.get("txt", ""), chain, lang),
        ))
    return tuple(sum(row[i] for row in rows) / len(rows) for i in range(3)) if rows else ()


def sch_rows(model, lang, pool, permissive=False):
    d = load(f"interventions/sch_amplification_{model}_{pool}.json")
    rows = []
    for item in d["by_lang"][lang]["items"]:
        chain = item["chain"]
        if chain not in CHAINS or not scoreable(chain, lang):
            continue
        amp, rand = item["amp_a1.0"], item["rand_a1.0"]
        rows.append({
            "cluster": CHAINS[chain]["answer_label_by_lang"].get("en") or chain,
            "amp_content": content(amp.get("txt", ""), chain, permissive=permissive),
            "rnd_content": content(rand.get("txt", ""), chain, permissive=permissive),
            "amp_ilr": ilr(amp.get("txt", ""), chain, lang, permissive=permissive),
            "rnd_ilr": ilr(rand.get("txt", ""), chain, lang, permissive=permissive),
        })
    return rows


def sch_macro_ci(model_rows, left, right, seed, B=4000):
    grouped = []
    for rows in model_rows:
        groups = {}
        for row in rows:
            groups.setdefault(row["cluster"], []).append((row[left], row[right]))
        grouped.append((list(groups), groups))
    rng = random.Random(seed)
    draws = []
    for _ in range(B):
        effects = []
        for keys, groups in grouped:
            sample = [
                pair
                for _ in keys
                for pair in groups[keys[rng.randrange(len(keys))]]
            ]
            effects.append(
                100 * sum(int(a) - int(b) for a, b in sample) / len(sample)
            )
        draws.append(sum(effects) / len(effects))
    draws.sort()
    return round(draws[int(.025 * B)], 1), round(draws[int(.975 * B)], 1)


def specific_gate(model, lang):
    d = load(f"interventions/sch_pool_{model}_specific_gate.json")
    return bool(d["by_lang"][lang]["positive_control_supported"])


if MATCHER_POLICY == "attribution_strict":
    strict_summary = {"spanish_prompt_families": {}}
    family_names = {"vT": "vT", "neuron": "language_neuron", "anchor": "MRRE"}
    for family, report_name in family_names.items():
        rate = roster_family(family, "es")
        eligible_by_model = {
            model: len(family_rows(family, model, "es")) for model in sorted(ROSTER)
        }
        original_by_model = {}
        for model in sorted(ROSTER):
            d = load("interventions/" + FAMILIES[family].format(model=model))
            original_by_model[model] = sum(
                chain in CHAINS and chain in d["floors"]
                and i < len(d["eval"]["es"]["steer_items"])
                and i < len(d["eval"]["es"]["rand_items"])
                for i, chain in enumerate(d["eval_chains"])
            )
        strict_summary["spanish_prompt_families"][report_name] = {
            "eligible_n_by_model": eligible_by_model,
            "excluded_n_by_model": {
                model: original_by_model[model] - eligible_by_model[model]
                for model in sorted(ROSTER)
            },
            "content_rate": round(rate[0], 1),
            "content_delta": round(rate[1], 1),
            "answer_form_rate": round(rate[2], 1),
            "answer_form_delta": round(rate[3], 1),
            "answer_form_ci95": list(roster_ci(family, "es", "si", "ri")),
            "per_cell_rates": {
                model: [round(x, 1) for x in family_cell(family, model, "es")]
                for model in sorted(ROSTER)
            },
        }
    sch_cells = [sch_rates(model, "es", "general") for model in SCH_MODELS]
    sch_rows_es = [sch_rows(model, "es", "general") for model in SCH_MODELS]
    strict_summary["sch_general_es"] = {
        "eligible_n_by_model": {
            model: len(sch_rows(model, "es", "general")) for model in SCH_MODELS
        },
        "excluded_n_by_model": {
            model: (
                len(load(f"interventions/sch_amplification_{model}_general.json")["by_lang"]["es"]["items"])
                - len(sch_rows(model, "es", "general"))
            )
            for model in SCH_MODELS
        },
        "content_rate": round(100 * sum(x[0] for x in sch_cells) / len(sch_cells), 1),
        "content_delta": round(
            sum(
                100 * sum(int(row["amp_content"]) - int(row["rnd_content"]) for row in rows)
                / len(rows)
                for rows in sch_rows_es
            ) / len(sch_rows_es),
            1,
        ),
        "content_ci95": list(sch_macro_ci(
            sch_rows_es, "amp_content", "rnd_content", 500
        )),
        "answer_form_rate": round(100 * sum(x[1] for x in sch_cells) / len(sch_cells), 1),
        "answer_form_delta": round(100 * sum(x[1] - x[2] for x in sch_cells) / len(sch_cells), 1),
        "answer_form_ci95": list(sch_macro_ci(
            sch_rows_es, "amp_ilr", "rnd_ilr", 600
        )),
        "per_cell_rates": {
            model: [
                round(100 * sum(int(row[key]) for row in rows) / len(rows), 1)
                for key in ("amp_content", "rnd_content", "amp_ilr", "rnd_ilr")
            ]
            for model, rows in zip(SCH_MODELS, sch_rows_es)
        },
        "per_cell_ci95": {
            model: {
                "content": (
                    list(sch_macro_ci([rows], "amp_content", "rnd_content", 500))
                    if any(row["amp_content"] != row["rnd_content"] for row in rows)
                    else None
                ),
                "answer_form": (
                    list(sch_macro_ci([rows], "amp_ilr", "rnd_ilr", 600))
                    if any(row["amp_ilr"] != row["rnd_ilr"] for row in rows)
                    else None
                ),
            }
            for model, rows in zip(SCH_MODELS, sch_rows_es)
        },
    }
    strict_summary["sch_specific_es"] = {}
    for model in SCH_ALL:
        if not specific_gate(model, "es"):
            strict_summary["sch_specific_es"][model] = {"gate": "fail"}
            continue
        rows = sch_rows(model, "es", "specific")
        if not rows:
            strict_summary["sch_specific_es"][model] = {"gate": "pass", "eligible_n": 0}
            continue
        d = load(f"interventions/sch_amplification_{model}_specific.json")
        original_n = len(d["by_lang"]["es"]["items"])
        strict_summary["sch_specific_es"][model] = {
            "gate": "pass",
            "eligible_n": len(rows),
            "excluded_n": original_n - len(rows),
            "content_delta": round(
                100 * sum(int(x["amp_content"]) - int(x["rnd_content"]) for x in rows)
                / len(rows), 1
            ),
            "answer_form_delta": round(
                100 * sum(int(x["amp_ilr"]) - int(x["rnd_ilr"]) for x in rows)
                / len(rows), 1
            ),
        }
    saved_strict = load("attribution_strict_summary.json")
    expected_strict = {
        "spanish_prompt_families": saved_strict["spanish_prompt_families"],
        "sch_general_es": saved_strict["sch_general_es"],
    }
    def contains_expected(got, want):
        if isinstance(want, dict):
            return isinstance(got, dict) and all(
                key in got and contains_expected(got[key], value)
                for key, value in want.items()
            )
        return got == want

    if contains_expected(strict_summary, expected_strict):
        PASS += 1
        print("[PASS] attribution-strict saved summary")
    else:
        FAIL.append("attribution-strict summary differs from attribution_strict_summary.json")
        print("[FAIL] attribution-strict saved summary")
    print("\nATTRIBUTION_STRICT_SUMMARY=" + json.dumps(strict_summary, sort_keys=True))


# ---------------------------------------------------------------- diagnostics
# Each block recomputes a reported diagnostic from the frozen JSONs and compares
# it with the value printed in the paper.

import importlib.util as _ilu


def _load_module(name):
    spec = _ilu.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- Retained-population coverage (App. A.2, Sec. 5.4) -----------------------
# Spanish coverage is defined by the attribution-strict exclusion, so the
# Spanish rows are checked only under that policy; Japanese is policy-invariant.
COVERAGE_FAMILIES = {"vT": ("surfacing_vector", 441), "neuron": ("language_neurons", 541)}
_cov_langs = ("ja", "es") if MATCHER_POLICY == "attribution_strict" else ("ja",)
for _fam, (_prefix, _common) in COVERAGE_FAMILIES.items():
    for _lang, _want in [(l, {("vT", "ja"): 99.3, ("vT", "es"): 78.7,
                              ("neuron", "ja"): 97.0, ("neuron", "es"): 82.8}[(_fam, l)])
                         for l in _cov_langs]:
        _kept = 0
        for _model in ROSTER:
            _d = load("interventions/" + FAMILIES[_fam].format(model=_model))
            _kept += sum(1 for _c in _d["eval_chains"]
                         if _c in CHAINS and _c in _d["floors"] and scoreable(_c, _lang))
        near(f"Section 5.4 retained-population coverage {_fam}/{_lang}",
             100 * _kept / _common, _want, tol=0.1)

# --- Hop-level solvability (App. A.7) --------------------------------------
_solv = _load_module("hop_solvability").rows()
_llama = [v[1] for (lbl, _), v in _solv.items() if lbl == "Llama-3.1-70B"]
_gemit = [v[1] for (lbl, _), v in _solv.items() if lbl == "Gemma-4-31B-It"]
near("Sec. 4 / Table 6 solvability Llama-3.1-70B min", min(_llama), 97.8, tol=0.1)
near("Sec. 4 / Table 6 solvability Llama-3.1-70B max", max(_llama), 99.7, tol=0.1)
near("Sec. 4 / Table 6 solvability Gemma-4-31B-It min", min(_gemit), 35.2, tol=0.1)
near("Sec. 4 / Table 6 solvability Gemma-4-31B-It max", max(_gemit), 63.3, tol=0.1)

# Retained-population English solvability (App. A.7): share of each
# checkpoint's retained items answered correctly under the English two-hop prompt,
# range over the twelve family--language cells.
_es = load("english_solvable/english_solvable_and_item_weighted.json")
_esr = {}
for _c in _es["cells"]:
    for _model, _n, _k in zip(sorted(ROSTER), _c["n_full"], _c["n_en_solvable"]):
        _esr.setdefault(_model, []).append(100 * _k / _n)
for _model, (_lo, _hi) in {"llama31_70": (94.4, 99.2), "llama31_70_it": (71.4, 92.6),
                           "gemma4_31b": (75.8, 82.0), "qwen36_27b": (56.5, 79.8),
                           "gemma4_31b_it": (30.7, 60.0)}.items():
    near(f"Appendix A.7 retained English solvability {_model} min", min(_esr[_model]), _lo, tol=0.06)
    near(f"Appendix A.7 retained English solvability {_model} max", max(_esr[_model]), _hi, tol=0.06)

# --- Degenerate scored lines (App. C.1) ------------------------------------
_deg = _load_module("degenerate_outputs").collect()
_pct = lambda k: 100 * _deg[k]["d"] / _deg[k]["n"]
near("Appendix C.1 vT Gemma zh control degeneracy",
     _pct(("$v_T$", "gemma4_31b", "zh", "C")), 27.5, tol=0.1)
near("Appendix C.1 vT Gemma ko intervention degeneracy",
     _pct(("$v_T$", "gemma4_31b", "ko", "I")), 31.2, tol=0.1)
near("Appendix C.1 vT Gemma ja intervention degeneracy",
     _pct(("$v_T$", "gemma4_31b", "ja", "I")), 55.0, tol=0.1)
check("Appendix C.1 MRRE Gemma-It control fully degenerate",
      all(round(_pct(("MRRE", "gemma4_31b_it", _l, "C"))) == 100 for _l in ("es", "zh", "ja", "ko")),
      True)
check("Appendix D SCH Gemma ko degeneracy, 19% of both arms (I, C)",
      (round(_pct(("SCH", "gemma4_31b", "ko", "I"))), round(_pct(("SCH", "gemma4_31b", "ko", "C")))), (19, 19))

# --- Attempt-rate decomposition (App. C.1) ---------------------------------
_ad = _load_module("attempt_decomposition")
for _fam, _lang, _uncond, _cond in (("neuron", "zh", 7.3, 1.8), ("neuron", "ko", 8.8, 11.2)):
    _m = _ad.macro(globals(), _fam, _lang)
    near(f"Appendix C.1 {_fam}/{_lang} content contrast, all items", _m["cU"], _uncond, tol=0.1)
    near(f"Appendix C.1 {_fam}/{_lang} content contrast, attempts only", _m["cK"], _cond, tol=0.1)

# --- Calibration/evaluation answer overlap (App. A.4) ----------------------
_en = lambda c: (CHAINS[c].get("answer_label_by_lang") or {}).get("en")
_item_overlap = 0
_eval_items = _shared = 0
for _fam in FAMILIES:
    for _model in ROSTER:
        _d = load("interventions/" + FAMILIES[_fam].format(model=_model))
        _cc, _ec = set(_d.get("calib_chains", [])), _d.get("eval_chains", [])
        if not _cc:
            continue
        _item_overlap += len(_cc & set(_ec))
        _ca = {_en(_c) for _c in _cc if _c in CHAINS} - {None}
        for _c in _ec:
            if _c in CHAINS:
                _eval_items += 1
                _shared += _en(_c) in _ca
check("Appendix A.4 calibration/evaluation item overlap", _item_overlap, 0)
near("Appendix A.4 evaluation items sharing a calibration answer",
     100 * _shared / _eval_items, 35.7, tol=0.1)
check("Appendix A.4 distinct English answers over frozen chains",
      len({_en(_c) for _c in CHAINS} - {None}), 126)

# --- Selection-aware pass and cluster-disjoint restriction (App. A.6, A.8)-----
# Computed by paired_effects.py, which reuses the scoring helpers above.  The
# checks below pin the values quoted in the body and appendix.
_sep = _load_module("paired_effects")

# Sign-flip p, Bonferroni percentile interval, and whether that interval
# excludes zero.  The two criteria agree except on neuron/ja, whose interval
# excludes zero at -0.1 while its randomization p exceeds 0.05/12.
SELECTION_AWARE = {
    ("neuron", "ko"): (0.0001, [2.6, 15.7], True),
    ("neuron", "zh"): (0.0001, [2.5, 12.6], True),
    ("neuron", "ja"): (0.0120, [-11.7, -0.1], True),
    ("vT", "ko"): (0.0072, [-0.1, 18.3], False),
}
for (_fam, _lang), (_p, _ci, _survives) in SELECTION_AWARE.items():
    _i = ("vT", "neuron", "anchor").index(_fam) * 4 + ("es", "zh", "ja", "ko").index(_lang)
    _m = _sep.multiplicity(_fam, _lang, 3000 + _i)
    check(f"Section 5.1 selection-aware {_fam}/{_lang}",
          (_m["p_signflip"], _m["bonferroni_ci"], _m["survives"]),
          (_p, _ci, _survives))
check("Section 5.1 corrected threshold",
      round(0.05 / _sep.MULT_FAMILY_SIZE, 4), 0.0042)

CLUSTER_DISJOINT = {
    ("neuron", "ko"): (10.6, [5.5, 16.2]),
    ("neuron", "zh"): (7.6, [3.4, 12.5]),
    ("vT", "ko"): (5.7, [-0.8, 12.2]),
}
for (_fam, _lang), (_d, _ci) in CLUSTER_DISJOINT.items():
    _i = ("vT", "neuron", "anchor").index(_fam) * 4 + ("es", "zh", "ja", "ko").index(_lang)
    _cd = _sep.cluster_disjoint_delta(_fam, _lang, 2100 + _i)
    check(f"Section 5.1 cluster-disjoint {_fam}/{_lang}",
          (_cd["delta"], _cd["delta_ci95"]), (_d, _ci))

_retained_es = []
for _fam in ("vT", "neuron", "anchor"):
    _keep = sum(len(_sep.spanish_rows(_fam, _m, "es", cluster_disjoint=True)) for _m in ROSTER)
    _tot = sum(len(_sep.spanish_rows(_fam, _m, "es")) for _m in ROSTER)
    _retained_es.append(round(100 * _keep / _tot))
check("Appendix cluster-disjoint retention is 58% for Spanish", _retained_es, [58, 58, 58])
_retained_cjk = []
for _fam in ("vT", "neuron", "anchor"):
    for _lang in ("zh", "ja", "ko"):
        _keep = sum(len(_sep.spanish_rows(_fam, _m, _lang, cluster_disjoint=True)) for _m in ROSTER)
        _tot = sum(len(_sep.spanish_rows(_fam, _m, _lang)) for _m in ROSTER)
        _retained_cjk.append(100 * _keep / _tot)
check("Appendix A.8 cluster-disjoint retention 64-66% in zh/ja/ko cells",
      (round(min(_retained_cjk)), round(max(_retained_cjk))), (64, 66))

# Han-only Japanese exclusion per checkpoint (App. A.3).
_hanonly = {}
for _fam in FAMILIES:
    for _model in ROSTER:
        _d = load("interventions/" + FAMILIES[_fam].format(model=_model))
        _items = [c for c in _d["eval_chains"] if c in CHAINS and c in _d["floors"]]
        _hanonly[(_fam, _model)] = (sum(not scoreable(c, "ja") for c in _items), len(_items))
_vt = [100 * k / n for (f, _), (k, n) in _hanonly.items() if f == "vT"]
_na = [100 * k / n for (f, _), (k, n) in _hanonly.items() if f != "vT"]
near("Appendix A.3 Han-only exclusion vT max", max(_vt), 1.9, tol=0.06)
check("Appendix A.3 Han-only exclusion vT Gemma-4-31B", _hanonly[("vT", "gemma4_31b")], (3, 160))
near("Appendix A.3 Han-only exclusion neuron/MRRE min", min(_na), 1.7, tol=0.06)
near("Appendix A.3 Han-only exclusion neuron/MRRE max", max(_na), 6.6, tol=0.06)
# Registered CJK-form matches always earn designated-form credit (App. F.2, token fertility).
_inv = _cred = 0
for _fam in FAMILIES:
    for _model in ROSTER:
        _d = load("interventions/" + FAMILIES[_fam].format(model=_model))
        for _lang in ("zh", "ja", "ko"):
            for _arm in ("steer_items", "rand_items"):
                _items = _d["eval"][_lang][_arm]
                for _i, _c in enumerate(_d["eval_chains"]):
                    if (_c not in CHAINS or _c not in _d["floors"] or _i >= len(_items)
                            or not scoreable(_c, _lang)):
                        continue
                    _t = _items[_i].get("txt", "") or ""
                    _line = _t.strip().splitlines()[0] if _t.strip() else ""
                    if matches(_line, inventory(_c, _lang)):
                        _inv += 1
                        _cred += ilr(_t, _c, _lang)
check("Appendix F registered CJK-form matches and credited", (_inv, _cred), (81, 81))

# Checkpoint-level SCH answer-form composition (App. D, Table schforms-models): (n, target, English, other),
# with n the union of the run's stored entity flag and the locked content matcher (as in the reviewed version).
_sfc = _load_module("sch_form_composition")
_SFC = {
    ('qwen36_27b', 'general', 'ko'): (3, 3, 0, 0),
    ('qwen36_27b', 'general', 'zh'): (7, 7, 0, 0),
    ('qwen36_27b', 'general', 'ja'): (2, 2, 0, 0),
    ('qwen36_27b', 'specific', 'ko'): (3, 3, 0, 0),
    ('llama31_70', 'general', 'ko'): (11, 8, 3, 0),
    ('llama31_70', 'general', 'zh'): (33, 24, 6, 3),
    ('llama31_70', 'general', 'ja'): (27, 26, 0, 1),
    ('llama31_70', 'specific', 'ko'): (23, 3, 11, 9),
    ('llama31_70', 'specific', 'zh'): (26, 5, 14, 7),
    ('llama31_70', 'specific', 'ja'): (24, 17, 3, 4),
    ('llama31_70_it', 'general', 'ko'): (16, 6, 1, 9),
    ('llama31_70_it', 'general', 'zh'): (17, 13, 1, 3),
    ('llama31_70_it', 'general', 'ja'): (6, 5, 0, 1),
    ('llama31_70_it', 'specific', 'ko'): (10, 10, 0, 0),
    ('llama31_70_it', 'specific', 'zh'): (11, 7, 1, 3),
    ('llama31_70_it', 'specific', 'ja'): (14, 14, 0, 0),
    ('gemma4_31b', 'general', 'ko'): (8, 6, 0, 2),
    ('gemma4_31b', 'general', 'zh'): (7, 7, 0, 0),
    ('gemma4_31b', 'general', 'ja'): (4, 4, 0, 0),
    ('gemma4_31b', 'specific', 'ko'): (16, 13, 2, 1),
    ('gemma4_31b', 'specific', 'zh'): (6, 6, 0, 0),
    ('gemma4_31b', 'specific', 'ja'): (7, 7, 0, 0),
}
check("Appendix D SCH answer-form composition by checkpoint and pool",
      {k: _sfc.composition(*k) for k in _SFC}, _SFC)

# Grid-wide validation of the locked matcher, stage 1 (App. B.2).
_gate = {"ko": lambda l: has_hangul(l), "ja": lambda l: has_kana(l),
         "zh": lambda l: has_han(l) and not has_kana(l) and not has_hangul(l)}
_c1f0, _scriptless = {}, 0
for _fam in FAMILIES:
    for _model in ROSTER:
        _d = load("interventions/" + FAMILIES[_fam].format(model=_model))
        for _lang in ("zh", "ja", "ko", "es"):
            for _arm in ("steer_items", "rand_items"):
                _items = _d["eval"][_lang][_arm]
                for _i, _c in enumerate(_d["eval_chains"]):
                    if (_c not in CHAINS or _c not in _d["floors"] or _i >= len(_items)
                            or not scoreable(_c, _lang)):
                        continue
                    _t = _items[_i].get("txt", "") or ""
                    if content(_t, _c) and not ilr(_t, _c, _lang):
                        _c1f0[_lang] = _c1f0.get(_lang, 0) + 1
                        _line = _t.strip().splitlines()[0] if _t.strip() else ""
                        if _lang != "es" and not _gate[_lang](_line):
                            _scriptless += 1
check("Appendix B.2 content-matched outputs without designated-form credit (es; total)",
      (_c1f0["es"], sum(_c1f0.values())), (47, 1096))
check("Appendix B.2 of these, CJK outputs without the designated script",
      _scriptless, 1025)
_gv = load("human_form_audit/grid_validation/summary.json")["strata"]
check("Appendix B.2 stage-1 judgments (zh, ja, ko: n and acceptable forms)",
      tuple((_gv[f"script_uncredited/{l}"]["n"], _gv[f"script_uncredited/{l}"]["human_form"]) for l in ("zh", "ja", "ko")),
      ((21, 0), (2, 0), (1, 0)))
_es = _gv["es_uncredited_content/es"]
check("Appendix B.2 stage-1 Spanish (n, content not correct, acceptable, target-specific)",
      (_es["n"], _es["n"] - _es["content"].get("correct"), _es["human_form"], _es["human_form_target_specific"]), (47, 1, 3, 2))
_sv = {(c["family"], c["lang"]): c for c in load("human_form_audit/grid_validation/sensitivity.json")["cells"]}
check("Appendix B.2 sensitivity, vT/es (locked Delta_F; delta: locked, all)",
      (_sv[("vT", "es")]["locked"]["delta_F"], _sv[("vT", "es")]["locked"]["delta"],
       _sv[("vT", "es")]["all"]["delta"], _sv[("vT", "es")]["all"]["delta_ci95"]),
      (15.6, -2.6, -3.1, [-6.3, -0.5]))
check("Appendix B.2 sensitivity, neuron/es (locked Delta_F; delta: locked, all)",
      (_sv[("neuron", "es")]["locked"]["delta_F"],
       _sv[("neuron", "es")]["locked"]["delta"], _sv[("neuron", "es")]["all"]["delta"], _sv[("neuron", "es")]["all"]["delta_ci95"]),
      (-1.3, 0.6, 0.2, [-1.1, 1.6]))
check("Appendix B.2 sensitivity leaves every Chinese, Japanese, and Korean cell unchanged",
      all(c["locked"] == c["all"] for k, c in _sv.items() if k[1] in ("zh", "ja", "ko")), True)

# Leave-one-checkpoint-out selection-aware status (Sec. 5.3, Table separation-robustness),
# written by leave_one_out_selection.py.
_loo = {(c["family"], c["lang"]): c for c in load("checkpoint_exclusion/leave_one_out_selection.json")["cells"]}
check("Table 8 joint pass in leave-one-out rosters (vT, neuron, MRRE x es/zh/ja/ko)",
      [_loo[(f, l)]["joint_pass_count"] for f in ("vT", "neuron", "anchor") for l in ("es", "zh", "ja", "ko")],
      [0, 0, 0, 2, 0, 3, 2, 4, 0, 0, 0, 0])
_d = lambda f, l, m: (lambda x: (x["delta"], x["bonferroni_pass"], x["signflip_pass"]))(_loo[(f, l)]["drop"][m])
check("Section 5.3 leave-one-out: zh without Gemma-4-31B, zh without Llama-3.1-70B, ko without Gemma-4-31B",
      (_d("neuron", "zh", "gemma4_31b"), _d("neuron", "zh", "llama31_70"), _d("neuron", "ko", "gemma4_31b")),
      ((5.2, True, False), (5.0, False, False), (4.8, False, False)))
check("Section 5.3 leave-one-out: ko loses the joint pass only without Gemma-4-31B",
      sorted(m for m, x in _loo[("neuron", "ko")]["drop"].items() if not x["joint_pass"]), ["gemma4_31b"])
# Excluded-Gemma-4-31B selection-aware values quoted in Section 5.3, written by
# neuron_robustness.py; the Llama-3.1-70B values come from leave_one_out_selection.py.
_rb = {(r["analysis"], r["language"]): r for r in load("checkpoint_exclusion/neuron_robustness.json")["analyses"]}
check("Section 5.3 without Gemma-4-31B: zh, ko (delta, Bonferroni, p_flip)",
      tuple((_rb[("drop_gemma_primary", l)]["separation_delta"], _rb[("drop_gemma_primary", l)]["bonferroni_ci"],
             _rb[("drop_gemma_primary", l)]["p_signflip"]) for l in ("zh", "ko")),
      ((5.2, [0.5, 10.8], 0.0055), (4.8, [-2.2, 12.6], 0.0563)))
_ll = _loo[("neuron", "zh")]["drop"]["llama31_70"]
check("Section 5.3 without Llama-3.1-70B: zh (delta, Bonferroni, p_flip)",
      (_ll["delta"], _ll["bonferroni_ci"], _ll["p_signflip"]), (5.0, [-0.1, 10.4], 0.0119))
check("Section 5.3 both-arms-attempt macros match neuron_robustness.py",
      tuple(_rb[("joint_attempt_all_models_diagnostic", l)]["separation_delta"] for l in ("zh", "ko")), (0.8, 10.7))
check("Section 5.3 leave-one-out: zh loses the joint pass only without Gemma-4-31B or Llama-3.1-70B",
      sorted(m for m, x in _loo[("neuron", "zh")]["drop"].items() if not x["joint_pass"]), ["gemma4_31b", "llama31_70"])

# Script-constrained decode (App. F.2, Table scriptdecode): the masked output is scored against the
# chain's own designated forms and, as a control, against those of the next chain in the item list.
_sd = load("rivals/scriptdecode_llama31_70_it.json")["by_target"]
for _lang, (_n, _pct) in {"ko": (141, 5.0), "ja": (109, 6.4)}.items():
    _its = _sd[_lang]["items"]
    _rows = []
    for _i, _x in enumerate(_its):
        _c, _cd = _x["chain"], _its[(_i + 1) % len(_its)]["chain"]
        if _c in CHAINS and scoreable(_c, _lang):
            _t = _x.get("key_txt", "")
            _rows.append((int(ilr(_t, _c, _lang)), int(_cd in CHAINS and ilr(_t, _cd, _lang)),
                          CHAINS[_c]["answer_label_by_lang"]["en"]))
    check(f"Appendix F script-constrained {_lang}: n, control",
          (len(_rows), sum(r[1] for r in _rows)), (_n, 0))
    near(f"Appendix F script-constrained {_lang} masked rate", 100 * sum(r[0] for r in _rows) / len(_rows), _pct, tol=0.06)
    # Answer-cluster bootstrap; the original resampling seed is not recorded, so the
    # printed bounds are matched within Monte Carlo tolerance.
    _rng = random.Random(0)
    _cl = {}
    for _h, _k, _a in _rows:
        _cl.setdefault(_a, []).append(_h - _k)
    _keys, _est = list(_cl), []
    for _ in range(4000):
        _s = [v for _k in (_keys[_rng.randrange(len(_keys))] for _ in _keys) for v in _cl[_k]]
        _est.append(100 * sum(_s) / len(_s))
    _est.sort()
    _want = {"ko": (1.8, 9.2), "ja": (0.9, 14.3)}[_lang]
    near(f"Appendix F script-constrained {_lang} interval low", _est[100], _want[0], tol=0.15)
    near(f"Appendix F script-constrained {_lang} interval high", _est[3899], _want[1], tol=0.15)
check("Appendix F script-constrained ja under the run's own flag: items, recoveries",
      (len(_sd["ja"]["items"]), round(_sd["ja"]["rate"]["key_correct_T"] * len(_sd["ja"]["items"]))), (112, 14))
# The stored lower bound is exactly 6.25 and is printed half-up as +6.3.
near("Appendix F script-constrained ja own-flag interval low", 100 * _sd["ja"]["ci_key_correct_minus_ctrl"][0], 6.3, tol=0.06)
near("Appendix F script-constrained ja own-flag interval high", 100 * _sd["ja"]["ci_key_correct_minus_ctrl"][1], 19.6, tol=0.06)

# English-axis ablation (App. F.2, Table hubablation).
_hub = load("rivals/hubablation_llama31_70_it_ko_es.json")["by_layer"]
_p1 = lambda v: round(100 * v, 1) + 0.0
check("Appendix F English-axis ablation (headroom, vs random, vs source)",
      {L: (_hub[L]["n"], _p1(_hub[L]["headroom"]),
           _p1(_hub[L]["EnOut_minus_RandOut"]), [_p1(x) for x in _hub[L]["ci_EnOut_RandOut"]],
           _p1(_hub[L]["EnOut_minus_LangOut"]), [_p1(x) for x in _hub[L]["ci_EnOut_LangOut"]])
       for L in ("40", "52", "64")},
      {"40": (80, 31.2, -0.8, [-8.1, 6.7], 5.0, [-3.8, 13.8]),
       "52": (80, 33.8, 0.8, [-2.2, 4.1], 6.2, [0.0, 13.8]),
       "64": (80, 42.5, -4.7, [-10.2, 0.0], -1.2, [-7.5, 3.8])})

# Japanese single-hop TDV implementation check (App. E), from the released recomputation.
_tj = load("tdv_japanese_check/results/recomputed/analysis.json")["subsets"]["implementation_failures"]
_tjn = _tj["n"]
check("Appendix E Japanese 70B check: items, TDV and R0 recoveries",
      (_tjn, round(_tj["rates"]["TDV"]["percent_or_pp"] * _tjn / 100), round(_tj["rates"]["R0"]["percent_or_pp"] * _tjn / 100)),
      (102, 16, 6))
check("Appendix E Japanese 70B check: TDV - R0",
      (round(_tj["contrasts"]["R0"]["percent_or_pp"], 1), tuple(round(x, 1) for x in _tj["contrasts"]["R0"]["ci95"])), (9.8, (-5.6, 26.5)))
_tjs = load("tdv_japanese_check/results/recomputed/analysis.json")["subsets"]
_c1 = lambda c: (round(c["percent_or_pp"], 1), tuple(round(x, 1) for x in c["ci95"]))
check("Appendix E Japanese 70B check: screened items, five-random mean, cluster-disjoint",
      (_tjs["all_screen"]["n"], _c1(_tj["contrasts"]["R_mean5"]),
       _tjs["implementation_failures_cluster_disjoint"]["n"],
       _c1(_tjs["implementation_failures_cluster_disjoint"]["contrasts"]["R0"])),
      (200, (10.2, (-2.2, 25.0)), 50, (2.0, (-20.7, 25.4))))
check("Appendix A.3 Han-only items per checkpoint, neuron/MRRE",
      (min(k for (f, _), (k, n) in _hanonly.items() if f != "vT"),
       max(k for (f, _), (k, n) in _hanonly.items() if f != "vT")), (2, 4))

# Pairs in which both arms attempt (Sec. 5.3, App. C.1).
_baa = {r["lang"]: r for r in load("both_arms_attempt/results.json")["results"]}
_r1 = lambda x: round(x, 1)
for _lang, (_nb, _d, _ci, _gnb, _gd, _wg, _wgci) in {
        "zh": (438, 0.8, [-6.7, 7.9], 18, -16.7, 5.2, [1.8, 9.1]),
        "ko": (417, 10.7, [1.3, 23.2], 3, 33.3, 5.1, [0.2, 10.4])}.items():
    _r = _baa[_lang]
    _g = next(c for c in _r["checkpoints"] if c["model"] == "gemma4_31b")
    check(f"Appendix C.1 both-attempt {_lang}",
          (_r["n_both"], _r1(_r["both"]["delta"]["estimate"]), [_r1(x) for x in _r["both"]["delta"]["ci95"]],
           _g["n_both"], _r1(_g["both"]["delta"]["estimate"]),
           _r1(_r["without_gemma"]["both"]["delta"]["estimate"]),
           [_r1(x) for x in _r["without_gemma"]["both"]["delta"]["ci95"]]),
          (_nb, _d, _ci, _gnb, _gd, _wg, _wgci))

# delta column of Table 1 (tab:paired-roster) and the SCH cells of tab:sch-boundary.
# paired_effects.py writes four appendix tables but not these, so the delta estimate
# and its paired interval are asserted here for all sixteen cells, using the seed
# offsets collect() assigns in order.
TABLE1_DELTA = {
    ("vT", "es"): (-2.6, [-5.9, 0.0]), ("vT", "zh"): (2.9, [-2.0, 7.8]),
    ("vT", "ja"): (2.4, [-2.7, 7.2]), ("vT", "ko"): (8.7, [2.4, 15.3]),
    ("neuron", "es"): (0.6, [-0.8, 2.2]), ("neuron", "zh"): (7.3, [4.0, 11.1]),
    ("neuron", "ja"): (-5.9, [-10.0, -1.9]), ("neuron", "ko"): (8.8, [4.5, 13.5]),
    ("anchor", "es"): (-1.2, [-2.9, 0.4]), ("anchor", "zh"): (-2.5, [-6.0, 0.9]),
    ("anchor", "ja"): (-1.7, [-4.7, 1.2]), ("anchor", "ko"): (1.0, [-2.3, 4.7]),
    ("sch", "es"): (0.3, [-0.8, 1.6]), ("sch", "zh"): (1.8, [0.5, 3.3]),
    ("sch", "ja"): (-0.7, [-2.0, 0.0]), ("sch", "ko"): (0.4, [-1.4, 2.2]),
}
for _offset, ((_fam, _lang), (_delta, _ci)) in enumerate(TABLE1_DELTA.items()):
    _cell = _sep.analyze(_fam, _lang, _offset)
    check(f"Table 1 separation {_fam}/{_lang}",
          (_cell["delta"], _cell["delta_ci95"]), (_delta, _ci))

# v_T seed sensitivity (App. A.4). No generator table: content and DFR effects over
# the five preserved matched-random draws, as the seed-7 estimate and the range
# across all five.
VT_SEEDS = {
    "es": (13.0, (6.0, 16.8), 15.6, (10.4, 18.1)),
    "zh": (11.3, (10.4, 13.2), 8.4, (8.4, 8.4)),
    "ja": (2.5, (1.3, 7.5), 0.2, (0.2, 0.2)),
    "ko": (8.7, (8.0, 13.5), 0.0, (0.0, 0.0)),
}
for _lang, _want in VT_SEEDS.items():
    _by_seed = [_sep.point(_sep.cell_rows("vT", _lang, seed=s))
                for s in (7, 17, 27, 37, 47)]
    _content = [round(p[0], 1) for p in _by_seed]
    _form = [round(p[1], 1) for p in _by_seed]
    check(f"Appendix vT seed sensitivity {_lang}",
          (_content[0], (min(_content), max(_content)),
           _form[0], (min(_form), max(_form))),
          _want)


# ------------------------------------------------ reported cells not covered above
# The tables generated by paired_effects.py, hop_solvability.py, degenerate_outputs.py,
# and attempt_decomposition.py are reproduced cell by cell by those scripts. The
# blocks below close the remaining gap: every value printed in a table that has no
# generator is asserted here, so no reported number rests on a hand transcription.

# Appendix tab:sch-gates --- the held-out mean-ablation gate behind "four of five
# general pools pass". The verdict decides which checkpoints enter the SCH macros.
for _model, _want in (
        ("qwen36_27b", (6, 0.289, -0.050, 0.340, "pass")),
        ("llama31_70", (62, 0.464, 0.042, 0.423, "pass")),
        ("llama31_70_it", (78, 0.702, 0.024, 0.678, "pass")),
        ("gemma4_31b", (18, 0.723, 0.030, 0.693, "pass")),
        ("gemma4_31b_it", (36, -2.235, 0.867, -3.102, "fail"))):
    _pool = load(f"interventions/sch_pool_{_model}_general.json")
    _val = _pool["validation"]
    check(f"Appendix SCH gate {_model}",
          (_pool["selected_count"],
           round(_val["mean_ablation_dnll"], 3),
           round(_val["layer_matched_random_dnll"], 3),
           round(_val["selected_minus_random_dnll"], 3),
           "pass" if _pool["pool_status"] == "positive_control_supported" else "fail"),
          _want)
check("Appendix four of five general pools pass the gate",
      sum(load(f"interventions/sch_pool_{m}_general.json")["pool_status"]
          == "positive_control_supported" for m in ROSTER), 4)

# Appendix tab:selected-configs --- the frozen configuration of every family--cell.
# Rendered as the table prints them: injection layers/alpha, amp/suppression, and
# MRRE mode/L2/a2 with A for anchor-only and F for full.
def _alpha(value):
    return ("%g" % value).lstrip("0") if value != int(value) else str(int(value))


SELECTED_CONFIGS = {
    "llama31_70": {"es": ("75--77/.5", "4/1", "A/72/.75"), "zh": ("76/1", "4/1", "F/72/.75"),
                   "ja": ("76/1", "9/.1", "F/65/.75"), "ko": ("67--69/.5", "4/.1", "F/72/.75")},
    "llama31_70_it": {"es": ("71--73/1", "4/1", "A/65/.75"), "zh": ("77/1", "9/.1", "F/72/.75"),
                      "ja": ("77--78/.5", "4/.1", "F/72/.75"), "ko": ("75--77/1", "9/.1", "A/65/.75")},
    "gemma4_31b": {"es": ("58/.25", "9/1", "A/54/.75"), "zh": ("57--58/1", "4/1", "A/49/.75"),
                   "ja": ("58/.5", "4/.1", "A/54/.75"), "ko": ("57--58/.25", "4/1", "A/49/.75")},
    "gemma4_31b_it": {"es": ("57--58/.5", "4/1", "A/54/.75"), "zh": ("58/1", "9/.1", "A/54/.75"),
                      "ja": ("57/1", "9/1", "A/49/.75"), "ko": ("58/1", "9/.1", "A/49/.75")},
    "qwen36_27b": {"es": ("62/.5", "4/1", "F/57/.75"), "zh": ("61--62/1", "4/.1", "A/57/.75"),
                   "ja": ("60/1", "9/1", "F/57/.75"), "ko": ("62/1", "4/1", "A/52/.75")},
}
for _model, _cells in SELECTED_CONFIGS.items():
    _vt = load(f"interventions/surfacing_vector_{_model}.json")["best"]
    _neuron = load(f"interventions/language_neurons_{_model}.json")["best"]
    _mrre = load(f"interventions/mrre_{_model}.json")["best"]
    for _lang, _want in _cells.items():
        _layers = _vt[_lang]["layers"]
        _span = f"{_layers[0]}--{_layers[-1]}" if len(_layers) > 1 else str(_layers[0])
        check(f"Appendix selected config {_model}/{_lang}",
              (f"{_span}/{_alpha(_vt[_lang]['alpha'])}",
               f"{int(_neuron[_lang]['amp'])}/{_alpha(_neuron[_lang]['supp'])}",
               f"{'A' if _mrre[_lang]['mode'] == 'anchor' else 'F'}"
               f"/{_mrre[_lang]['L2']}/{_alpha(_mrre[_lang]['a2'])}"),
              _want)
check("Appendix all reported MRRE cells select a2 = 0.75",
      sorted({load(f"interventions/mrre_{m}.json")["best"][lang]["a2"]
              for m in SELECTED_CONFIGS for lang in ("es", "zh", "ja", "ko")}), [0.75])

# ---------------------------------------------------------------- manuscript
# The numbers printed in the paper are stored in PAPER below and compared with their recomputation from
# the per-analysis result folders next to this file. Values are rounded to the precision printed in the manuscript;
# intervals are (lower, upper) pairs.
_PAPER_SEEN = []


def paper(label, got):
    _PAPER_SEEN.append((label, got))
    check(label, got, PAPER.get(label, "<not in PAPER>"))


def _ms_load(*parts):
    with open(os.path.join(RESULTS, *parts)) as fh:
        return json.load(fh)


def r1(v):
    return round(v, 1) + 0.0


def r2(v):
    return round(v, 2) + 0.0


def ci1(ci):
    return (r1(ci[0]), r1(ci[1]))


def ci2(ci):
    return (r2(ci[0]), r2(ci[1]))


PAPER = {
    'Table 1 vT/es': (13.0, 15.6, -2.6, (-5.9, 0.0), False),
    'Table 1 vT/zh': (11.3, 8.4, 2.9, (-2.0, 7.8), False),
    'Table 1 vT/ja': (2.5, 0.2, 2.4, (-2.7, 7.2), False),
    'Table 1 vT/ko': (8.7, 0.0, 8.7, (2.4, 15.3), False),
    'Table 1 neuron/es': (-0.7, -1.3, 0.6, (-0.8, 2.2), False),
    'Table 1 neuron/zh': (7.3, 0.0, 7.3, (4.0, 11.1), True),
    'Table 1 neuron/ja': (-5.9, 0.0, -5.9, (-10.0, -1.9), False),
    'Table 1 neuron/ko': (8.8, 0.0, 8.8, (4.5, 13.5), True),
    'Table 1 anchor/es': (3.3, 4.5, -1.2, (-2.9, 0.4), False),
    'Table 1 anchor/zh': (-2.0, 0.5, -2.5, (-6.0, 0.9), False),
    'Table 1 anchor/ja': (-1.0, 0.7, -1.7, (-4.7, 1.2), False),
    'Table 1 anchor/ko': (2.2, 1.2, 1.0, (-2.3, 4.7), False),
    'Checkpoint decomposition vT/es': (-2.6, 3.8, -2.9, -13.9, 0.0, 0.0),
    'Checkpoint decomposition vT/zh': (2.9, 4.4, -6.8, 6.0, 10.7, 0.0),
    'Checkpoint decomposition vT/ja': (2.4, 15.3, -6.8, 8.0, -10.7, 6.1),
    'Checkpoint decomposition vT/ko': (8.7, 26.2, -2.3, 0.0, 17.9, 1.7),
    'Checkpoint decomposition neuron/es': (0.6, 0.0, 1.0, 0.0, 1.9, 0.0),
    'Checkpoint decomposition neuron/zh': (7.3, 15.8, -5.8, 16.7, 11.5, -1.7),
    'Checkpoint decomposition neuron/ja': (-5.9, -39.7, 1.7, 6.0, 3.5, -0.8),
    'Checkpoint decomposition neuron/ko': (8.8, 25.0, 5.0, 10.8, 0.0, 3.3),
    'Checkpoint decomposition anchor/es': (-1.2, 1.0, 0.0, -2.1, -1.9, -3.0),
    'Checkpoint decomposition anchor/zh': (-2.5, 1.7, 10.0, -10.0, -8.2, -5.8),
    'Checkpoint decomposition anchor/ja': (-1.7, 3.4, 0.9, -11.1, 0.0, -1.7),
    'Checkpoint decomposition anchor/ko': (1.0, 0.0, 0.8, -5.8, 4.9, 5.0),
    'Multiplicity vT/es': (-2.6, False, (-7.5, 0.8), -2.5, (-6.9, 0.8)),
    'Multiplicity vT/zh': (2.9, False, (-4.1, 10.2), 1.8, (-4.4, 8.4)),
    'Multiplicity vT/ja': (2.4, False, (-4.7, 9.5), -1.3, (-7.3, 4.7)),
    'Multiplicity vT/ko': (8.7, False, (-0.1, 18.3), 5.7, (-0.8, 12.2)),
    'Multiplicity neuron/es': (0.6, False, (-1.5, 3.2), -0.4, (-2.1, 1.3)),
    'Multiplicity neuron/zh': (7.3, True, (2.5, 12.6), 7.6, (3.4, 12.5)),
    'Multiplicity neuron/ja': (-5.9, False, (-11.7, -0.1), -8.2, (-13.1, -3.4)),
    'Multiplicity neuron/ko': (8.8, True, (2.6, 15.7), 10.6, (5.5, 16.2)),
    'Multiplicity anchor/es': (-1.2, False, (-3.8, 1.1), -0.5, (-2.5, 1.4)),
    'Multiplicity anchor/zh': (-2.5, False, (-7.7, 2.7), -1.1, (-6.0, 3.2)),
    'Multiplicity anchor/ja': (-1.7, False, (-6.2, 2.5), -3.9, (-7.8, -0.7)),
    'Multiplicity anchor/ko': (1.0, False, (-4.0, 6.4), 3.4, (-1.7, 8.5)),
    'Robustness vT/es': (-2.6, (-5.9, 0.0), (-4.2, 0.2)),
    'Robustness vT/zh': (2.9, (-2.0, 7.8), (0.9, 5.3)),
    'Robustness vT/ja': (2.4, (-2.7, 7.2), (-0.9, 5.6)),
    'Robustness vT/ko': (8.7, (2.4, 15.3), (4.3, 11.5)),
    'Robustness neuron/es': (0.6, (-0.8, 2.2), (0.2, 0.7)),
    'Robustness neuron/zh': (7.3, (4.0, 11.1), (5.0, 10.6)),
    'Robustness neuron/ja': (-5.9, (-10.0, -1.9), (-8.8, 2.6)),
    'Robustness neuron/ko': (8.8, (4.5, 13.5), (4.8, 11.0)),
    'Robustness anchor/es': (-1.2, (-2.9, 0.4), (-1.7, -0.7)),
    'Robustness anchor/zh': (-2.5, (-6.0, 0.9), (-5.6, -0.6)),
    'Robustness anchor/ja': (-1.7, (-4.7, 1.2), (-3.0, 0.7)),
    'Robustness anchor/ko': (1.0, (-2.3, 4.7), (0.0, 2.7)),
    'Robustness v_T/es seeds': (-5.5, -1.2),
    'Robustness v_T/zh seeds': (2.0, 4.8),
    'Robustness v_T/ja seeds': (1.1, 7.4),
    'Robustness v_T/ko seeds': (8.0, 13.5),
    'Table 10 vT/es (gained, in form)': (91, 86),
    'Table 10 vT/zh (gained, in form)': (93, 61),
    'Table 10 vT/ja (gained, in form)': (55, 0),
    'Table 10 vT/ko (gained, in form)': (62, 0),
    'Table 10 neuron/es (gained, in form)': (27, 21),
    'Table 10 neuron/zh (gained, in form)': (60, 0),
    'Table 10 neuron/ja (gained, in form)': (27, 0),
    'Table 10 neuron/ko (gained, in form)': (73, 0),
    'Table 10 anchor/es (gained, in form)': (33, 28),
    'Table 10 anchor/zh (gained, in form)': (27, 3),
    'Table 10 anchor/ja (gained, in form)': (16, 4),
    'Table 10 anchor/ko (gained, in form)': (32, 7),
    'Section 5.2 content-only gains': 195,
    'Table 10 Spanish gains in the Spanish form': ((86, 91), (21, 27), (28, 33)),
    'Table 10 Chinese v_T gains in the Chinese-designated form': (61, 93),
    'Solvability Llama-3.1-70B/es': (98, 92),
    'Solvability Llama-3.1-70B/zh': (100, 421),
    'Solvability Llama-3.1-70B/ja': (100, 386),
    'Solvability Llama-3.1-70B/ko': (100, 451),
    'Solvability Llama-3.1-70B-It/es': (69, 42),
    'Solvability Llama-3.1-70B-It/zh': (93, 165),
    'Solvability Llama-3.1-70B-It/ja': (91, 119),
    'Solvability Llama-3.1-70B-It/ko': (91, 150),
    'Solvability Gemma-4-31B/es': (74, 413),
    'Solvability Gemma-4-31B/zh': (72, 219),
    'Solvability Gemma-4-31B/ja': (60, 152),
    'Solvability Gemma-4-31B/ko': (71, 254),
    'Solvability Gemma-4-31B-It/es': (35, 128),
    'Solvability Gemma-4-31B-It/zh': (63, 221),
    'Solvability Gemma-4-31B-It/ja': (52, 164),
    'Solvability Gemma-4-31B-It/ko': (50, 158),
    'Solvability Qwen3.6-27B/es': (59, 155),
    'Solvability Qwen3.6-27B/zh': (70, 218),
    'Solvability Qwen3.6-27B/ja': (63, 169),
    'Solvability Qwen3.6-27B/ko': (69, 208),
    'Section 4 solvability range Llama-3.1-70B': (97.8, 99.7),
    'Section 4 solvability range Gemma-4-31B-It': (35.2, 63.3),
    'Degenerate $v_T$ llama31_70': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate $v_T$ llama31_70_it': ((7, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate $v_T$ gemma4_31b': ((3, 0), (0, 28), (55, 9), (31, 3)),
    'Degenerate $v_T$ gemma4_31b_it': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate $v_T$ qwen36_27b': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate Lang.-neuron llama31_70': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate Lang.-neuron llama31_70_it': ((0, 0), (2, 0), (0, 0), (0, 0)),
    'Degenerate Lang.-neuron gemma4_31b': ((2, 70), (2, 85), (48, 37), (31, 96)),
    'Degenerate Lang.-neuron gemma4_31b_it': ((0, 0), (0, 0), (3, 0), (6, 0)),
    'Degenerate Lang.-neuron qwen36_27b': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate MRRE llama31_70': ((0, 0), (14, 3), (8, 10), (11, 3)),
    'Degenerate MRRE llama31_70_it': ((0, 0), (0, 3), (0, 3), (0, 0)),
    'Degenerate MRRE gemma4_31b': ((81, 100), (93, 100), (92, 100), (94, 100)),
    'Degenerate MRRE gemma4_31b_it': ((0, 100), (0, 100), (4, 100), (1, 100)),
    'Degenerate MRRE qwen36_27b': ((0, 1), (0, 0), (1, 1), (0, 0)),
    'Degenerate SCH llama31_70': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate SCH llama31_70_it': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Degenerate SCH gemma4_31b': ((97, 98), (5, 7), (0, 0), (19, 19)),
    'Degenerate SCH qwen36_27b': ((0, 0), (0, 0), (0, 0), (0, 0)),
    'Attempt vT/es': (98, 100, True, 13.0, 13.9, 15.6, 16.5),
    'Attempt vT/zh': (100, 94, True, 11.3, 10.6, 8.4, 8.4),
    'Attempt vT/ja': (89, 98, True, 2.5, 10.0, 0.2, 0.2),
    'Attempt vT/ko': (94, 99, True, 8.7, 12.8, 0.0, 0.0),
    'Attempt neuron/es': (100, 86, True, -0.7, -7.9, -1.3, -7.6),
    'Attempt neuron/zh': (99, 83, True, 7.3, 1.8, 0.0, 0.0),
    'Attempt neuron/ja': (90, 93, True, -5.9, -10.3, 0.0, 0.0),
    'Attempt neuron/ko': (93, 81, True, 8.8, 11.2, 0.0, 0.0),
    'Attempt anchor/es': (84, 60, False, 3.3, 1.1, 4.5, 3.4),
    'Attempt anchor/zh': (78, 59, False, -2.0, -6.6, 0.5, 0.8),
    'Attempt anchor/ja': (79, 57, False, -1.0, -5.0, 0.7, 0.0),
    'Attempt anchor/ko': (79, 59, False, 2.2, 2.0, 1.2, 0.0),
    'Section 4 cell sizes': (22, 160),
    'Table 1 v_T content and DFR (Spanish, Chinese)': ((13.0, 15.6), (11.3, 8.4)),
    'Crossed comparator delta_R': (5.8, (2.8, 9.5), 3.0, (-0.8, 7.1)),
    'Crossed comparator delta_M': (4.9, (2.0, 8.4), 4.0, (0.3, 7.8)),
    'Crossed comparator delta_L': (5.0, (2.0, 8.6), 2.4, (-0.9, 6.2)),
    'Table 16 Korean neurons without Gemma: comparators whose interval includes zero': 2,
    'C.2 neuron strength gemma4_31b/zh R': (18.8, (10.6, 27.8)),
    'C.2 neuron strength gemma4_31b/zh M': (10.5, (1.9, 19.5)),
    'C.2 neuron strength gemma4_31b/zh change_in_effect_M_minus_R': (-8.3, (-11.7, -5.5)),
    'Strength roster gemma4_31b/zh': (18.8, 10.5, 9.5),
    'C.2 neuron strength gemma4_31b_it/zh R': (1.0, (-4.0, 6.2)),
    'C.2 neuron strength gemma4_31b_it/zh M': (3.8, (-1.2, 9.4)),
    'C.2 neuron strength gemma4_31b_it/zh change_in_effect_M_minus_R': (2.8, (-0.5, 6.5)),
    'Strength roster gemma4_31b_it/zh': (1.0, 3.8, 3.3),
    'C.2 neuron strength llama31_70/zh R': (14.7, (7.0, 24.3)),
    'C.2 neuron strength llama31_70/zh M': (8.2, (1.0, 17.0)),
    'C.2 neuron strength llama31_70/zh change_in_effect_M_minus_R': (-6.5, (-10.2, -3.5)),
    'Strength roster llama31_70/zh': (14.7, 8.2, 7.0),
    'C.2 neuron strength llama31_70_it/zh R': (6.9, (0.6, 16.4)),
    'C.2 neuron strength llama31_70_it/zh M': (8.2, (1.8, 17.5)),
    'C.2 neuron strength llama31_70_it/zh change_in_effect_M_minus_R': (1.3, (-2.8, 5.4)),
    'Strength roster llama31_70_it/zh': (6.9, 8.2, 8.5),
    'C.2 neuron strength qwen36_27b/zh R': (0.7, (-2.6, 4.3)),
    'C.2 neuron strength qwen36_27b/zh M': (-0.5, (-3.5, 2.9)),
    'C.2 neuron strength qwen36_27b/zh change_in_effect_M_minus_R': (-1.2, (-2.8, 0.3)),
    'Strength roster qwen36_27b/zh': (0.7, -0.5, 1.3),
    'C.2 seed-inclusive macro zh delta_M': (6.0, (2.6, 9.8)),
    'C.2 macro zh delta_R': (8.4, (5.4, 11.8)),
    'C.2 macro zh M-R': (-2.4, (-3.9, -1.0)),
    'Output-matched gemma4_31b zh': (9.5, (0.8, 18.9), 4, 68.3),
    'Output-matched gemma4_31b_it zh': (3.3, (-2.8, 10.0), 2, 21.3),
    'Output-matched llama31_70 zh': (7.0, (-0.5, 16.2), 3, 1.3),
    'Output-matched llama31_70_it zh': (8.5, (2.7, 17.3), 0, 19.3),
    'Output-matched qwen36_27b zh': (1.3, (-1.8, 4.9), 0, 0.0),
    'C.2 output-matched macro zh': (5.9, (2.9, 9.4)),
    'C.2 output-matched L-R zh': (-2.5, (-4.0, -1.1)),
    'C.2 neuron strength gemma4_31b/ko R': (24.3, (15.0, 34.9)),
    'C.2 neuron strength gemma4_31b/ko M': (16.0, (7.5, 25.4)),
    'C.2 neuron strength gemma4_31b/ko change_in_effect_M_minus_R': (-8.3, (-12.1, -4.6)),
    'Strength roster gemma4_31b/ko': (24.3, 16.0, 22.7),
    'C.2 neuron strength gemma4_31b_it/ko R': (3.7, (-1.7, 9.9)),
    'C.2 neuron strength gemma4_31b_it/ko M': (4.3, (-1.9, 11.8)),
    'C.2 neuron strength gemma4_31b_it/ko change_in_effect_M_minus_R': (0.7, (-1.6, 3.0)),
    'Strength roster gemma4_31b_it/ko': (3.7, 4.3, 2.3),
    'C.2 neuron strength llama31_70/ko R': (7.3, (-1.2, 17.2)),
    'C.2 neuron strength llama31_70/ko M': (5.8, (-1.7, 14.8)),
    'C.2 neuron strength llama31_70/ko change_in_effect_M_minus_R': (-1.5, (-5.4, 2.3)),
    'Strength roster llama31_70/ko': (7.3, 5.8, 4.0),
    'C.2 neuron strength llama31_70_it/ko R': (0.0, (-9.4, 10.8)),
    'C.2 neuron strength llama31_70_it/ko M': (4.3, (-3.7, 13.0)),
    'C.2 neuron strength llama31_70_it/ko change_in_effect_M_minus_R': (4.3, (-0.9, 9.3)),
    'Strength roster llama31_70_it/ko': (0.0, 4.3, 2.3),
    'C.2 neuron strength qwen36_27b/ko R': (0.8, (-4.8, 7.2)),
    'C.2 neuron strength qwen36_27b/ko M': (1.5, (-4.4, 8.3)),
    'C.2 neuron strength qwen36_27b/ko change_in_effect_M_minus_R': (0.7, (-1.9, 3.4)),
    'Strength roster qwen36_27b/ko': (0.8, 1.5, 1.2),
    'C.2 seed-inclusive macro ko delta_M': (6.4, (2.5, 10.7)),
    'C.2 macro ko delta_R': (7.2, (3.6, 11.2)),
    'C.2 macro ko M-R': (-0.8, (-2.5, 0.8)),
    'Output-matched gemma4_31b ko': (22.7, (13.7, 32.6), 5, 29.3),
    'Output-matched gemma4_31b_it ko': (2.3, (-3.0, 8.7), 1, 1.5),
    'Output-matched llama31_70 ko': (4.0, (-3.3, 12.4), 4, 5.3),
    'Output-matched llama31_70_it ko': (2.3, (-5.4, 11.7), 1, 11.8),
    'Output-matched qwen36_27b ko': (1.2, (-4.6, 7.3), 1, 0.7),
    'C.2 output-matched macro ko': (6.5, (3.2, 10.1)),
    'C.2 output-matched L-R ko': (-0.7, (-2.1, 0.6)),
    'C.2 held-out output-matched seeds': (21, 50),
    'Section 5.3 Chinese drop under the energy-matched comparator': 2.4,
    'Neuron sensitivity zh cluster-disjoint': (7.6, 3.4, 12.5),
    'Neuron sensitivity zh both arms attempt': (0.8, -6.7, 7.9),
    'Neuron sensitivity zh construction-matched R': (8.4, 5.4, 11.8),
    'Neuron sensitivity zh energy-matched M': (6.0, 3.0, 9.3),
    'Neuron sensitivity zh output-matched L': (5.9, 2.9, 9.4),
    'Neuron sensitivity ko cluster-disjoint': (10.6, 5.5, 16.2),
    'Neuron sensitivity ko both arms attempt': (10.7, 1.3, 23.2),
    'Neuron sensitivity ko construction-matched R': (7.2, 3.6, 11.2),
    'Neuron sensitivity ko energy-matched M': (6.4, 3.0, 10.0),
    'Neuron sensitivity ko output-matched L': (6.5, 3.2, 10.1),
    'D.3 SCH strength es content R': 6.63,
    'D.3 SCH strength es content M': 0.22,
    'D.3 SCH strength es form R': 6.4,
    'D.3 SCH strength es form M': 0.85,
    'D.3 SCH strength zh content R': 11.55,
    'D.3 SCH strength zh content M': 7.44,
    'D.3 SCH strength zh form R': 10.88,
    'D.3 SCH strength zh form M': 8.71,
    'D.3 SCH strength ja content R': 3.37,
    'D.3 SCH strength ja content M': -1.16,
    'D.3 SCH strength ja form R': 5.04,
    'D.3 SCH strength ja form M': 0.98,
    'D.3 SCH strength ko content R': 2.44,
    'D.3 SCH strength ko content M': -1.22,
    'D.3 SCH strength ko form R': 2.38,
    'D.3 SCH strength ko form M': 0.65,
    'D SCH table es': (11.5, 11.2, 0.3, (-0.8, 1.6)),
    'D SCH table zh': (11.3, 9.5, 1.8, (0.5, 3.3)),
    'D SCH table ja': (6.0, 6.7, -0.7, (-2.0, 0.0)),
    'D SCH table ko': (4.8, 4.4, 0.4, (-1.4, 2.2)),
    'SCH gates qwen36_27b': (6, 0.289, -0.05, 0.34, True),
    'SCH gates llama31_70': (62, 0.464, 0.042, 0.423, True),
    'SCH gates llama31_70_it': (78, 0.702, 0.024, 0.678, True),
    'SCH gates gemma4_31b': (18, 0.723, 0.03, 0.693, True),
    'SCH gates gemma4_31b_it': (36, -2.235, 0.867, -3.102, False),
}

if MATCHER_POLICY != "attribution_strict":
    print(f"[SKIP] manuscript checks: the manuscript reports the attribution-strict policy, not {MATCHER_POLICY}")
else:
    import importlib as _il
    _CK = ["gemma4_31b", "gemma4_31b_it", "llama31_70", "llama31_70_it", "qwen36_27b"]
    sys.path.insert(0, HERE)
    _S = _il.import_module("paired_effects")

    # ---- Table 1: arm rates and paired effects (shared answer-cluster bootstrap, seeds 900+i) ----
    _cells = _S.collect()
    _C = {(c["family"], c["lang"]): c for c in _cells}
    def _arm_rates(_fam, _lang):
        """Checkpoint-equal intervention and control rates (content, designated form) from the paired rows."""
        _rows = _S.cell_rows(_fam, _lang)
        _mac = lambda _i: 100 * sum(sum(_r[_i] for _r in _m) / len(_m) for _m in _rows) / len(_rows)
        return {"content_I": _mac(1), "content_R": _mac(2), "ilr_I": _mac(3), "ilr_R": _mac(4)}
    _arm = {(_f, _l): _arm_rates(_f, _l) for _f in _S.SPANISH_FAMILIES for _l in _S.LANGS}
    for _fam in _S.SPANISH_FAMILIES:
        for _lang in _S.LANGS:
            _c, _a = _C[(_fam, _lang)], _arm[(_fam, _lang)]
            near(f"Table 1 {_fam}/{_lang} content I-R from arm rates", _a["content_I"] - _a["content_R"], _c["content"], 0.11)
            near(f"Table 1 {_fam}/{_lang} DFR I-R from arm rates", _a["ilr_I"] - _a["ilr_R"], _c["form"], 0.11)
            _mu = _c.get("multiplicity", {})
            _robust = bool(_mu.get("survives")) and _mu["p_signflip"] < 0.05 / _S.MULT_FAMILY_SIZE \
                and (_c["delta_cluster_disjoint"]["delta_ci95"][0] > 0 or _c["delta_cluster_disjoint"]["delta_ci95"][1] < 0)
            # Delta_C, Delta_F, delta and its interval, and the bold joint-pass mark of Table 1; the
            # printed Delta_C and Delta_F intervals are checked against PAIRED_INTERVALS above.
            paper(f"Table 1 {_fam}/{_lang}",
                  (r1(_c["content"]), r1(_c["form"]), r1(_c["delta"]), ci1(_c["delta_ci95"]), _robust))

    # ---- Appendix tables: checkpoint decomposition, multiplicity, separation robustness ----
    _seeds = _S.seed_sensitivity()
    for _c in _cells:
        if _c["family"] in _S.SPANISH_FAMILIES:
            _by = {p["model"]: p["delta"] for p in _c["per_checkpoint"]}
            paper(f"Checkpoint decomposition {_c['family']}/{_c['lang']}",
                  (r1(_c["delta"]),) + tuple(r1(_by[m]) for m in _S.ROSTER))
    _robust_cells = []
    for _fam in _S.SPANISH_FAMILIES:
        for _lang in _S.LANGS:
            _c = _C[(_fam, _lang)]
            _mu, _cd = _c["multiplicity"], _c["delta_cluster_disjoint"]
            _all3 = _mu["survives"] and _mu["p_signflip"] < 0.05 / _S.MULT_FAMILY_SIZE and \
                (_cd["delta_ci95"][0] > 0 or _cd["delta_ci95"][1] < 0)
            if _all3:
                _robust_cells.append((_fam, _lang))
            paper(f"Multiplicity {_fam}/{_lang}",
                  (r1(_c["delta"]), _all3, ci1(_mu["bonferroni_ci"]), r1(_cd["delta"]), ci1(_cd["delta_ci95"])))
    check("Robust cells (Bonferroni, cluster-disjoint, sign-flip)", sorted(_robust_cells), [("neuron", "ko"), ("neuron", "zh")])
    _nj = _C[("neuron", "ja")]
    check("Japanese neurons: Bonferroni and cluster-disjoint hold, sign-flip does not",
          (_nj["multiplicity"]["survives"], _nj["delta_cluster_disjoint"]["delta_ci95"][1] < 0,
           _nj["multiplicity"]["p_signflip"] < 0.05 / _S.MULT_FAMILY_SIZE), (True, True, False))
    for _fam in _S.SPANISH_FAMILIES:
        for _lang in _S.LANGS:
            _c = _C[(_fam, _lang)]
            _lo = sorted(_c["delta_leave_one_checkpoint_out"].values())
            paper(f"Robustness {_fam}/{_lang}", (r1(_c["delta"]), ci1(_c["delta_ci95"]), (r1(_lo[0]), r1(_lo[-1]))))
    for _lang in _S.LANGS:
        paper(f"Robustness v_T/{_lang} seeds", (r1(_seeds[_lang]["min"]), r1(_seeds[_lang]["max"])))

    # ---- Table 10: item-level transitions ----
    _ST = _il.import_module("state_transitions")
    _st_cells = {(f, l): _ST.cell(f, l) for f in _ST.FAMILIES for l in _S.LANGS}
    _stj = _ms_load("state_transitions", "summary.json")
    for (_f, _l), _v in _st_cells.items():
        check(f"Transitions {_f}/{_l} match artifact summary", {k: _v[k] for k in _ST.KINDS},
              {k: _stj[f"{_f}:{_l}"][k] for k in _ST.KINDS})
        paper(f"Table 10 {_f}/{_l} (gained, in form)", (_v["0->F"] + _v["0->c"], _v["0->F"]))

    # ---- Realized forms (Figure 3, App. A.9) and the Sec. 5.2 summary ----
    _rf = _ms_load("realized_form_composition", "realized_form_composition.json")["cells"]
    for _k, _c in _rf.items():
        _v = _st_cells[tuple(_k.split(":"))]
        check(f"Realized forms {_k} match the transitions (gained, in form)",
              (_c["gained"], _c["designated"]), (_v["0->F"] + _v["0->c"], _v["0->F"]))
    paper("Section 5.2 content-only gains", sum(_rf[k]["gained"] for k in ("vT:ko", "neuron:zh", "neuron:ko")))
    paper("Table 10 Spanish gains in the Spanish form",
          tuple((_rf[k]["designated"], _rf[k]["gained"]) for k in ("vT:es", "neuron:es", "anchor:es")))
    paper("Table 10 Chinese v_T gains in the Chinese-designated form", (_rf["vT:zh"]["designated"], _rf["vT:zh"]["gained"]))

    # ---- Hop-level solvability (Sec. 4 and App. A.7) ----
    _hop = _il.import_module("hop_solvability").rows()
    for (_label, _lang), (_n, _pct) in _hop.items():
        paper(f"Solvability {_label}/{_lang}", (round(_pct), _n))
    for _label in ("Llama-3.1-70B", "Gemma-4-31B-It"):
        _v = [_pct for (_lb, _), (_, _pct) in _hop.items() if _lb == _label]
        paper(f"Section 4 solvability range {_label}", (r1(min(_v)), r1(max(_v))))

    # ---- Degeneration and attempt tables (App. C.1) ----
    _DG = _il.import_module("degenerate_outputs")
    _deg = _DG.collect()
    for _family in [f for _, f in _DG.FAMILIES] + ["SCH"]:
        for _ckpt, _ in _DG.CHECKPOINTS:
            _vals = []
            for _lang in _DG.LANGS:
                _i, _c = _DG.pct(_deg[(_family, _ckpt, _lang, "I")]), _DG.pct(_deg[(_family, _ckpt, _lang, "C")])
                _vals.append(None if _i is None else (round(_i), round(_c)))
            if any(_vals):
                paper(f"Degenerate {_family} {_ckpt}", tuple(_vals))
    _att = _il.import_module("attempt_decomposition").collect()
    for (_f, _l), _r in _att.items():
        paper(f"Attempt {_f}/{_l}", (round(_r["attI"]), round(_r["attC"]), _r["cells"] == _r["total"],
                                     r1(_r["cU"]), r1(_r["cK"]), r1(_r["fU"]), r1(_r["fK"])))

    # ---- Checkpoint-level values, checkpoint exclusion, crossed comparators ----
    _gk = {_p["model"]: _p["delta"] for _p in _C[("vT", "ko")]["per_checkpoint"]}
    check("Figure 4 Korean v_T rests mainly on Gemma-4-31B", max(_gk, key=_gk.get), "gemma4_31b")
    _ns = [_p["n"] for _f in _S.SPANISH_FAMILIES for _l in _S.LANGS for _p in _C[(_f, _l)]["per_checkpoint"]]
    paper("Section 4 cell sizes", (min(_ns), max(_ns)))
    paper("Table 1 v_T content and DFR (Spanish, Chinese)",
          tuple((r1(_C[("vT", _l)]["content"]), r1(_C[("vT", _l)]["form"])) for _l in ("es", "zh")))
    _ce = _ms_load("checkpoint_exclusion", "checkpoint_exclusion.json")
    for _k in ("delta_R", "delta_M", "delta_L"):
        _z, _o = _ce["neuron_comparators"]["zh"][_k], _ce["neuron_comparators"]["ko"][_k]
        paper(f"Crossed comparator {_k}", (r1(_z[0]), ci1(_z[1]), r1(_o[0]), ci1(_o[1])))
        check(f"Crossed comparator {_k}: Chinese interval excludes zero", _z[1][0] > 0, True)
    paper("Table 16 Korean neurons without Gemma: comparators whose interval includes zero",
          sum(_ce["neuron_comparators"]["ko"][_k][1][0] <= 0 for _k in ("delta_R", "delta_M", "delta_L")))
    # ---- Language-neuron comparators (appendix B.2) and strength-roster table ----
    _macro = _ms_load("energy_matched_control", "summary", "macro.json")["results"]
    _lm = _ms_load("output_matched_control", "summary.json")
    for _t in ("zh", "ko"):
        for _m in _CK:
            _run = _ms_load("energy_matched_control", _m, "analysis.json")["results"][_t]["effects"]["delta"]
            _per = _macro[_t]["per_checkpoint"][_m]
            near(f"Strength summary {_m}/{_t} delta_R vs run", _per["delta_R"], _run["R"]["pp"], 0.006)
            near(f"Strength summary {_m}/{_t} delta_M vs run", _per["delta_M"], _run["M"]["pp"], 0.006)
            for _k in ("R", "M", "change_in_effect_M_minus_R"):
                paper(f"C.2 neuron strength {_m}/{_t} {_k}", (r1(_run[_k]["pp"]), ci1(_run[_k]["ci95_pp"])))
            paper(f"Strength roster {_m}/{_t}", (r1(_per["delta_R"]), r1(_per["delta_M"]),
                                                r1(_lm["results"][_t]["per_checkpoint"][_m]["delta_L"])))
        _mm = _macro[_t]["macro"]
        _si = _ms_load("energy_matched_control", "summary", "seed_and_checkpoint_random_effects.json")["results"][_t]["M"]["equal_checkpoint_macro"]
        near(f"C.2 seed-inclusive macro {_t} equals checkpoint macro", _si["delta"], _mm["delta_M"], 0.06)
        paper(f"C.2 seed-inclusive macro {_t} delta_M", (r1(_si["delta"]), ci1(_si["seed_inclusive_ci95"])))
        paper(f"C.2 macro {_t} delta_R", (r1(_mm["delta_R"]), ci1(_mm["delta_R_ci95"])))
        paper(f"C.2 macro {_t} M-R", (r1(_mm["M_minus_R"]), ci1(_mm["M_minus_R_ci95"])))
        _r = _lm["results"][_t]
        for _m in _CK:
            _p = _r["per_checkpoint"][_m]
            _a = _ms_load("output_matched_control", _m, "analysis.json")["results"][_t]["effects"]["delta"]
            check(f"Output-matched {_m} {_t} delta_L matches run", round(_p["delta_L"], 4), round(_a["L"]["pp"], 4))
            paper(f"Output-matched {_m} {_t}", (r1(_p["delta_L"]), ci1(_a["L"]["ci95_pp"]), len(_p["matched_seeds"]),
                                               r1(_p["degenerate"]["L"])))
        paper(f"C.2 output-matched macro {_t}", (r1(_r["macro"]["delta_L"]), ci1(_r["macro"]["delta_L_ci95"])))
        paper(f"C.2 output-matched L-R {_t}", (r1(_r["macro"]["L_minus_R"]), ci1(_r["macro"]["L_minus_R_ci95"])))
    paper("C.2 held-out output-matched seeds", (_lm["held_out_matched_seeds"], _lm["seeds_total"]))
    paper("Section 5.3 Chinese drop under the energy-matched comparator",
          r1(_macro["zh"]["macro"]["delta_R"] - _macro["zh"]["macro"]["delta_M"]))

    # ---- Language-neuron sensitivities (Tables 1, 8, 14, 15; App. C.1) ----
    _pa = {r["lang"]: r for r in _ms_load("both_arms_attempt", "results.json")["results"]}
    for _t in ("zh", "ko"):
        _c, _cd = _C[("neuron", _t)], _C[("neuron", _t)]["delta_cluster_disjoint"]
        _rows = {"cluster-disjoint": (_cd["delta"], _cd["delta_ci95"], 1),
                 "both arms attempt": (_pa[_t]["both"]["content"]["estimate"], _pa[_t]["both"]["content"]["ci95"], 1),
                 "construction-matched R": (_macro[_t]["macro"]["delta_R"], _macro[_t]["macro"]["delta_R_ci95"], 1),
                 "energy-matched M": (_macro[_t]["macro"]["delta_M"], _macro[_t]["macro"]["delta_M_ci95"], 1),
                 "output-matched L": (_lm["results"][_t]["macro"]["delta_L"], _lm["results"][_t]["macro"]["delta_L_ci95"], 1)}
        for _name, (_pt, (_lo, _hi), _d) in _rows.items():
            paper(f"Neuron sensitivity {_t} {_name}", (round(_pt, _d) + 0.0, r1(_lo), r1(_hi)))
    check("Appendix C.1 both-attempt Chinese interval includes zero", _pa["zh"]["both"]["content"]["ci95"][0] < 0, True)

    # ---- SCH (App. D) ----
    _SCHM = ("gemma4_31b", "llama31_70", "llama31_70_it", "qwen36_27b")
    _sch_eff = {(_l, _m): _ms_load("sch_strength_control", _m, "analysis.json")[_l]["results"][_l]["effects"]
                for _l in ("es", "zh", "ja", "ko") for _m in _SCHM}
    for _lang in ("es", "zh", "ja", "ko"):
        for _metric in ("content", "form"):
            for _k in ("R", "M"):
                paper(f"D.3 SCH strength {_lang} {_metric} {_k}",
                      r2(sum(_sch_eff[(_lang, _m)][_metric][_k]["pp"] for _m in _SCHM) / len(_SCHM)))
    _gapmax = max(abs(_C[("sch", _l)]["content"] - _C[("sch", _l)]["form"]) for _l in ("es", "zh", "ja", "ko"))
    check("Appendix D SCH content and DFR within two points", _gapmax < 2.0, True)
    _schM = {_l: [sum(_sch_eff[(_l, _m)][_k]["M"]["pp"] for _m in _SCHM) / 4 for _k in ("content", "form")]
             for _l in ("es", "zh", "ja", "ko")}
    check("Appendix D.3 SCH calibrated: three languages near zero or negative content, Chinese positive both",
          (sum(_schM[_l][0] < 1 for _l in ("es", "ja", "ko")), _schM["zh"][0] > 0 and _schM["zh"][1] > 0), (3, True))
    for _lang in ("es", "zh", "ja", "ko"):
        _c = _C[("sch", _lang)]
        paper(f"D SCH table {_lang}", (r1(_c["content"]), r1(_c["form"]), r1(_c["delta"]), ci1(_c["delta_ci95"])))
    for _m in ("qwen36_27b", "llama31_70", "llama31_70_it", "gemma4_31b", "gemma4_31b_it"):
        _pool = load(f"interventions/sch_pool_{_m}_general.json")
        _val = _pool["validation"]
        paper(f"SCH gates {_m}", (_pool["selected_count"], round(_val["mean_ablation_dnll"], 3) + 0.0,
                                  round(_val["layer_matched_random_dnll"], 3) + 0.0, round(_val["selected_minus_random_dnll"], 3) + 0.0,
                                  _pool["pool_status"] == "positive_control_supported"))

    _seen = {_lb for _lb, _ in _PAPER_SEEN}
    check("Every stored manuscript value is checked", sorted(set(PAPER) - _seen), [])
    check("Every manuscript label is checked once", len(_PAPER_SEEN), len(_seen))


# --- Values verified at camera-ready (main text and appendix) ---------------------------
# Sec. 5.3 / App. C.2: Gemma-4-31B energy-matched rerun (five construction-matched R and
# five calibrated M draws; the rerun regenerates the intervention arm).
_emg = load("energy_matched_control/gemma4_31b/analysis.json")["results"]
check("Sec. 5.3 Gemma calibrated control: delta_M - delta_R, delta_M (zh, ko)",
      tuple((round(_emg[_t]["effects"]["delta"]["change_in_effect_M_minus_R"]["pp"], 1),
             round(_emg[_t]["effects"]["delta"]["M"]["pp"], 1)) for _t in ("zh", "ko")),
      ((-8.3, 10.5), (-8.3, 16.0)))
check("App. C.2 Gemma rerun delta_R (zh, ko)",
      tuple(round(_emg[_t]["effects"]["delta"]["R"]["pp"], 1) for _t in ("zh", "ko")), (18.8, 24.3))
_mz = [_emg["zh"]["arm_rates"][f"M{_s}"]["degenerate"] for _s in (7, 17, 27, 37, 47)]
check("Sec. 5.3 / App. C.2 Gemma zh: rerun intervention and calibrated-control degeneracy",
      (round(_emg["zh"]["arm_rates"]["I"]["degenerate"], 1), round(sum(_mz) / len(_mz), 1)), (4.2, 72.5))
# Sec. 5.2 / App. B.1: language-neuron gain subset and its offsets.
_pe_mod = _load_module("paired_effects")
_gain = tuple(sum(1 for _m in _pe_mod.cell_rows("neuron", _l) for _r in _m if _r[1] and not _r[2]) for _l in ("zh", "ko"))
_off = tuple(sum(1 for _m in _pe_mod.cell_rows("neuron", _l) for _r in _m if _r[2] and not _r[1]) for _l in ("zh", "ko"))
check("Sec. 5.2 language-neuron intervention-only gains and control-only offsets (zh, ko)", (_gain, _off), ((60, 73), (23, 20)))
_aud = load("human_form_audit/annotations/summary.json")
_tr = {}
for _l in ("zh", "ko"):
    for _t in _aud[_l]["content_transitions"]:
        if _t["intervention"] == "correct":
            _tr[_t["control"]] = _tr.get(_t["control"], 0) + _t["n"]
check("App. B.1 correct-intervention pairs by control judgment (incorrect, unassessable, uncertain, correct)",
      (_tr["incorrect"], _tr["unassessable"], _tr["uncertain"], _tr["correct"]), (34, 46, 4, 40))
# Sec. 5.2: the stage-1 judgments move each Spanish cell by less than half a point.
_hv = _load_module("human_validation_sensitivity")
_hj = _hv.judgments()
_shift = [abs(_pe_mod.point([_hv.rows(_f, _m, "es", _hj, "all") for _m in _pe_mod.ROSTER])[2]
              - _pe_mod.point([_hv.rows(_f, _m, "es", _hj, "locked") for _m in _pe_mod.ROSTER])[2])
          for _f in ("vT", "neuron")]
check("Sec. 5.2 stage-1 judgments move the Spanish cells by less than half a point", max(_shift) < 0.5, True)

print(f"\n{PASS} pass, {len(FAIL)} mismatch")
if FAIL:
    for failure in FAIL:
        print(" -", failure)
    sys.exit(1)
