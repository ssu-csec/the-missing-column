#!/usr/bin/env python3
"""Attempt-rate decomposition of the audited Spanish-prompt contrasts.

Scoring credits an output only if its scored line realizes the designated form,
so an arm that stops producing answer attempts registers as an answer-form
failure. This script separates the two by reporting, per family and language:

  * the attempt rate of each arm (1 - degenerate-scored-line rate);
  * the primary intervention-minus-control contrasts, unconditional; and
  * the same contrasts computed within the attempting subset of each arm.

The conditional contrast is a difference of arm-level rates on different
denominators, not a paired item-level estimand. It is a diagnostic decomposition
and does not replace the primary contrast.

Content and answer-form scoring are imported from verify_numbers.py so that the
unconditional columns match the main paired-effects table (Table 1); the values
are those of the attempt-rate decomposition table (App. C.1, Table 13).

Usage:
    python3 attempt_decomposition.py
"""
import contextlib
import io
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
VERIFY = os.path.join(HERE, "verify_numbers.py")

MARKER = re.compile(r"^(Instruction|Question|Answer|Pregunta|Respuesta)\s*[::]")
LANGS = ["es", "zh", "ja", "ko"]
LABEL = {"vT": "v_T", "neuron": "Neuron", "anchor": "MRRE"}


def load_scorers():
    """Exec verify_numbers.py up to its first top-level assertion."""
    src = io.open(VERIFY, encoding="utf-8").read().splitlines(True)
    cut = next(i for i, l in enumerate(src) if re.match(r"^(check|near)\(", l))
    ns = {"__name__": "verify_numbers", "__file__": VERIFY}
    with contextlib.redirect_stdout(io.StringIO()):
        exec("".join(src[:cut]), ns)
    return ns


def first_line(text):
    for line in (text or "").strip().splitlines():
        if line.strip():
            return line.strip()
    return ""


def is_degenerate(line):
    if not line:
        return True
    if MARKER.match(line):
        return True
    tokens = line.split()
    if len(tokens) >= 4:
        run = 1
        for a, b in zip(tokens, tokens[1:]):
            run = run + 1 if a == b else 1
            if run >= 4:
                return True
    squashed = re.sub(r"\s+", "", line)
    for n in (1, 2, 3):
        for i in range(max(0, len(squashed) - n * 4 + 1)):
            gram = squashed[i : i + n]
            if gram and squashed[i : i + n * 4] == gram * 4:
                return True
    return False


def cell(ns, family, model, lang):
    """Per-checkpoint counts for one family/model/language."""
    content, ilr, scoreable = ns["content"], ns["ilr"], ns["scoreable"]
    d = ns["load"]("interventions/" + ns["FAMILIES"][family].format(model=model))
    chains, floors = d["eval_chains"], d["floors"]
    steer, rand = d["eval"][lang]["steer_items"], d["eval"][lang]["rand_items"]
    acc = {"n": 0, "aI": 0, "aC": 0, "cI": 0, "cC": 0, "fI": 0, "fC": 0}
    for i, chain in enumerate(chains):
        if (chain not in ns["CHAINS"] or chain not in floors
                or i >= len(steer) or i >= len(rand)
                or not scoreable(chain, lang)):
            continue
        acc["n"] += 1
        for arm, tag in ((steer[i], "I"), (rand[i], "C")):
            txt = arm.get("txt", "")
            acc["a" + tag] += not is_degenerate(first_line(txt))
            acc["c" + tag] += bool(content(txt, chain))
            acc["f" + tag] += bool(ilr(txt, chain, lang))
    return acc


def macro(ns, family, lang):
    rows = [cell(ns, family, m, lang) for m in ns["ROSTER"]]
    rows = [r for r in rows if r["n"]]
    mean = lambda v: sum(v) / len(v) if v else float("nan")
    uncond_c = mean([100 * (r["cI"] - r["cC"]) / r["n"] for r in rows])
    uncond_f = mean([100 * (r["fI"] - r["fC"]) / r["n"] for r in rows])
    ok = [r for r in rows if r["aI"] and r["aC"]]
    cond_c = mean([100 * (r["cI"] / r["aI"] - r["cC"] / r["aC"]) for r in ok])
    cond_f = mean([100 * (r["fI"] / r["aI"] - r["fC"] / r["aC"]) for r in ok])
    return {
        "attI": mean([100 * r["aI"] / r["n"] for r in rows]),
        "attC": mean([100 * r["aC"] / r["n"] for r in rows]),
        "cU": uncond_c, "cK": cond_c,
        "fU": uncond_f, "fK": cond_f,
        "cells": len(ok), "total": len(rows),
    }


def collect():
    ns = load_scorers()
    return {(f, l): macro(ns, f, l) for f in ("vT", "neuron", "anchor") for l in LANGS}


def console(res):
    print(f"{'family':14s}{'lang':5s}{'attempt I/C':>13s}"
          f"{'content D':>11s}{'| attempt':>11s}{'form D':>11s}{'| attempt':>11s}")
    for family in ("vT", "neuron", "anchor"):
        for lang in LANGS:
            r = res[(family, lang)]
            note = "" if r["cells"] == r["total"] else f"  [{r['cells']}/{r['total']} cells]"
            print(f"{LABEL[family]:14s}{lang:5s}{r['attI']:5.1f}/{r['attC']:<7.1f}"
                  f"{r['cU']:+11.1f}{r['cK']:+11.1f}{r['fU']:+11.1f}{r['fK']:+11.1f}{note}")
        print()


if __name__ == "__main__":
    console(collect())
