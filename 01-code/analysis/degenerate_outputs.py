#!/usr/bin/env python3
"""Degenerate-output diagnostic for the audited intervention arms.

Reparses the preserved generation text in 02-data/results/interventions/ and reports,
for every (family, checkpoint, language, arm) cell, the fraction of items whose
scored line is not an answer attempt.

Scoring in the paper uses the first non-empty generated line, so the diagnostic
is defined on that line. A line is DEGENERATE if any of:

  1. it is empty (the whole generation is whitespace);
  2. it begins with a prompt-template marker
     (Instruction:/Question:/Answer:/Pregunta:/Respuesta:);
  3. a whitespace-delimited token repeats >= 4 times consecutively;
  4. a character 1-3-gram repeats >= 4 times consecutively (whitespace removed).

The rule is deliberately conservative and UNDERCOUNTS: off-task but fluent and
non-repetitive output (e.g. "announcement_id: 12345") is not flagged.

Usage:
    python3 degenerate_outputs.py
"""
import collections
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.environ.get("RESULTS_DIR") or os.path.join(HERE, "..", "..", "02-data", "results"))
INTERVENTIONS = os.path.join(RESULTS, "interventions")

MARKER = re.compile(r"^(Instruction|Question|Answer|Pregunta|Respuesta)\s*[::]")

FAMILIES = [("surfacing_vector", "$v_T$"), ("language_neurons", "Lang.-neuron"), ("mrre", "MRRE")]
CHECKPOINTS = [
    ("llama31_70", "Llama-70B"),
    ("llama31_70_it", "Llama-70B-It"),
    ("gemma4_31b", "Gemma-31B"),
    ("gemma4_31b_it", "Gemma-31B-It"),
    ("qwen36_27b", "Qwen-27B"),
]
LANGS = ["es", "zh", "ja", "ko"]


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


def collect():
    """cell -> Counter(n=, d=) keyed by (family, checkpoint, lang, arm)."""
    cells = collections.defaultdict(collections.Counter)

    for prefix, family in FAMILIES:
        for path in sorted(glob.glob(os.path.join(INTERVENTIONS, prefix + "_*.json"))):
            ckpt = os.path.basename(path)[len(prefix) + 1 : -len(".json")]
            data = json.load(open(path))
            for lang, cell in data.get("eval", {}).items():
                for key, arm in (("steer_items", "I"), ("rand_items", "C")):
                    for item in cell.get(key, []):
                        k = (family, ckpt, lang, arm)
                        cells[k]["n"] += 1
                        if is_degenerate(first_line(item.get("txt", ""))):
                            cells[k]["d"] += 1

    for path in sorted(glob.glob(os.path.join(INTERVENTIONS, "sch_amplification_*_general.json"))):
        base = os.path.basename(path)
        ckpt = base[len("sch_amplification_") : -len("_general.json")]
        data = json.load(open(path))
        for lang, cell in data.get("by_lang", {}).items():
            for item in cell.get("items", []):
                for key, arm in (("amp_a1.0", "I"), ("rand_a1.0", "C")):
                    if key not in item:
                        continue
                    k = ("SCH", ckpt, lang, arm)
                    cells[k]["n"] += 1
                    if is_degenerate(first_line(item[key].get("txt", ""))):
                        cells[k]["d"] += 1
    return cells


def pct(counter):
    return 100.0 * counter["d"] / counter["n"] if counter["n"] else None


def console(cells):
    for family in [f for _, f in FAMILIES] + ["SCH"]:
        print(f"=== {family} === degenerate first-line %, intervention/control")
        print(f"{'checkpoint':14s} " + " ".join(f"{l:>13s}" for l in LANGS))
        for ckpt, label in CHECKPOINTS:
            out = []
            for lang in LANGS:
                i = pct(cells[(family, ckpt, lang, "I")])
                c = pct(cells[(family, ckpt, lang, "C")])
                out.append("--" if i is None else f"{i:5.1f}/{c:5.1f}")
            print(f"{label:14s} " + " ".join(f"{o:>13s}" for o in out))
        print()


if __name__ == "__main__":
    console(collect())
