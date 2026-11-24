#!/usr/bin/env python3
"""Checkpoint-level answer-form composition of content-matched SCH outputs (Appendix D, Table 23).

For each qualified checkpoint, designated language (Korean, Chinese, Japanese), and head pool
(general, language-specific), counts the SCH-arm outputs at alpha = 1.0 that either the run's
stored entity flag or the locked content matcher credits with content (the union of the two
matchers), and splits them: the designated (target) form under the locked matcher, i.e. ILR;
otherwise outputs that the run's stored output-language label marks as English; otherwise
other/unattributed.  Items follow the attribution-strict population used everywhere else.
This is the rule behind the reviewed version of the table.  Prints LaTeX rows.

    python3 sch_form_composition.py
"""
import paired_effects as S

V = S.V
MODELS = [("qwen36_27b", "Qwen-27B"), ("llama31_70", "Llama-70B"),
          ("llama31_70_it", "Llama-70B-It"), ("gemma4_31b", "Gemma-31B")]
LANGS = ("ko", "zh", "ja")
# Pools that did not pass their held-out gate (see Table sch-model-results).
FAILED = {("qwen36_27b", "specific", "zh"), ("qwen36_27b", "specific", "ja")}


def first_line(text):
    text = (text or "").strip()
    return text.splitlines()[0] if text else ""


def composition(model, pool, lang):
    d = V.load(f"interventions/sch_amplification_{model}_{pool}.json")
    n = target = english = 0
    for item in d["by_lang"][lang]["items"]:
        chain = item["chain"]
        if chain not in V.CHAINS or not V.scoreable(chain, lang):
            continue
        arm = item["amp_a1.0"]
        text = arm.get("txt", "")
        if not (arm.get("entity") or V.content(text, chain)):
            continue
        n += 1
        if V.ilr(text, chain, lang):
            target += 1
        elif arm.get("out") == "en":
            english += 1
    return n, target, english, n - target - english


def cells(n, t, e, o):
    pct = lambda k: f"{round(100 * k / n)}\\%" if n else "--"
    return f"{n} & {t} ({pct(t)}) & {e} ({pct(e)}) & {o} ({pct(o)})"


def main():
    out = {}
    for model, label in MODELS:
        for k, lang in enumerate(LANGS):
            row = [label if k == 0 else "", lang]
            for pool in ("general", "specific"):
                if (model, pool, lang) in FAILED:
                    row.append("\\multicolumn{4}{c}{Did not pass held-out gate}")
                else:
                    counts = composition(model, pool, lang)
                    out[(model, pool, lang)] = counts
                    row.append(cells(*counts))
            print(" & ".join(row) + " \\\\")
        print("\\addlinespace")
    return out


if __name__ == "__main__":
    main()
