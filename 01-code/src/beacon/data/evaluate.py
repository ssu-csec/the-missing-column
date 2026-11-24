"""Run models over every item and mark correctness per language.

Writes eval.prediction / eval.correct / eval.outcome per item (each item is one chain x lang x variant).
Correctness is gold-entity match, not bare surface match: the gold label (plus any aliases carried on the
item) is normalized and tested against the greedy continuation after the "Answer:" colon. subsets.py
then reads the per-(chain, lang, variant) flags to build the evaluation subsets.
"""
from __future__ import annotations

import re
import unicodedata

import torch

from beacon.data.prompting import encode_batch


_ARTICLES = {"the", "a", "an", "el", "la", "los", "las", "un", "una"}


def _fold(text: str) -> str:
    """Drop Latin diacritics (Türkiye->Turkiye, ruandés->ruandes) while leaving CJK/Hangul intact.

    NFKD splits accents into combining marks (and Hangul syllables into conjoining jamo); we drop only the
    combining marks (jamo are not combining), then NFC recomposes the jamo back, so Korean round-trips."""
    t = unicodedata.normalize("NFKD", text)
    t = "".join(c for c in t if not unicodedata.combining(c))
    return unicodedata.normalize("NFC", t)


def _normalize(text: str) -> str:
    """Lowercase, fold diacritics, drop punctuation and a leading article --- for lenient entity match."""
    text = text.strip().splitlines()[0] if text.strip() else ""
    text = re.sub(r"[^\w\s]", " ", _fold(text).lower(), flags=re.UNICODE)
    toks = [t for t in text.split() if t]
    if toks and toks[0] in _ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


def _has_cjk(s: str) -> bool:
    """True if the string holds any CJK / kana / hangul codepoint (scripts written without spaces)."""
    return any("぀" <= c <= "鿿" or "가" <= c <= "힣" for c in s)


def _tok_eq(a: str, b: str) -> bool:
    """Two Latin tokens match if identical, or share a 5-char prefix when both are >=6 chars. The prefix
    rule absorbs the country-adjective gap (rwanda/rwandan, guyana/guyanese, hungary/hungarian) without
    firing on short abbreviations (uk, us, rf), which stay exact-match only."""
    return a == b or (len(a) >= 6 and len(b) >= 6 and a[:5] == b[:5])


def _latin_run(pred_toks, gold: str) -> bool:
    """gold (already normalized) matches as a WHOLE token run in pred_toks, token-wise via _tok_eq."""
    gt = gold.split()
    n = len(gt)
    return bool(n) and any(
        all(_tok_eq(pred_toks[i + j], gt[j]) for j in range(n))
        for i in range(len(pred_toks) - n + 1))


def _form_matches(pred_norm: str, pred_toks, form: str) -> bool:
    """One surface form matches, by script: CJK as a substring (no word boundaries), Latin as a robust
    whole-token run. So "uk" still matches "uk" but NOT "ukraine" (len<6, exact only)."""
    fn = _normalize(form or "")
    if not fn:
        return False
    return (fn in pred_norm) if _has_cjk(fn) else _latin_run(pred_toks, fn)


def entity_match(prediction: str, gold_label: str, aliases=None) -> bool:
    """True if the prediction names the gold entity in ANY language (gold label or cross-lingual alias)."""
    pred = _normalize(prediction)
    if not pred:
        return False
    toks = pred.split()
    return any(_form_matches(pred, toks, g) for g in [gold_label, *(aliases or [])])


def entity_outcome(prediction: str, item: dict) -> str:
    """Three-way cross-lingual outcome for one item (drives the content-match/answer-form/capability split):
      x_correct    --- gold entity in the TARGET-language surface (content match AND designated answer-form realized);
      content_only --- gold entity present, but only in a non-target-language surface (typically English):
                       content recovered, designated answer-form not realized;
      wrong        --- gold entity in no language.
    content_only is split out only for CJK targets (ko/zh/ja), where an English answer is unambiguous by
    script; for a Latin target (es) any entity match counts as x_correct (es stays in Spanish in practice).
    """
    pred = _normalize(prediction)
    if not pred:
        return "wrong"
    toks = pred.split()
    forms = [item.get("answer_label"), *(item.get("answer_aliases") or [])]
    if item.get("lang") in ("ko", "zh", "ja"):
        cjk = [f for f in forms if f and _has_cjk(f)]
        latin = [f for f in forms if f and not _has_cjk(f)]
        if any(_form_matches(pred, toks, f) for f in cjk):
            return "x_correct"
        if any(_form_matches(pred, toks, f) for f in latin):
            return "content_only"
        return "wrong"
    return "x_correct" if any(_form_matches(pred, toks, f) for f in forms if f) else "wrong"


@torch.no_grad()
def evaluate_items(loaded, items, max_new_tokens: int = 32, batch_report: int = 0,
                   batch_size: int = 16, chat: bool = False) -> list[dict]:
    """Greedy-decode prompts in BATCHES, map each continuation to the gold entity, set item['eval'].

    Batched generation is the difference between a tractable and an intractable 70B eval (~batch_size x
    fewer forward passes than one-item-at-a-time). Greedy decoding is deterministic, so batching matches
    the per-item result as long as we LEFT-pad (continuations then align at the same offset) and pass the
    attention mask. Progress prints flush so a redirected log shows progress live (no block buffering).
    """
    tok = loaded.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    orig_side = tok.padding_side
    tok.padding_side = "left"                       # left-pad: every continuation starts at in_len
    try:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            enc = encode_batch(loaded, [it["prompt"] for it in batch], chat)
            gen = loaded.model.generate(
                **enc, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
            in_len = enc["input_ids"].shape[1]
            for j, it in enumerate(batch):
                cont = tok.decode(gen[j, in_len:], skip_special_tokens=True)
                if cont.lstrip().startswith("<think>"):
                    raise RuntimeError(
                        "thinking-mode output detected; Qwen evaluation must use --chat with "
                        "enable_thinking=False"
                    )
                outcome = entity_outcome(cont, it)               # x_correct | content_only | wrong
                it["eval"] = {"prediction": cont.strip(), "correct": outcome != "wrong", "outcome": outcome}
            done = start + len(batch)
            if batch_report and (done % batch_report < batch_size or done == len(items)):
                print(f"  [eval] {done}/{len(items)}", flush=True)
    finally:
        tok.padding_side = orig_side
    return items


@torch.no_grad()
def base_two_hop_nll(loaded, items) -> float:
    """Mean teacher-forced NLL of the gold answer on CORRECT two_hop items."""
    tok = loaded.tokenizer
    nlls = []
    for it in items:
        if it["variant"] != "two_hop" or not it.get("eval", {}).get("correct"):
            continue
        prompt_ids = tok(it["prompt"], return_tensors="pt").input_ids
        # CJK/Hangul labels take no leading space: ' '+CJK tokenizes to a broken UTF-8 byte-fragment
        # first token (same boundary rule as pipeline.patching.first_answer_token_id), inflating NLL.
        lab = it["answer_label"]
        lead = "" if lab and ("぀" <= lab[0] <= "鿿" or "가" <= lab[0] <= "힣") else " "
        answer_ids = tok(lead + lab, add_special_tokens=False, return_tensors="pt").input_ids
        full = torch.cat([prompt_ids, answer_ids], dim=1).to(loaded.device)
        logits = loaded.model(full).logits
        # score only the answer tokens: logits at position t predict token t+1
        ans_start = prompt_ids.shape[1]
        logp = torch.log_softmax(logits[0, ans_start - 1:-1].float(), dim=-1)
        gold = full[0, ans_start:]
        nlls.append(-logp[range(len(gold)), gold].mean().item())
    return float(sum(nlls) / len(nlls)) if nlls else float("nan")
