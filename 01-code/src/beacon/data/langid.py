"""Script-gated output-language attribution for decoded answers.

The alias union is byte-identical across all 5 languages (verified 466/466 chains), so it CANNOT attribute an
output to a language --- attribution must match answer_label_by_lang[L] ONLY, gated by script. This decides
which LANGUAGE a decoded answer is in (separating a language switch from content transfer): Hangul => ko
(466/466 ko labels carry Hangul), kana => ja, Han-only => zh (or the 10 Han-only ja labels -> zh_ja_amb),
Latin => es/en via EXACT diacritic-folded token match (no 5-char-prefix relaxation, so dolar!=dollar,
franco!=franc, ruandes!=ruandan stay distinct). Ambiguous buckets are excluded from the language contrast.
"""
from __future__ import annotations

import re

from beacon.data.evaluate import _normalize

def _has_hangul(s: str) -> bool:
    return any("가" <= c <= "힣" for c in s)

def _has_kana(s: str) -> bool:
    return any(("぀" <= c <= "ゟ") or ("゠" <= c <= "ヿ") or c == "ー" for c in s)

def _has_han(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)

def _latin_toks(s: str) -> list[str]:
    """Diacritic-FOLDED, lowercased, punctuation-split tokens (via _normalize) --- exact-equality units."""
    return _normalize(s).split()

def _cjk_hit(label: str, pred: str) -> bool:
    ln = _normalize(label or "")
    return bool(ln) and ln in _normalize(pred)

def _latin_exact_run(pred_toks: list[str], label: str) -> bool:
    lt = _latin_toks(label or "")
    n = len(lt)
    if not n:
        return False
    return any(pred_toks[i:i + n] == lt for i in range(len(pred_toks) - n + 1))

def output_language(pred: str, label_by_lang: dict) -> str:
    """Attribute the decoded answer to one of en/es/ko/zh/ja, or none / en_es_amb / zh_ja_amb / code_switch.
    Matches answer_label_by_lang[L] ONLY (never the cross-lingual alias union), gated by script."""
    s = pred.strip().splitlines()[0] if pred.strip() else ""
    if not s:
        return "none"
    matched: set[str] = set()
    if _has_hangul(s) and _cjk_hit(label_by_lang.get("ko"), s):
        matched.add("ko")
    if _has_kana(s) and _cjk_hit(label_by_lang.get("ja"), s):
        matched.add("ja")
    if _has_han(s) and not _has_kana(s) and not _has_hangul(s):

        if _cjk_hit(label_by_lang.get("zh"), s):
            matched.add("zh")
        if _cjk_hit(label_by_lang.get("ja"), s):
            matched.add("ja")
    if re.search("[A-Za-z]", s):
        toks = _latin_toks(s)
        if _latin_exact_run(toks, label_by_lang.get("es")):
            matched.add("es")
        if _latin_exact_run(toks, label_by_lang.get("en")):
            matched.add("en")
    if not matched:
        return "none"
    if len(matched) == 1:
        return next(iter(matched))
    if matched == {"es", "en"}:
        return "en_es_amb"
    if matched == {"zh", "ja"}:
        return "zh_ja_amb"

    cjk = matched & {"ko", "zh", "ja"}
    lat = matched & {"es", "en"}
    if cjk and lat:
        return "code_switch"
    return "zh_ja_amb" if matched <= {"zh", "ja"} else "code_switch"
