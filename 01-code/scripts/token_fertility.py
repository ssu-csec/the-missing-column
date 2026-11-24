"""TOKEN FERTILITY --- does a vocabulary/tokenizer gap explain the cross-lingual results?

Rules out a confound for the accessibility findings (script 46) and the intervention results
(scripts 42/43/44): if ko/ja/zh tokenize much less efficiently than en/es on these
checkpoints, that alone -- not a representational gap -- could depress target-language
delivery. Run once per model, like scripts 45/46 (model key from configs/models.yaml):
  1. corpus-level tokens/byte and byte-fallback rate for question/answer text
     (chain_lang_prompts.json) -> token_fertility_<model>.json
  2. item-level entity fragmentation vs. in-language correctness, using the
     intervention result files in 02-data/results/interventions/ -> token_entity_fragmentation_<model>.json

The related tautology check on the delivery criterion itself (does out == lang ever
co-occur with correct=False) is a separate, model-agnostic scan that needs no
tokenizer and is not run per model; it is not included.

    python scripts/token_fertility.py --model gemma4_31b
    python scripts/token_fertility.py --model llama31_70_it
    python scripts/token_fertility.py --verify
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path

from transformers import AutoTokenizer

from beacon.config import REPO_ROOT, load_configs

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

LANGS = ["en", "es", "ko", "zh", "ja"]

BYTE_FALLBACK_RE = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")

INTERVENTION_FAMILIES = ("calibrated_", "mrre_", "sunneurons_")

def load_chains(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def question_only(prompt: str) -> str:
    line = prompt.split("\n")[-2] if "\n" in prompt else prompt
    for l in prompt.split("\n"):
        if l.startswith("Question:"):
            line = l
            break
    return line.split(":", 1)[-1].strip() if ":" in line else line.strip()

def build_corpora(chains: dict) -> dict:
    corpora = {lang: {"questions": [], "answers": []} for lang in LANGS}
    for chain_id, rec in chains.items():
        prompts = rec.get("prompts", {})
        labels = rec.get("answer_label_by_lang", {})
        for lang in LANGS:
            if lang in prompts:
                corpora[lang]["questions"].append(question_only(prompts[lang]))
            if lang in labels and labels[lang]:
                corpora[lang]["answers"].append(labels[lang])
    return corpora

def char_len(text: str) -> int:
    return len(text)

def byte_len(text: str) -> int:
    return len(text.encode("utf-8"))

def fertility_for(tok, texts: list[str]) -> dict:
    texts = [t for t in texts if t]
    if not texts:
        return {"n_items": 0}
    all_ids = tok(texts, add_special_tokens=False)["input_ids"]
    n_tokens = sum(len(ids) for ids in all_ids)
    n_bytes = sum(max(byte_len(t), 1) for t in texts)
    n_items = len(texts)
    try:
        flat_ids = [i for ids in all_ids for i in ids]
        pieces = tok.convert_ids_to_tokens(flat_ids)
        n_fallback = sum(1 for p in pieces if BYTE_FALLBACK_RE.match(p))
    except Exception:
        n_fallback = 0
    return {
        "n_items": n_items,
        "tokens_per_byte": round(n_tokens / n_bytes, 4),
        "byte_fallback_rate": round(n_fallback / n_tokens, 4) if n_tokens else 0.0,
        "mean_tokens_per_item": round(n_tokens / n_items, 2),
    }

def load_tokenizer(hf_id: str):
    hf_token = os.environ.get("HF_TOKEN")
    return AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True, token=hf_token)

def _intervention_files(interventions_dir: Path, model: str):
    for path in interventions_dir.glob("*.json"):
        stem = path.stem
        for fam in INTERVENTION_FAMILIES:
            if stem.startswith(fam) and stem[len(fam):] == model:
                yield path
                break

def analyze_entity_fragmentation(tok, interventions_dir: Path, model: str, chains: dict) -> list[dict]:
    entries = []  # [(gold_str, requested_lang, out_lang, correct), ...]
    for path in _intervention_files(interventions_dir, model):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        eval_chains = d.get("eval_chains", [])
        for requested_lang, block in d.get("eval", {}).items():
            items = block.get("steer_items", []) if isinstance(block, dict) else []
            if len(items) != len(eval_chains):
                print(f"[tf] warn: {path.name}/{requested_lang}: {len(items)} items vs "
                      f"{len(eval_chains)} eval_chains -- skipping (order not guaranteed)", flush=True)
                continue
            for chain_id, item in zip(eval_chains, items):
                if not item.get("entity"):
                    continue
                out_lang = item.get("out")
                if out_lang not in LANGS:  # drop "none", "en_es_amb", "zh_ja_amb", etc.
                    continue
                gold = chains.get(chain_id, {}).get("answer_label_by_lang", {}).get(out_lang)
                if not gold:
                    continue
                entries.append((gold, requested_lang, out_lang, bool(item.get("correct", False))))

    agg = defaultdict(lambda: {"tok_sum": 0, "char_sum": 0, "n_items": 0})
    if entries:
        all_ids = tok([e[0] for e in entries], add_special_tokens=False)["input_ids"]
        for (gold, requested_lang, out_lang, correct), ids in zip(entries, all_ids):
            key = (requested_lang, out_lang, correct)
            agg[key]["tok_sum"] += len(ids)
            agg[key]["char_sum"] += max(char_len(gold), 1)
            agg[key]["n_items"] += 1

    records = []
    for (requested_lang, out_lang, correct), a in agg.items():
        records.append({
            "requested_lang": requested_lang,
            "out_lang": out_lang,
            "correct": correct,
            "n_items": a["n_items"],
            "mean_tokens_per_char": round(a["tok_sum"] / a["char_sum"], 3),
            "mean_token_count": round(a["tok_sum"] / a["n_items"], 2),
        })
    return records

def _verify():
    """$0 CPU smoke test of the pure-Python logic --- no tokenizer/model load."""
    prompt = "Some context line.\nQuestion: What is the capital of Korea?\nAnswer:"
    assert question_only(prompt) == "What is the capital of Korea?"
    assert char_len("한국은행") == 4
    assert byte_len("한국은행") == len("한국은행".encode("utf-8")) == 12
    print("[tf] VERIFY OK: question extraction, char/byte length.", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--model", help="key from configs/models.yaml, e.g. gemma4_31b")
    ap.add_argument("--data", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--out", default="02-data/results/access")
    args = ap.parse_args()
    if args.verify:
        _verify(); return
    assert args.model, "--model is required (see configs/models.yaml for valid keys)"

    cfg = load_configs()
    hf_id = cfg.models["models"][args.model]["hf_id"]

    chains = load_chains(_path(args.data))
    print(f"[tf] loaded {len(chains)} chains from {args.data}", flush=True)
    corpora = build_corpora(chains)
    for lang in LANGS:
        print(f"[tf]   {lang}: {len(corpora[lang]['questions'])} questions, "
              f"{len(corpora[lang]['answers'])} answer labels", flush=True)

    print(f"[tf] loading tokenizer: {args.model} <- {hf_id}", flush=True)
    tok = load_tokenizer(hf_id)

    out_dir = _path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_lang = {
        lang: {
            "question": fertility_for(tok, corpora[lang]["questions"]),
            "answer": fertility_for(tok, corpora[lang]["answers"]),
        }
        for lang in LANGS
    }
    json_path = out_dir / f"token_fertility_{args.model}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "by_lang": by_lang}, f, ensure_ascii=False, indent=2)
    print(f"[tf] wrote {json_path}", flush=True)

    interventions_dir = ROOT / "02-data/results/interventions"
    if not interventions_dir.is_dir():
        print(f"[tf] skip: entity-fragmentation analysis, {interventions_dir} not found", flush=True)
        return

    frag_records = analyze_entity_fragmentation(tok, interventions_dir, args.model, chains)
    frag_path = out_dir / f"token_entity_fragmentation_{args.model}.json"
    with open(frag_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "records": frag_records}, f, ensure_ascii=False, indent=2)
    print(f"[tf] wrote {frag_path}", flush=True)

if __name__ == "__main__":
    main()
