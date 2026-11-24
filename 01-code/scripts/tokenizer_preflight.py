#!/usr/bin/env python3
"""CPU tokenizer/configuration preflight. No model weights or generation are loaded.

Resolves the model revision to an immutable commit, snapshots the allowlisted tokenizer/config files, and
records tokenizer identity, special tokens, marker token IDs, prompt encodings and the effective generation
configuration. The resulting report.json is consumed by the experiment scripts via --tokenizer-pilot / --pilot.
--dry-run requires only Python's standard library and makes no network calls.
A full run needs the same transformers/tokenizers environment as the model runs,
plus huggingface_hub. Only allowlisted tokenizer/config files are downloaded.
"""
from __future__ import annotations

import argparse
import fnmatch
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
sys.path.insert(0, str(CODE / "src"))
from beacon.data.tokenizer_checks import (
    TokenizerCheckError, check_prompt, check_special_tokens, digest, file_digest, marker_ids, tokenizer_identity,
)

ALLOW = ["config.json", "generation_config.json", "tokenizer*.json", "tokenizer.model",
         "special_tokens_map.json", "added_tokens.json", "vocab.json", "vocab.txt", "merges.txt",
         "spiece.model", "sentencepiece.model", "sentencepiece.bpe.model", "*.tiktoken",
         "chat_template.jinja", "chat_templates/*.jinja"]


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prompt_cases(chains, pairs):
    for chain, meta in chains.items():
        for lang in ("en", "es", "ko", "ja", "zh"):
            if meta["prompts"].get(lang):
                yield dict(source="chains", key=chain, language=lang, text=meta["prompts"][lang])
    for i, pair in enumerate(pairs):
        for lang in ("en", "es", "ko", "ja", "zh"):
            if pair.get(lang):
                yield dict(source="discovery", key=str(i), language=lang, text=pair[lang])


def run_checks(tokenizer, model_config, generation_config, frozen, chains, pairs, chat, output):
    """Return explicit checks; write token-level artifacts without touching a model."""
    checks, warnings = [], []
    identity = tokenizer_identity(tokenizer)
    save(output / "tokenizer_identity.json", identity)
    specials = check_special_tokens(tokenizer, model_config, generation_config)
    warnings.extend(specials["warnings"])
    save(output / "special_tokens.json", specials)
    save(output / "effective_generation_config.json", generation_config)
    markers = marker_ids(tokenizer)
    save(output / "marker_token_ids.json", markers)
    checks.append(dict(name="historical_marker_counts", passed={k: len(v) for k, v in markers.items()}
                       == frozen["marker_counts"], expected=frozen["marker_counts"],
                       actual={k: len(v) for k, v in markers.items()}))
    warnings.append("Historical marker IDs/tokenizer snapshot were not saved; matching counts do not prove identity")
    count = 0
    prompt_failures = []
    with (output / "prompt_tokens.jsonl").open("w", encoding="utf-8") as fh:
        for case in prompt_cases(chains, pairs):
            count += 1
            try:
                result = check_prompt(tokenizer, case["text"], chat, specials["model_vocab_size"])
                if result["thinking_argument"] == "unsupported_argument_fallback":
                    warnings.append("enable_thinking=False unsupported; inspect rendered prompts for thinking behavior")
                row = dict(**case, **result, passed=True)
            except (ValueError, TypeError, KeyError) as exc:
                row = dict(**case, passed=False, error=f"{type(exc).__name__}: {exc}")
                prompt_failures.append(dict(source=case["source"], key=case["key"], language=case["language"]))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    checks.append(dict(name="prompt_encoding", passed=count > 0 and not prompt_failures,
                       n_prompts=count, failures=prompt_failures))

    # Decode probes are diagnostics: normalized spelling differences are not automatically bugs.
    probes = 0
    changed = 0
    unknown = 0
    with (output / "answer_roundtrips.jsonl").open("w", encoding="utf-8") as fh:
        for chain, meta in chains.items():
            for lang, label in meta["answer_label_by_lang"].items():
                if not label:
                    continue
                ids = tokenizer(label, add_special_tokens=False)["input_ids"]
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                has_unk = getattr(tokenizer, "unk_token_id", None) in ids
                probes += 1
                changed += decoded != label
                unknown += has_unk
                fh.write(json.dumps(dict(chain=chain, language=lang, label=label, token_ids=ids,
                                         decoded=decoded, exact_roundtrip=decoded == label,
                                         contains_unk=has_unk), ensure_ascii=False) + "\n")
    if changed:
        warnings.append(f"{changed}/{probes} answer labels change on roundtrip; inspect normalization, not automatic failure")
    if unknown:
        warnings.append(f"{unknown}/{probes} answer labels contain an unknown token")
    # Boundary smoke test demonstrates EOS stripping; it is NOT a model stopping test.
    first_label = next(label for meta in chains.values() for label in meta["answer_label_by_lang"].values() if label)
    label_ids = tokenizer(first_label, add_special_tokens=False)["input_ids"]
    decode_probes = []
    for eos in specials["generation_eos_ids"]:
        ids = label_ids + [eos]
        decode_probes.append(dict(input_ids=ids, appended_generation_eos=eos,
                                  keep_specials=tokenizer.decode(ids, skip_special_tokens=False),
                                  skip_specials=tokenizer.decode(ids, skip_special_tokens=True)))
        if eos not in set(tokenizer.all_special_ids):
            warnings.append(f"Generation EOS {eos} is not marked special by tokenizer; inspect scoring-boundary output")
    save(output / "decode_boundary_probes.json", decode_probes)
    return dict(status="PASS" if all(c["passed"] for c in checks) else "FAIL",
                checks=checks, warnings=list(dict.fromkeys(warnings)),
                identity_sha256=identity["sha256"], marker_ids_sha256=digest(markers),
                n_prompts=count, n_answer_probes=probes)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--repo-id", required=True, help="HF model repo, checked against GPU config before use")
    parser.add_argument("--revision", default="main", help="Resolved once to an immutable commit before download")
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, required=True)
    parser.add_argument("--pairs", type=Path, default=ROOT / "02-data/inputs/parallel_pairs.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true", help="Requires a full commit SHA already cached")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.local_files_only and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("Offline mode requires a full immutable commit SHA")
    frozen = json.loads(args.frozen_run.read_text(encoding="utf-8"))
    chains_path = ROOT / "02-data/inputs/chain_lang_prompts.json"
    chains = json.loads(chains_path.read_text(encoding="utf-8"))
    pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
    inputs = {"frozen_run": args.frozen_run, "chains": chains_path, "pairs": args.pairs,
              "prompting": CODE / "src/beacon/data/prompting.py",
              "tokenizer_checks": CODE / "src/beacon/data/tokenizer_checks.py",
              "neuron_discovery": CODE / "scripts/language_neurons.py"}
    report = dict(schema_version=1, status="PLANNED", model=frozen["model"],
                  repo_id=args.repo_id, requested_revision=args.revision, chat=args.chat,
                  scope="tokenizer_and_prompt_configuration_only", model_weights_loaded=False,
                  generation_tested=False, historical_tokenizer_identity="unavailable_no_saved_snapshot",
                  input_sha256={key: file_digest(path) for key, path in inputs.items()},
                  download_allowlist=ALLOW, prompt_count=sum(1 for _ in prompt_cases(chains, pairs)))
    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.out.exists():
        parser.error("Output exists; use a new pilot directory")
    args.out.mkdir(parents=True)
    try:
        from huggingface_hub import HfApi, snapshot_download
        from transformers import AutoConfig, AutoTokenizer, GenerationConfig
        revision = (args.revision if re.fullmatch(r"[0-9a-f]{40}", args.revision)
                    else HfApi().model_info(args.repo_id, revision=args.revision).sha)
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Could not resolve an immutable model revision")
        report["resolved_revision"] = revision
        report["versions"] = {name: importlib.metadata.version(name)
                              for name in ("transformers", "tokenizers", "huggingface_hub")}
        source = Path(snapshot_download(args.repo_id, revision=revision, allow_patterns=ALLOW,
                                        local_files_only=args.local_files_only))
        # Copy only allowlisted files; never modify the Hub cache or copy weights.
        snapshot = args.out / "snapshot"
        snapshot.mkdir()
        for path in source.rglob("*"):
            relative = path.relative_to(source).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(relative, pattern) for pattern in ALLOW):
                target = snapshot / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
        tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        config = AutoConfig.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
        gen = (GenerationConfig.from_pretrained(snapshot, local_files_only=True)
               if (snapshot / "generation_config.json").exists() else GenerationConfig.from_model_config(config))
        report.update(run_checks(tokenizer, config.to_dict(), gen.to_dict(), frozen,
                                 chains, pairs, args.chat, args.out))
        report["artifact_sha256"] = {
            p.relative_to(args.out).as_posix(): file_digest(p)
            for p in sorted(args.out.rglob("*")) if p.is_file() and p.name != "report.json"}
    except Exception as exc:
        report.update(status="FAIL" if isinstance(exc, TokenizerCheckError) else "UNAVAILABLE",
                      error_type=type(exc).__name__, error=str(exc))
        save(args.out / "report.json", report)
        print(f"Tokenizer pilot {report['status']}: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
    save(args.out / "report.json", report)
    print(f"Tokenizer pilot {report['status']}: {args.out / 'report.json'}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
