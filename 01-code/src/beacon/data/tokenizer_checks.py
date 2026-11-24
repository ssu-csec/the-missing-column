"""Tokenizer preflight primitives: prompt rendering, token-ID checks, tokenizer identity hashes,
and verification of preflight artifacts. No torch, transformers, or network imports."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path


class TokenizerCheckError(ValueError):
    pass


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_prompt(tokenizer, prompt, chat):
    """Render a prompt exactly as the experiments format it; never silently fall back to raw input."""
    if not chat:
        return dict(text=prompt, add_special_tokens=True, thinking_argument="not_applicable")
    if not getattr(tokenizer, "chat_template", None):
        raise TokenizerCheckError("chat requested but tokenizer has no chat template")
    messages = [{"role": "user", "content": prompt.rsplit("\nAnswer:", 1)[0].strip()}]
    kwargs = dict(add_generation_prompt=True, tokenize=False)
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        thinking = "requested_false"  # Acceptance does not prove that a template uses this flag.
    except TypeError:
        text = tokenizer.apply_chat_template(messages, **kwargs)
        thinking = "unsupported_argument_fallback"
    # ValueError is not swallowed: a broken template must not silently change the experiment.
    if not isinstance(text, str) or not text:
        raise TokenizerCheckError("Chat template produced no text")
    return dict(text=text, add_special_tokens=False, thinking_argument=thinking)


def token_ids(tokenizer, rendered):
    ids = tokenizer(rendered["text"], add_special_tokens=rendered["add_special_tokens"])["input_ids"]
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise TokenizerCheckError("Expected a nonempty single-prompt list of integer token IDs")
    return ids


def check_prompt(tokenizer, prompt, chat, model_vocab_size):
    rendered = render_prompt(tokenizer, prompt, chat)
    ids = token_ids(tokenizer, rendered)
    if any(i < 0 or i >= model_vocab_size for i in ids):
        raise TokenizerCheckError("Prompt token ID is outside the model embedding vocabulary")
    bos = getattr(tokenizer, "bos_token_id", None)
    if bos is not None and len(ids) > 1 and ids[:2] == [bos, bos]:
        raise TokenizerCheckError("Duplicated leading BOS tokens")
    direct_type = None
    if chat:
        messages = [{"role": "user", "content": prompt.rsplit("\nAnswer:", 1)[0].strip()}]
        kwargs = dict(add_generation_prompt=True, tokenize=True)
        if rendered["thinking_argument"] == "requested_false":
            kwargs["enable_thinking"] = False
        direct = tokenizer.apply_chat_template(messages, **kwargs)
        direct_type = type(direct).__name__
        # Depending on the Transformers version, tokenize=True returns a mapping or a plain list.
        if isinstance(direct, Mapping):
            direct = direct["input_ids"]
        if direct != ids:
            raise TokenizerCheckError("Template tokenize=True differs from render + no-added-specials")
    all_special = set(getattr(tokenizer, "all_special_ids", []))
    return dict(rendered_text=rendered["text"], input_ids=ids, n_tokens=len(ids),
                input_ids_sha256=digest(ids), add_special_tokens=rendered["add_special_tokens"],
                thinking_argument=rendered["thinking_argument"],
                direct_template_output_type=direct_type,
                special_token_positions=[dict(position=j, token_id=i) for j, i in enumerate(ids) if i in all_special],
                decoded_with_specials=tokenizer.decode(ids, skip_special_tokens=False),
                decoded_without_specials=tokenizer.decode(ids, skip_special_tokens=True))


def tokenizer_identity(tokenizer):
    vocab = tokenizer.get_vocab()
    added = tokenizer.get_added_vocab()
    special = {k: str(v) if not isinstance(v, (list, tuple)) else [str(x) for x in v]
               for k, v in tokenizer.special_tokens_map.items()}
    backend = getattr(tokenizer, "backend_tokenizer", None)
    backend_config = json.loads(backend.to_str()) if backend is not None else None
    if backend_config is not None:
        # Padding/truncation is per-call mutable state, not a vocabulary/normalizer change.
        backend_config.pop("padding", None)
        backend_config.pop("truncation", None)
    details = dict(tokenizer_class=type(tokenizer).__name__, vocab_size=tokenizer.vocab_size,
                   length=len(tokenizer), vocab_sha256=digest(vocab), added_vocab=added,
                   special_tokens_map=special, special_token_ids={name: getattr(tokenizer, name, None)
                       for name in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id")},
                   chat_template=getattr(tokenizer, "chat_template", None),
                   backend_sha256=digest(backend_config) if backend_config is not None else None,
                   padding_side=getattr(tokenizer, "padding_side", None),
                   truncation_side=getattr(tokenizer, "truncation_side", None),
                   clean_up_tokenization_spaces=getattr(tokenizer, "clean_up_tokenization_spaces", None))
    return dict(details=details, sha256=digest(details))


def marker_ids(tokenizer, cap=4000):
    """Exact vocabulary range/script order of language_neurons.py (excludes added IDs)."""
    ids = {lang: [] for lang in ("ko", "ja", "zh", "es")}
    for i in range(tokenizer.vocab_size):
        token = tokenizer.convert_ids_to_tokens(i)
        if token is None:
            continue
        try:
            text = tokenizer.convert_tokens_to_string([token])
        except Exception:
            text = token  # Same fallback as marker discovery in language_neurons.py.
        bucket = None
        for char in text:
            if "가" <= char <= "힣":
                bucket = "ko"
            elif "぀" <= char <= "ヿ":
                bucket = "ja"
            elif "一" <= char <= "鿿":
                bucket = "zh"
            elif char in "ñáéíóúü¿¡ÑÁÉÍÓÚÜ":
                bucket = "es"
            else:
                continue
            break
        if bucket is not None and len(ids[bucket]) < cap:
            ids[bucket].append(i)
    return ids


def check_special_tokens(tokenizer, model_config, generation_config):
    text_config = model_config.get("text_config", model_config)
    size = text_config.get("vocab_size", model_config.get("vocab_size"))
    if type(size) is not int or size <= 0:
        raise TokenizerCheckError("Cannot determine model text vocabulary size")
    if any(type(i) is not int or i < 0 or i >= size for i in tokenizer.get_vocab().values()):
        raise TokenizerCheckError("Tokenizer vocabulary contains IDs outside model text vocabulary")
    eos = generation_config.get("eos_token_id")
    eos = eos if isinstance(eos, list) else [eos]
    if not eos or any(type(i) is not int or not 0 <= i < size for i in eos):
        raise TokenizerCheckError("Generation EOS is missing or outside the model vocabulary")
    tok_eos = getattr(tokenizer, "eos_token_id", None)
    if type(tok_eos) is not int or not 0 <= tok_eos < size:
        raise TokenizerCheckError("Tokenizer EOS (the runner's pad override) is invalid")
    warnings = []
    if tok_eos not in eos:
        warnings.append("Tokenizer EOS differs from generation stop IDs; not automatically an error")
    return dict(model_vocab_size=size, generation_eos_ids=eos,
                generation_eos_tokens=tokenizer.convert_ids_to_tokens(eos),
                tokenizer_eos_id=tok_eos, effective_pad_token_id=tok_eos, warnings=warnings)


def validate_pilot_files(report_path):
    """Fail closed on a failed or modified tokenizer-preflight report before any generation."""
    path = Path(report_path)
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS" or report.get("schema_version") != 1:
        raise TokenizerCheckError("Tokenizer pilot did not pass")
    required = {"snapshot/config.json", "tokenizer_identity.json", "special_tokens.json",
                "effective_generation_config.json", "marker_token_ids.json", "prompt_tokens.jsonl",
                "answer_roundtrips.jsonl", "decode_boundary_probes.json"}
    if not required.issubset(report.get("artifact_sha256", {})):
        raise TokenizerCheckError("Tokenizer pilot artifacts are incomplete")
    for relative, expected in report["artifact_sha256"].items():
        part = Path(relative)
        if part.is_absolute() or ".." in part.parts:
            raise TokenizerCheckError("Invalid pilot artifact path")
        artifact = path.parent / part
        if not artifact.is_file() or file_digest(artifact) != expected:
            raise TokenizerCheckError(f"Pilot artifact missing or changed: {relative}")
    return report
