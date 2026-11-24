"""Model loading (up to 70B, sharded across GPUs) behind one `LoadedModel` interface.

The rest of the codebase talks to a `LoadedModel`, never to nnsight / TransformerLens / HF
directly, so the framework choice (configs/models.yaml: framework) stays swappable.

The implemented path is `hf_hooks`: raw HuggingFace models plus torch forward hooks, which cover
all reads and interventions in this package. If the config asks for nnsight but it is not
installed, load_model falls back to hf_hooks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Any

import torch


@dataclass
class ModelSpec:
    name: str
    hf_id: str
    local_path: str | None
    dtype: str
    device_map: str
    n_layers: int
    n_heads: int            # query heads
    n_kv_heads: int         # grouped-query attention: KV heads < query heads
    d_model: int
    head_dim: int | None = None   # Qwen3 DECOUPLES head_dim from d_model/n_heads; read from config
    # Architecture dispatch keys: hooks need the architecture family to read per-sublayer
    # writes / routers correctly.
    arch: str = "dense"                 # dense | moe | hybrid
    attn_layout: str = "standard"       # standard | gemma_local_global | qwen35_hybrid (24 DeltaNet + 8 attn)
    residual_recipe: str = "standard"   # standard (pre-norm: submodule out == additive write)
    #                                     | gemma_postnorm (post-norm: write = h_{L+1} - h_L, NOT submodule out)

    @property
    def d_head(self) -> int:
        # Qwen3 sets an explicit head_dim (e.g. 128) that need NOT equal d_model // n_heads, so the
        # o_proj input width is n_heads * head_dim, not d_model. Llama / Qwen2.5 have head_dim == None
        # or head_dim == d_model // n_heads, so the fallback is exact for them.
        # This is a model-level fallback. Gemma-4 has heterogeneous attention
        # widths; head-level hooks must use hooks.attention_head_dim(layer, ...).
        return self.head_dim if self.head_dim else self.d_model // self.n_heads


@dataclass
class LoadedModel:
    spec: ModelSpec
    model: Any          # framework-native handle (nnsight wrapper / HookedTransformer / HF)
    tokenizer: Any
    framework: str

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _resolve_device_map(device_map: str) -> Any:
    """Pick a device placement for the current machine.

    cuda -> honor the configured device_map (accelerate shards 70B across GPUs).
    apple silicon -> a single mps device. cpu-only -> a single cpu device. The same code thus runs
    small models locally and shards a 70B model on a multi-GPU machine.
    """
    forced = os.environ.get("BEACON_DEVICE")          # cpu | mps override; 'cuda' falls through to sharding
    if forced and forced not in ("cuda", ""):
        return {"": forced}
    if torch.cuda.is_available():
        return device_map
    if torch.backends.mps.is_available():
        return {"": "mps"}
    return {"": "cpu"}


def _spec_from_config(name: str, configs) -> ModelSpec:
    m = configs.models["models"][name]
    return ModelSpec(
        name=name,
        hf_id=m["hf_id"],
        local_path=m.get("local_path"),
        dtype=m.get("dtype", "bfloat16"),
        device_map=m.get("device_map", "auto"),
        n_layers=m["n_layers"],
        n_heads=m["n_heads"],
        n_kv_heads=m["n_kv_heads"],
        d_model=m["d_model"],
        head_dim=m.get("head_dim"),
        arch=m.get("arch", "dense"),
        attn_layout=m.get("attn_layout", "standard"),
        residual_recipe=m.get("residual_recipe", "standard"),
    )


def load_model(
    name: str,
    configs,
    framework: str | None = None,
    attn_implementation: str = "eager",
    revision: str | None = None,
    tokenizer: Any | None = None,
) -> LoadedModel:
    """Load one model by config key (e.g. 'llama31_70'), sharded per device_map.

    Honors configs.models['framework'] unless `framework` overrides it. 'hf_hooks' is the implemented
    path; 'nnsight' falls back to hf_hooks if the package is absent.
    Supply revision to pin the Hub checkpoint and tokenizer to reuse a preflighted instance.

    attn_implementation defaults to 'eager' so the (B, H, S, S) attention probabilities exist for
    the qk capture (see hooks.py); pass 'sdpa' for the ov_norm / projection-only path if you want
    speed and do not need attention weights. Off CUDA, bfloat16 is promoted to float32 because MPS
    bf16 coverage is incomplete; the per-head OV math is dtype-agnostic.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    framework = framework or configs.models.get("framework", "hf_hooks")
    if framework == "nnsight":
        try:
            import nnsight  # noqa: F401
        except ImportError:
            framework = "hf_hooks"

    spec = _spec_from_config(name, configs)
    src = spec.local_path or spec.hf_id
    dtype = _DTYPES.get(spec.dtype, torch.float32)
    if not torch.cuda.is_available() and dtype is torch.bfloat16:
        dtype = torch.float32

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(src, revision=revision)
    dm = _resolve_device_map(spec.device_map)
    import transformers as _tf
    _dtype_key = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
    load_kwargs = {
        _dtype_key: dtype,
        "device_map": dm,
        "attn_implementation": attn_implementation,
    }
    if revision is not None:
        load_kwargs["revision"] = revision
    # device_map="auto" on a multi-GPU machine otherwise silently offloads most of a 70B to CPU RAM (the layers
    # then run on CPU -> 10-100x slower / apparent hang). Pin an explicit per-GPU budget and forbid CPU so
    # accelerate SHARDS across GPUs (fail-loud if it truly won't fit, which is the correct signal).
    if dm == "auto" and torch.cuda.is_available():
        # BEACON_GPU_MEM_FRACTION: some Transformers loaders transiently exceed the per-GPU budget while
        # materializing weights (copies are double-buffered); lower the fraction (e.g. ~0.80) if loading OOMs.
        frac = float(os.environ.get("BEACON_GPU_MEM_FRACTION", "0.92"))
        mm = {i: f"{int(torch.cuda.get_device_properties(i).total_memory / (1024 ** 3) * frac)}GiB"
              for i in range(torch.cuda.device_count())}
        mm["cpu"] = "0GiB"
        load_kwargs["max_memory"] = mm
    try:
        model = AutoModelForCausalLM.from_pretrained(src, **load_kwargs)
    except (ValueError, KeyError):
        # Multimodal checkpoints (Gemma 3 4B+/27B) are not registered for CausalLM. Load the
        # image-text-to-text wrapper; hooks._text_model reaches its text decoder, and a text-only
        # forward (input_ids, no pixel_values) drives just the language model.
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(src, **load_kwargs)
    model.eval()

    # Trust the live model config for head/layer dims: a loaded checkpoint may differ from the
    # numbers in models.yaml, and every per-head slice depends on these being exact. Multimodal
    # configs nest the text dims under .text_config, so read from there when present.
    cfg = model.config
    tcfg = getattr(cfg, "text_config", cfg)
    from beacon.config_metadata import nominal_dimensions
    # Scalar head metadata is nominal only. Attention hooks resolve actual
    # per-layer dimensions from the layer; do not disable Transformers' guard.
    spec = replace(spec, **nominal_dimensions(tcfg, {
        key: getattr(spec, key) for key in ('n_layers', 'n_heads', 'n_kv_heads', 'd_model', 'head_dim')}))
    return LoadedModel(spec=spec, model=model, tokenizer=tokenizer, framework=framework)


def answer_colon_position(loaded: LoadedModel, prompt: str) -> int:
    """Token index of the final ':' in 'Answer:' --- the readout position for raw base-model prompts.

    The token is ASCII in every language, so this position is language-invariant. We char-locate the
    last ':' via the offset mapping, then return the token that covers it. This is robust to ':'
    merging into a neighboring subword (e.g. an "Answer:" single token), which a naive id-equality
    search would miss.
    """
    enc = loaded.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]
    colon_char = prompt.rfind(":")
    if colon_char < 0:
        raise ValueError("prompt has no ':' to read at (expected an 'Answer:' site)")
    for i, (a, b) in enumerate(offsets):
        if b > a and a <= colon_char < b:
            return i
    # ':' fell in a zero-width / special slot; fall back to the last real token.
    return max(i for i, (a, b) in enumerate(offsets) if b > a)
