"""Per-head readout at the 'Answer:' colon, plus decoder-layer and head-dimension helpers.

Per general-pool head, at the answer-colon, in a single forward pass, capture:
  - ov_norm            : L2 norm of the head's OV-side output (its write into the residual stream)
  - bridge_projection  : projection of that output onto the gold-bridge token's unembedding direction
  - qk_attention       : attention weights over the source entity A and the relation words ONLY

================================================================================
WHERE TO HOOK (HuggingFace Llama 3.1 / Qwen 2.5; see modeling_llama.py and
modeling_qwen2.py in transformers).
================================================================================
Both models compute, inside `LlamaAttention.forward` / `Qwen2Attention.forward`:

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()   # (B, S, n_heads*d_head)
    attn_output = self.o_proj(attn_output)                             # (B, S, d_model)

Hook the INPUT to `o_proj` (a forward_pre_hook on `model.model.layers[L].self_attn.o_proj`),
NOT the attention module output (which is already summed over heads and post-o_proj). The
pre-hook input is the concatenation of all 64 query heads along the last axis, head h occupying
the CONTIGUOUS slice [h*d_head : (h+1)*d_head]. This slice convention is identical in Llama and
Qwen; Qwen adds q/k/v_proj bias but o_proj has bias=False in both, so per-head OV reconstruction
is unaffected. d_model=8192, d_head=128, n_heads=64, n_kv_heads=8.

Per-head residual write (the OV-side output we score). Let a_in be the o_proj input at the colon:
    a_in            : (d_model,)          == concat over 64 heads of (attn_h @ V_group(h))
    a_h = a_in[h*d_head:(h+1)*d_head]     : (d_head=128,)
    W_O = o_proj.weight                   : (d_model=8192, d_model=8192), maps in->out
    W_O_h = W_O[:, h*d_head:(h+1)*d_head] : (8192, 128)   per-head output columns
    o_h = W_O_h @ a_h                     : (d_model=8192,)   head h's write into the residual
Equivalently o_h = (attn_h @ V_group(h)) @ W_O[:, h*d_head:(h+1)*d_head].T with a_h row-vector.
ov_norm[h] = ||o_h||_2 . Reading a_in once and reconstructing every pool head from W_O slices is
one matmul per head and avoids 64 separate module hooks.

GQA gotcha. Grouped-query attention (64 query heads, 8 KV heads, group size n_rep=8). HF expands
KV with `repeat_kv` BEFORE the score, so attn_h is already a distinct 64-head probability row even
though V is shared within a group: V_group(h) = value_states[:, h // n_rep]. The W_O column slice
is per query head, so the per-head OV write is well defined. Do NOT assume one V per head, and do
NOT divide d_model by n_kv_heads --- the slice width is the per-head dim, 128, not 1024.

head_dim gotcha (Qwen3). Qwen3 DECOUPLES head_dim from d_model // n_heads: e.g. Qwen3-30B-A3B has
d_model=2048, n_heads=32 but head_dim=128, so the o_proj INPUT width is n_heads*head_dim=4096, NOT
d_model=2048. Always slice by spec.d_head (= config.head_dim when set, else d_model // n_heads), and
never assume the o_proj input equals d_model. Llama 3.1 / Qwen 2.5 have head_dim==d_model//n_heads==
128, so the slice is 128 there too --- but the decoupled case silently corrupts every head if you
hard-code d_model // n_heads. Note MoE (Qwen3-30B-A3B) only changes the FFN; attention o_proj and
this OV read are unaffected.

QK caveat + eager requirement. Read attention only on tokens that exist in the prompt (A, relation
words). Never "attention to the bridge token" --- the bridge B is latent and has no token to attend
to. The (B, 64, S, S) probability tensor is only materialized under attn_implementation="eager";
SDPA / FlashAttention2 never form it. Load BOTH models with attn_implementation="eager" for any
run that captures qk_attention (ov_norm and bridge_projection do not need eager).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import answer_colon_position


def _text_model(model):
    """The text decoder stack --- the object holding `.layers` and the final `.norm`.

    Plain CausalLM wrappers (Llama / Qwen 2.5 / Qwen3 / Gemma 2) keep it at model.model. Multimodal
    wrappers (Gemma 3 4B+/27B) nest it one deeper at model.model.language_model. Walk down the common
    sub-module names until the decoder `.layers` ModuleList appears, so the OV hook is wrapper-agnostic.
    """
    m = model
    for _ in range(4):
        if hasattr(m, "layers"):
            return m
        nxt = next((getattr(m, a) for a in ("model", "language_model", "text_model") if hasattr(m, a)), None)
        if nxt is None:
            break
        m = nxt
    if hasattr(m, "layers"):
        return m
    raise AttributeError(f"could not locate decoder .layers under {type(model).__name__}")


def _decoder_layers(model):
    """The decoder-block ModuleList (Llama/Qwen: model.model.layers; Gemma 3: via language_model)."""
    return _text_model(model).layers


def eligible_attention_layers(model) -> list[int]:
    """Decoder layers whose attention output has a query-head-concatenated ``o_proj`` input.

    SCH is defined only at this interface.  In hybrid models such as Qwen3.6, DeltaNet
    layers are intentionally excluded rather than being treated as attention heads.
    """
    out = []
    for index, layer in enumerate(_decoder_layers(model)):
        attn = getattr(layer, "self_attn", None)
        if attn is not None and getattr(attn, "o_proj", None) is not None:
            out.append(index)
    return out


def attention_head_dim(layer, n_heads: int) -> int:
    """Return the actual per-query-head width at this layer's ``o_proj`` input.

    Gemma-4 uses different widths for sliding and global attention
    (``head_dim`` versus ``global_head_dim``), so a model-wide config value is
    insufficient.  The projection input is the authoritative concatenated
    query-head axis for every architecture supported by the SCH hooks.
    """
    width = int(layer.self_attn.o_proj.in_features)
    if width % int(n_heads):
        raise RuntimeError(
            f"o_proj input width {width} is not divisible by {n_heads} query heads"
        )
    return width // int(n_heads)


def _final_norm(model):
    """The final pre-unembedding RMSNorm; used for the logit-lens read."""
    return getattr(_text_model(model), "norm", None)


def _output_embedding_weight(model) -> torch.Tensor:
    """Unembedding matrix W_U as (vocab, d_model); row t is token t's unembedding direction."""
    return model.get_output_embeddings().weight


@dataclass
class HeadReadout:
    """Per-item capture at the colon, indexed by general-pool head.

    bridge_projection is the unit-normalized read so it is comparable across layer bands:
        d_B = unit gold-bridge direction (see bridge_direction); o_h = head h's residual write
        bridge_projection[h] = <o_h, d_B>            (signed scalar, NOT divided by ||o_h||)
    Keep both bridge_projection and ov_norm: their ratio recovers the cosine alignment, which splits
    magnitude (ov_norm) from direction (cosine). layer_band tags each head in-band / out-of-band
    (see assign_layer_band).
    """
    layers: torch.Tensor          # (n_pool,)  layer index per pool head
    heads: torch.Tensor           # (n_pool,)  head index per pool head
    layer_band: torch.Tensor      # (n_pool,)  0=in-band, 1=out-of-band
    ov_norm: torch.Tensor         # (n_pool,)  ||o_h||_2 of the head's residual write
    bridge_projection: torch.Tensor  # (n_pool,)  <o_h, unit gold-bridge direction>, signed
    qk_to_subject: torch.Tensor   # (n_pool,)  attention mass on source entity A span (summed)
    qk_to_relation: torch.Tensor  # (n_pool,)  attention mass on relation-word span (summed)
    residual: torch.Tensor | None = None  # (d_model,) full residual stream at the colon


def bridge_direction(loaded, gold_bridge_token_ids, mode: str = "mean") -> torch.Tensor:
    """Form the unit gold-bridge unembedding direction for the projection.

    The bridge B is LATENT and may be MULTI-TOKEN, so there is no single unembedding row to use.
    Build the direction in unembedding space, then unit-normalize:
      mode="first" : d_B = W_U[:, t0]                         (first bridge token only)
      mode="mean"  : d_B = mean_k W_U[:, t_k]                 (mean over B's token unembeddings)
    where W_U = lm_head.weight.T (or tied embed_tokens.weight.T), columns indexed by token id.
    gold_bridge_token_ids are the tokenizer ids of B's gold surface form in the ITEM language; for a
    one-token B both modes coincide. Default mean: it is robust to subword splits and is the mode
    validate_projection uses. Returns d_B : (d_model,), ||d_B||_2 = 1.

    NOTE on the lens correction. For mid-layer heads o_h sits far from the unembedding, so raw W_U columns are a biased readout (logit-lens covariance
    drift). Two options, both fed by validate_projection: (a) logit-lens---apply final RMSNorm to o_h
    before the dot product; (b) tuned-lens---learn a per-layer affine A_L,b_L and project
    (A_L @ o_h + b_L) onto d_B; validate_projection reports which lens tracks the residual. The signed scalar
    stored in HeadReadout.bridge_projection is <lens(o_h), d_B> under the chosen lens.
    """
    W_U = _output_embedding_weight(loaded.model)            # (vocab, d_model); row = unembedding
    ids = torch.as_tensor(list(gold_bridge_token_ids), dtype=torch.long, device=W_U.device)
    if ids.numel() == 0:
        raise ValueError("no gold-bridge token ids given")
    if mode == "first":
        d = W_U[ids[0]].float()
    elif mode == "mean":
        d = W_U[ids].float().mean(0)
    else:
        raise ValueError(f"unknown bridge_direction mode {mode!r}")
    return d / d.norm().clamp_min(1e-8)


def _lens_read(o_h: torch.Tensor, lens: str, final_norm) -> torch.Tensor:
    """Apply the chosen lens to a single head's residual write before projecting.

    lens="raw"   : the write as-is (HeadReadout stores this by default).
    lens="logit" : final RMSNorm applied to o_h (logit lens of the component) --- corrects the
                   covariance drift of mid-layer heads relative to the unembedding.
    A tuned lens (fitted per-layer affine) is not implemented here.
    """
    if lens == "raw" or final_norm is None:
        return o_h.float()
    if lens == "logit":
        return final_norm(o_h.to(final_norm.weight.dtype)).float()
    raise ValueError(f"unknown lens {lens!r}")


@torch.no_grad()
def capture_at_colon(
    loaded,
    prompt,
    pool,
    gold_bridge_dir,
    span_index=None,
    layer_bands=None,
    lens: str = "raw",
) -> HeadReadout:
    """Run one forward pass and read the general pool at the answer-colon.

    Mechanics (HF eager). Register a forward_pre_hook on each pooled layer's self_attn.o_proj that
    grabs its INPUT a_in at the colon token; reconstruct each pool head's write o_h from the W_O
    column slice (see module docstring). Read attention from the captured (B, 64, S, S) probs at the
    colon query row. One forward pass, no backward.

    pool           : list of (layer, head) for the general head pool.
    gold_bridge_dir: unit gold-bridge direction (d_model,) from bridge_direction(); the projection is
                     <lens(o_h), gold_bridge_dir>. Pass a direction, not a token id, so the
                     multi-token / lens choice is resolved upstream once per item.
    span_index     : {"subject": (s0, s1), "relation": (r0, r1)} half-open token spans into the
                     prompt for the source entity A and the relation words (see locate_prompt_spans).
                     The colon query attends BACK over these prompt positions only. If None, qk is
                     not read (ov_norm / projection do not need it) and the qk fields are
                     left NaN.
    layer_bands    : per-model {low, high} in-band cutoff (configs/pools.yaml); assign_layer_band
                     tags each pool head 0/1 (in-band / out-of-band). If None, all heads tag 0.

    Returns one HeadReadout (n_pool heads) for this item at the colon.
    """
    model = loaded.model
    decoder_layers = _decoder_layers(model)
    final_norm = _final_norm(model)
    colon = answer_colon_position(loaded, prompt)

    # One pre-hook per UNIQUE pooled layer; grab the o_proj input (concat over query heads).
    layers_needed = sorted({int(layer) for (layer, _head) in pool})
    captured: dict[int, torch.Tensor] = {}
    handles = []

    def _make_pre_hook(layer_idx):
        def _pre_hook(_module, args):
            captured[layer_idx] = args[0].detach()      # (B, S, d_model) input to o_proj
        return _pre_hook

    for layer_idx in layers_needed:
        h = decoder_layers[layer_idx].self_attn.o_proj.register_forward_pre_hook(_make_pre_hook(layer_idx))
        handles.append(h)

    enc = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
    want_attn = span_index is not None
    try:
        with torch.no_grad():
            out = model(**enc, output_attentions=want_attn, output_hidden_states=True, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    attentions = out.attentions if want_attn else None
    # full residual stream at the colon (last layer, pre-final-norm)
    residual = out.hidden_states[-1][0, colon].float().cpu()

    n = len(pool)
    layers_t = torch.zeros(n, dtype=torch.long)
    heads_t = torch.zeros(n, dtype=torch.long)
    ov_norm = torch.zeros(n)
    proj = torch.zeros(n)
    qk_subj = torch.full((n,), float("nan"))
    qk_rel = torch.full((n,), float("nan"))
    d_B = gold_bridge_dir.float().to("cpu")

    for i, (layer_idx, head_idx) in enumerate(pool):
        layer_idx = int(layer_idx)
        head_idx = int(head_idx)
        d_head = attention_head_dim(decoder_layers[layer_idx], loaded.spec.n_heads)
        layers_t[i] = layer_idx
        heads_t[i] = head_idx
        a_in = captured[layer_idx][0, colon]                                   # (d_model,)
        a_h = a_in[head_idx * d_head:(head_idx + 1) * d_head]                  # (d_head,)
        W_O = decoder_layers[layer_idx].self_attn.o_proj.weight               # (d_out, d_in)
        W_O_h = W_O[:, head_idx * d_head:(head_idx + 1) * d_head]             # (d_model, d_head)
        o_h = (W_O_h @ a_h)                                                    # (d_model,) residual write
        ov_norm[i] = o_h.float().norm().cpu()
        o_read = _lens_read(o_h, lens, final_norm).cpu()
        proj[i] = torch.dot(o_read, d_B)
        if attentions is not None:
            row = attentions[layer_idx][0, head_idx, colon].float().cpu()      # (S,) colon -> keys
            subj = span_index.get("subject")
            rel = span_index.get("relation")
            if subj is not None:
                qk_subj[i] = row[subj[0]:subj[1]].sum()
            if rel:
                spans = rel if isinstance(rel, list) else [rel]
                qk_rel[i] = sum(row[r0:r1].sum() for (r0, r1) in spans)

    band = assign_layer_band(pool, loaded.spec.name, layer_bands) if layer_bands is not None \
        else torch.zeros(n, dtype=torch.long)
    return HeadReadout(layers_t, heads_t, band, ov_norm, proj, qk_subj, qk_rel, residual=residual)


def locate_prompt_spans(loaded, prompt, item, lang) -> dict:
    """Token spans for the source entity A and the relation words, per-language tokenization.

    QK is read ONLY on prompt tokens (A, relation words); never the latent bridge B. Method:
      1. Take the gold surface strings from the item: e1_label (entity A, kept in EN surface form
         across all languages per data.yaml) and the r1/r2 relation rendering in `lang`.
      2. Tokenize the prompt with offset_mapping=True (tokenizer(..., return_offsets_mapping=True)),
         then char-align each surface string to its token span by offset overlap. Offset alignment
         is tokenizer-agnostic, so it handles Llama BPE and Qwen BPE and CJK (ko/zh/ja) without
         whitespace assumptions.
      3. The relation may be multi-word and discontiguous in ko/ja (postpositions/particles); return
         a list of spans and let capture_at_colon sum attention mass over all of them.
    Returns {"subject": (s0, s1), "relation": [(r0, r1), ...]} as half-open token-index spans.
    Summarize attention as a SUM of probability mass over span tokens (a mass, comparable across
    span lengths), not a mean; report mean only as a per-token-normalized secondary feature.
    """
    enc = loaded.tokenizer(prompt, return_offsets_mapping=True, add_special_tokens=True)
    offsets = enc["offset_mapping"]

    def char_to_token_span(c0: int, c1: int):
        toks = [i for i, (a, b) in enumerate(offsets) if b > a and a < c1 and b > c0]
        return (min(toks), max(toks) + 1) if toks else None

    result: dict = {"subject": None, "relation": []}
    subject = item.get("e1_label") or item.get("subject")
    if subject:
        c0 = prompt.find(subject)
        if c0 >= 0:
            result["subject"] = char_to_token_span(c0, c0 + len(subject))
    for relation_word in (item.get("relation_words") or []):
        c0 = prompt.find(relation_word)
        if c0 >= 0:
            span = char_to_token_span(c0, c0 + len(relation_word))
            if span:
                result["relation"].append(span)
    return result


def assign_layer_band(pool, model_name: str, bands) -> torch.Tensor:
    """Tag each pool head in-band (0) vs out-of-band (1) by layer.

    The pool is not layer-coherent, so a single pooled attention statistic averages over heads that
    read structurally different positions; this tag lets statistics be stratified by layer band.
    Per-model cutoffs live in configs/pools.yaml.
    Returns (n_pool,) of {0,1}.

    `bands` is the in-band dict {"low": int, "high": int} for this model (inclusive bounds).
    """
    low, high = int(bands["low"]), int(bands["high"])
    out = torch.zeros(len(pool), dtype=torch.long)
    for i, (layer_idx, _head) in enumerate(pool):
        out[i] = 0 if low <= int(layer_idx) <= high else 1
    return out


@torch.no_grad()
def validate_projection(loaded, items, pool, lens: str = "logit", layer_bands=None) -> dict:
    """Validate the per-head bridge projection against a residual-level logit-lens decode.

    Protocol (known-correct EN items, where the gold bridge B is decodable):
      1. For each item, run capture_at_colon and also decode the FULL residual stream at the colon
         through the chosen lens to a bridge logit: logit_B = <lens(resid), W_U[:, bridge_ids]>.
      2. Per pool head, correlate bridge_projection across items against logit_B
         (Spearman rho); the head-level projection should track the residual-level lens signal.
      3. Criterion: mean |rho| over in-band heads (assign_layer_band == 0) >= 0.5. rho is reported
         per layer band, so a failure for mid-layer / out-of-band heads (covariance drift far from
         the unembedding) is attributable to distance from the unembedding.
    Returns {"lens": lens, "rho_in_band": float, "rho_out_band": float, "passed": bool, ...}.

    The residual-level target is the model's own next-token logit for the bridge tokens at the colon
    (lm_head(final_norm(resid))), i.e. the logit lens at the final layer. We correlate each head's
    lens-read projection against it. `items` are dicts with keys 'prompt' and 'bridge_token_ids'.
    `layer_bands` is the in-band {low, high} dict; if None, all heads count as in-band.
    """
    from scipy.stats import spearmanr

    if not items or not pool:
        return {"lens": lens, "rho_in_band": float("nan"), "rho_out_band": float("nan"),
                "passed": False, "n_items": len(items or []), "n_pool": len(pool or [])}

    proj_rows = []          # per item: (n_pool,) head projections under `lens`
    bridge_logit = []       # per item: scalar residual-level bridge logit
    for it in items:
        prompt = it["prompt"]
        bridge_ids = it["bridge_token_ids"]
        d_B = bridge_direction(loaded, bridge_ids, mode="mean")
        readout = capture_at_colon(loaded, prompt, pool, d_B, span_index=None,
                                   layer_bands=None, lens=lens)
        proj_rows.append(readout.bridge_projection)

        colon = answer_colon_position(loaded, prompt)
        enc = loaded.tokenizer(prompt, return_tensors="pt").to(loaded.device)
        with torch.no_grad():
            out = loaded.model(**enc, use_cache=False)
        logits = out.logits[0, colon].float().cpu()
        ids = torch.as_tensor(list(bridge_ids), dtype=torch.long)
        bridge_logit.append(logits[ids].mean().item())

    proj = torch.stack(proj_rows)                       # (n_items, n_pool)
    target = torch.tensor(bridge_logit)                 # (n_items,)
    band = assign_layer_band(pool, loaded.spec.name, layer_bands) if layer_bands is not None \
        else torch.zeros(len(pool), dtype=torch.long)

    rho = torch.zeros(len(pool))
    proj_np = proj.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    for h in range(proj.shape[1]):
        r, _ = spearmanr(proj_np[:, h], target_np)
        rho[h] = 0.0 if r != r else r                   # NaN (constant column) -> 0

    in_mask = band == 0
    out_mask = band == 1
    rho_in = float(rho[in_mask].abs().mean()) if in_mask.any() else float("nan")
    rho_out = float(rho[out_mask].abs().mean()) if out_mask.any() else float("nan")
    return {
        "lens": lens,
        "rho_in_band": rho_in,
        "rho_out_band": rho_out,
        "rho_in_band_best": float(rho[in_mask].abs().max()) if in_mask.any() else float("nan"),
        "passed": bool(rho_in >= 0.5) if rho_in == rho_in else False,
        "n_items": len(items),
        "n_pool": len(pool),
    }
