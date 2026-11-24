"""Activation patching, head amplification, and residual steering at the readout position.

Head-level activation patch via o_proj-INPUT substitution: the same per-head decomposition the colon
readout uses (a_in[h*d_head:(h+1)*d_head] is head h's o_proj input). To patch head h we overwrite that
slice at the colon with a DONOR run's slice, so the head's write into the residual becomes the donor's.
Forward-only, no gradients. The random-pool control patches a size-matched random head set instead.

The readout position is the 'Answer:' colon for raw prompts and the last token under a chat template.
This module also provides head amplification at that position, diff-in-means steering vectors, greedy
decoding with interventions active at every step, and the layer-matched random head-pool control.
"""
from __future__ import annotations

import torch

from beacon.hooks import _decoder_layers, attention_head_dim
from beacon.models import answer_colon_position
from beacon.data.prompting import encode_inputs, has_chat_template


def _encoded_col(loaded, prompt: str, chat: bool):
    enc = encode_inputs(loaded, prompt, chat)
    if chat and has_chat_template(loaded):
        return enc, enc["input_ids"].shape[1] - 1
    return enc, answer_colon_position(loaded, prompt)


def _heads_by_layer(pool):
    by: dict[int, list[int]] = {}
    for (layer, head) in pool:
        by.setdefault(int(layer), []).append(int(head))
    return by


@torch.no_grad()
def capture_oproj_inputs(loaded, prompt: str, layers_needed, chat: bool = False) -> dict:
    """{layer: o_proj input at the colon, (d_model,)} for a donor run --- the source of patched slices."""
    layers = _decoder_layers(loaded.model)
    enc, col = _encoded_col(loaded, prompt, chat)
    grabbed: dict[int, torch.Tensor] = {}

    def mk(L):
        def pre(_m, args):
            grabbed[L] = args[0][0, col].detach().clone()
        return pre

    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L)) for L in layers_needed]
    try:
        loaded.model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return grabbed


@torch.no_grad()
def colon_logits(loaded, prompt: str, pool=None, donor_inputs=None, d_head: int | None = None,
                 chat: bool = False):
    """Next-token logits at the colon (the answer's first token). If pool + donor_inputs are given, the
    pool heads' o_proj-input slices at the colon are REPLACED by the donor's before the forward."""
    layers = _decoder_layers(loaded.model)
    enc, col = _encoded_col(loaded, prompt, chat)
    handles = []
    if pool and donor_inputs is not None:
        hbl = _heads_by_layer(pool)

        def mk(L, heads):
            dh = d_head or attention_head_dim(layers[L], loaded.spec.n_heads)
            def pre(_m, args):
                a = args[0].clone()
                for hh in heads:
                    sl = donor_inputs[L][hh * dh:(hh + 1) * dh].to(device=a.device, dtype=a.dtype)
                    a[0, col, hh * dh:(hh + 1) * dh] = sl
                return (a,) + tuple(args[1:])
            return pre

        handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L, hs))
                   for L, hs in hbl.items() if L in donor_inputs]
    try:
        out = loaded.model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, col].float().cpu()


@torch.no_grad()
def colon_logits_amplify(loaded, prompt: str, pool, alpha: float, d_head: int | None = None,
                         chat: bool = False):
    """Next-token logits at the colon with the pool heads' o_proj-input slices SCALED by (1+alpha) at the
    colon --- the AMPLIFICATION intervention (magnitude, not content). Scaling a_h by (1+alpha) scales
    the head write o_h = W_O_h @ a_h by (1+alpha). Pairs with colon_logits (the content/donor intervention) so a
    per-item amp-vs-patch contrast separates under-activation (amp fixes) from re-routing (only patch does)."""
    layers = _decoder_layers(loaded.model)
    enc, col = _encoded_col(loaded, prompt, chat)
    hbl = _heads_by_layer(pool)

    def mk(L, heads):
        dh = d_head or attention_head_dim(layers[L], loaded.spec.n_heads)
        def pre(_m, args):
            a = args[0].clone()
            for hh in heads:
                a[0, col, hh * dh:(hh + 1) * dh] *= (1.0 + alpha)
            return (a,) + tuple(args[1:])
        return pre

    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L, hs)) for L, hs in hbl.items()]
    try:
        out = loaded.model(**enc, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return out.logits[0, col].float().cpu()


def _unwrap_layer_out(out):
    """A decoder layer's forward output is a bare tensor OR a tuple whose [0] is the hidden state."""
    if isinstance(out, tuple):
        return out[0], out[1:]
    return out, None


@torch.no_grad()
def colon_residuals(loaded, prompt: str, layers_needed, chat: bool = False):
    """{L: residual written by layer L at the 'Answer:' colon} = hidden_states[L+1][colon]. Source for the
    diff-in-means language-steering vector (captured on generic content-orthogonal prompts)."""
    enc, col = _encoded_col(loaded, prompt, chat)
    hs = loaded.model(**enc, use_cache=False, output_hidden_states=True).hidden_states
    return {int(L): hs[int(L) + 1][0, col].detach().float().cpu() for L in layers_needed}


@torch.no_grad()
def build_steering_vectors(loaded, pairs, layers, tgt_key: str = "es", src_key: str = "en",
                           chat: bool = False):
    """Chain-independent language-steering direction per layer: unit-normalized mean of (residual_tgt -
    residual_src) at the colon over generic parallel prompt pairs (answer-agnostic: the paired difference
    cancels shared content, averaging over many distinct answers suppresses per-answer residue ~1/sqrt(N)).
    tgt_key/src_key pick the parallel prompt fields (es-en by default; ko/ja for the cross-language control).
    Returns (vhat: {L: unit tensor}, cancel: {L: ||mean diff|| / mean ||diff||} --- high => content cancelled)."""
    acc = {int(L): None for L in layers}
    norms = {int(L): 0.0 for L in layers}
    n = 0
    for p in pairs:
        re = colon_residuals(loaded, p[tgt_key], layers, chat)
        rn = colon_residuals(loaded, p[src_key], layers, chat)
        for L in acc:
            d = re[L] - rn[L]
            acc[L] = d.clone() if acc[L] is None else acc[L] + d
            norms[L] += float(d.norm())
        n += 1
    vhat, cancel = {}, {}
    for L in acc:
        mean = acc[L] / n
        mn = float(mean.norm())
        vhat[L] = mean / (mn + 1e-8)
        cancel[L] = mn / (norms[L] / n + 1e-8)
    return vhat, cancel


@torch.no_grad()
def colon_decode(loaded, prompt: str, max_new: int = 5, pool=None, donor_inputs=None,
                 alpha: float | None = None, d_head: int | None = None, steer=None,
                 chat: bool = False) -> str:
    """Greedy-decode max_new tokens AFTER the colon WITH interventions active at every step (no KV-cache, so
    hooks re-fire each forward). Content intervention (at the fixed colon): pool heads' o_proj-INPUT slices are
    REPLACED by donor slices (patch) or SCALED by (1+alpha) (amp). Language intervention `steer` = list of
    (layer, vhat, alpha_s): adds alpha_s*||resid_last||*vhat to the residual WRITTEN by that layer at the
    LAST (next-token) position on every step --- so it re-languages each GENERATED token, not just the colon.
    Returns the decoded continuation string."""
    layers = _decoder_layers(loaded.model)
    enc, col = _encoded_col(loaded, prompt, chat)
    hbl = _heads_by_layer(pool) if pool else {}
    steer = steer or []

    def mk(L, heads):
        dh = d_head or attention_head_dim(layers[L], loaded.spec.n_heads)
        def pre(_m, args):
            a = args[0].clone()
            for hh in heads:
                if donor_inputs is not None and L in donor_inputs:
                    a[0, col, hh * dh:(hh + 1) * dh] = donor_inputs[L][hh * dh:(hh + 1) * dh].to(
                        device=a.device, dtype=a.dtype)
                elif alpha is not None:
                    a[0, col, hh * dh:(hh + 1) * dh] *= (1.0 + alpha)
            return (a,) + tuple(args[1:])
        return pre

    def mk_steer(vhat, a_s):
        def hook(_m, _inp, out):
            h, rest = _unwrap_layer_out(out)
            h = h.clone()
            v = vhat.to(device=h.device, dtype=h.dtype)
            last = h[0, -1]
            h[0, -1] = last + a_s * last.norm() * v
            return (h,) + tuple(rest) if rest is not None else h
        return hook

    ids = enc["input_ids"]
    out_ids = []
    for _ in range(max_new):
        handles = []
        if hbl:
            handles += [layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L, hs))
                        for L, hs in hbl.items() if (donor_inputs is None or L in donor_inputs)]
        for (L, vhat, a_s) in steer:
            handles.append(layers[int(L)].register_forward_hook(mk_steer(vhat, a_s)))
        try:
            logits = loaded.model(input_ids=ids, use_cache=False).logits
        finally:
            for h in handles:
                h.remove()
        nxt = int(logits[0, -1].argmax())
        out_ids.append(nxt)
        if nxt == loaded.tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return loaded.tokenizer.decode(out_ids, skip_special_tokens=True)


def first_answer_token_id(loaded, answer_label: str) -> int:
    """First token id of the answer. A leading space is prepended for Latin (BPE word-boundary), but NOT for
    CJK/Hangul --- prepending ' ' before a CJK char makes the first token a broken UTF-8 byte fragment, so the
    pivot would track a garbage id for zh/ja/ko labels."""
    cjk = any("぀" <= c <= "鿿" or "가" <= c <= "힣" or "぀" <= c <= "ヿ" for c in (answer_label or "")[:1])
    text = answer_label if cjk else " " + (answer_label or "")
    ids = loaded.tokenizer(text, add_special_tokens=False)["input_ids"]
    return int(ids[0]) if ids else -1


def random_pool_like(
    pool, n_layers: int, n_heads: int, seed: int = 0, allow_min_overlap: bool = False
):
    """A size- and layer-matched random head pool (the mandatory control).

    By default the control must be fully disjoint from the selected pool.  If a
    layer contains more than half of all heads, that exact control is
    mathematically impossible and this function raises ``ValueError``.

    ``allow_min_overlap=True`` is an explicitly exploratory fallback: include
    every non-selected head and randomly sample the minimum number of selected
    heads needed to preserve both layer count and distinct head identities.
    """
    g = torch.Generator().manual_seed(seed)
    by_layer = _heads_by_layer(pool)
    out = []
    for layer, original in sorted(by_layer.items()):
        original_set = set(original)
        available = [h for h in range(n_heads) if h not in original_set]
        if len(available) < len(original) and not allow_min_overlap:
            raise ValueError(
                f"cannot draw {len(original)} distinct disjoint controls at layer {layer}; "
                f"only {len(available)} non-selected heads exist"
            )
        take_external = min(len(original), len(available))
        external_order = torch.randperm(len(available), generator=g)[:take_external].tolist()
        chosen = [available[i] for i in external_order]
        shortfall = len(original) - len(chosen)
        if shortfall:
            original_order = torch.randperm(len(original), generator=g)[:shortfall].tolist()
            chosen.extend(original[i] for i in original_order)
        out.extend((layer, h) for h in chosen)
    return out
