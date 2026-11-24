"""Stage 1: layer-wise block ablation to localize the cross-lingual two-hop signal.

A coarse per-layer localizer. For each decoder layer, zero-ablate the ATTENTION block output and
(separately) the MLP block output, and measure the increase in two-hop answer NLL. Layers where the
attention effect exceeds the MLP effect are the candidate layers in which Stage 2 scores heads.
Only layers with standard full attention (see `eligible_attention_layers`) are ablated on the
attention side; other layers report NaN.
"""
from __future__ import annotations

import torch

from beacon.hooks import _decoder_layers, eligible_attention_layers
from beacon.data.prompting import encode


@torch.no_grad()
def _two_hop_nll(loaded, items, chat: bool = False) -> float:
    tok = loaded.tokenizer
    nlls = []
    for it in items:
        if it["variant"] != "two_hop":
            continue
        p_ids = encode(loaded, it["prompt"], chat).cpu()
        a_ids = tok(" " + it["answer_label"], add_special_tokens=False, return_tensors="pt").input_ids
        full = torch.cat([p_ids, a_ids], dim=1).to(loaded.device)
        logits = loaded.model(full).logits
        s = p_ids.shape[1]
        logp = torch.log_softmax(logits[0, s - 1:-1].float(), dim=-1)
        nlls.append(-logp[range(full.shape[1] - s), full[0, s:]].mean().item())
    return float(sum(nlls) / len(nlls)) if nlls else float("nan")


def _zero_output_hook(_module, _args, output):
    """Zero a block's output (tuple or tensor) to ablate its contribution at this layer."""
    if isinstance(output, tuple):
        return (torch.zeros_like(output[0]),) + tuple(output[1:])
    return torch.zeros_like(output)


@torch.no_grad()
def localize_signal(loaded, items, chat: bool = False) -> dict:
    """Return per-layer attention-vs-MLP ablation effects; expect attention to dominate.

    {"attn": [dNLL per layer], "mlp": [dNLL per layer], "attn_dominates_layers": [...]}.
    """
    layers = _decoder_layers(loaded.model)
    two_hop = [it for it in items if it["variant"] == "two_hop"]
    base = _two_hop_nll(loaded, two_hop, chat)

    eligible = set(eligible_attention_layers(loaded.model))
    attn_eff, mlp_eff = [float("nan")] * loaded.spec.n_layers, []
    for L in range(loaded.spec.n_layers):
        if L in eligible:
            h = layers[L].self_attn.register_forward_hook(_zero_output_hook)
            attn_eff[L] = _two_hop_nll(loaded, two_hop, chat) - base
            h.remove()
        h = layers[L].mlp.register_forward_hook(_zero_output_hook)
        mlp_eff.append(_two_hop_nll(loaded, two_hop, chat) - base)
        h.remove()

    dominate = [L for L in eligible if attn_eff[L] > mlp_eff[L]]
    return {"attn": attn_eff, "mlp": mlp_eff, "eligible_attention_layers": sorted(eligible),
            "attn_dominates_layers": dominate, "base_nll": base}
