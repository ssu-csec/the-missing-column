"""Stage 2: LAHIS gradient importance and head-pool selection (SCH discovery).

Score every head with LAHIS under the three-condition contrast (configs/pools.yaml):
    BRS(h) = z(I_FH) + z(I_TH) - z(I_SH)
Heads that appear in the per-language top-N pools of at least `general_pool.min_languages` languages
form the candidate general pool, subject to a per-layer cap that guarantees a layer-matched, disjoint
random control exists. Stage 3 reports mean-ablation diagnostics on the result.

NOTE: this stage needs a backward pass (gradient importance); the other stages are forward-only.

Method (attribution patching). Insert a per-head multiplicative mask m_h (=1) on each head's OV write
by scaling the o_proj INPUT slice for head h (a_in[h*d_head:(h+1)*d_head] *= m_h, the same per-head
decomposition the colon readout uses). Then o_proj output = sum_h m_h * o_h, and dL/dm_h = <dL/do_h,
o_h> is head h's first-order importance for the loss L (the NLL of the gold answer for that condition).
Weights are FROZEN (requires_grad=False), so the masks are the only grad-bearing tensors --- this is
what lets LAHIS fit a 70B backward on 2x80GB (configs/compute.yaml: lahis.freeze_weights). Conditions
map to query variants: first_hop -> I_FH, two_hop -> I_TH, shortcut_no_r1 -> I_SH.
"""
from __future__ import annotations

import torch

from beacon.hooks import _decoder_layers, attention_head_dim, eligible_attention_layers
from beacon.data.prompting import encode

_VARIANT_TO_CONDITION = {"first_hop": "I_FH", "two_hop": "I_TH", "shortcut_no_r1": "I_SH"}


def _answer_nll(loaded, prompt: str, answer_label: str, chat: bool = False) -> torch.Tensor:
    """Differentiable teacher-forced NLL of the gold answer tokens after the prompt."""
    tok = loaded.tokenizer
    p_ids = encode(loaded, prompt, chat).cpu()
    a_ids = tok(" " + answer_label, add_special_tokens=False, return_tensors="pt").input_ids
    full = torch.cat([p_ids, a_ids], dim=1).to(loaded.device)
    logits = loaded.model(full).logits
    ans_start = p_ids.shape[1]
    logp = torch.log_softmax(logits[0, ans_start - 1:-1].float(), dim=-1)
    gold = full[0, ans_start:]
    return -logp[range(len(gold)), gold].mean()


def _make_mask_prehook(mask: torch.Tensor, d_head: int):
    """forward_pre_hook that scales the o_proj input per query head by `mask` (n_heads,).

    On a model sharded across GPUs (device_map), this layer's o_proj input a_in lives on the layer's
    device, which need not be the device the mask was created on. Cast scale to a_in's device+dtype ---
    a differentiable copy, so the gradient still flows back to `mask` (which we read for importance).
    """
    def _hook(_module, args):
        a_in = args[0]
        scale = mask.repeat_interleave(d_head).to(device=a_in.device, dtype=a_in.dtype)
        return (a_in * scale,) + tuple(args[1:])
    return _hook


def lahis_scores(loaded, items, gradient_checkpointing: bool = False,
                 chat: bool = False) -> dict:
    """Per-head LAHIS importance for each of I_FH, I_TH, I_SH conditions (z-scored), plus BRS.

    Returns {"I_FH": (L,H), "I_TH": (L,H), "I_SH": (L,H), "BRS": (L,H)} as CPU tensors. `items` are
    rendered prompts (one language); only first_hop / two_hop / shortcut_no_r1 variants contribute.
    """
    model = loaded.model
    layers = _decoder_layers(model)
    n_layers, n_heads = loaded.spec.n_layers, loaded.spec.n_heads
    eligible = eligible_attention_layers(model)
    if not eligible:
        raise RuntimeError("SCH discovery found no layers with self_attn.o_proj")

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    accum = {c: torch.zeros(n_layers, n_heads) for c in ("I_FH", "I_TH", "I_SH")}
    counts = {c: 0 for c in accum}

    for item in items:
        cond = _VARIANT_TO_CONDITION.get(item["variant"])
        if cond is None:
            continue
        # one mask per layer, on THAT layer's o_proj device (the model may be sharded across GPUs)
        masks = {L: torch.ones(n_heads, device=layers[L].self_attn.o_proj.weight.device,
                               requires_grad=True) for L in eligible}
        handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(
            _make_mask_prehook(masks[L], attention_head_dim(layers[L], n_heads)))
            for L in eligible]
        try:
            model.zero_grad(set_to_none=True)
            loss = _answer_nll(loaded, item["prompt"], item["answer_label"], chat)
            loss.backward()
            for L in eligible:
                if masks[L].grad is not None:
                    accum[cond][L] += masks[L].grad.detach().abs().cpu()
            counts[cond] += 1
        finally:
            for h in handles:
                h.remove()

    scores = {}
    for c, a in accum.items():
        mean = a / max(1, counts[c])
        valid = mean[eligible].flatten()
        z = torch.full_like(mean, float("-inf"))
        z[eligible] = ((valid - valid.mean()) / (valid.std() + 1e-8)).reshape(len(eligible), n_heads)
        scores[c] = z
    scores["BRS"] = scores["I_FH"] + scores["I_TH"] - scores["I_SH"]
    scores["_counts"] = counts
    scores["_eligible_layers"] = eligible
    return scores


def select_general_pool(
    scores_by_language: dict, configs, return_diagnostics: bool = False
) -> list[tuple[int, int]] | tuple[list[tuple[int, int]], dict]:
    """Per-language top-N pools combined into the candidate general pool.

    scores_by_language maps lang -> the scores dict from lahis_scores (uses its "BRS" map). A head joins
    the candidate pool if it lands in the per-language top-N for at least `general_pool.min_languages`
    languages (configs/pools.yaml). When that key is absent, the STRICT intersection over ALL languages
    is used. Returns (layer, head) coordinates.
    """
    from collections import Counter

    gp = configs.pools["general_pool"]
    total_eligible = int(torch.isfinite(
        next(iter(scores_by_language.values()))["BRS"]).sum().item())
    top_n = round(total_eligible * gp["per_language_top_fraction"]) \
        if gp.get("per_language_top_fraction") is not None else gp["per_language_top_n"]
    top_n = max(1, int(top_n))
    per_lang_pools = []
    for _lang, scores in scores_by_language.items():
        brs = scores["BRS"]
        n_heads = brs.shape[1]
        finite = torch.isfinite(brs.flatten())
        eligible_indices = finite.nonzero(as_tuple=False).flatten()
        take = min(int(top_n), int(eligible_indices.numel()))
        order = eligible_indices[brs.flatten()[eligible_indices].argsort(descending=True)[:take]]
        per_lang_pools.append({(int(i // n_heads), int(i % n_heads)) for i in order.tolist()})
    if not per_lang_pools:
        return []
    k = gp.get("min_languages") or len(per_lang_pools)       # default: strict all-language intersection
    counts = Counter(h for pool in per_lang_pools for h in pool)
    general = {h for h, c in counts.items() if c >= k}

    unconstrained_count = len(general)

    # The mandatory random control is layer matched, equal sized, distinct, and
    # disjoint from the selected pool.  Such a control exists iff no selected
    # layer contains more than half of that layer's query heads.  This
    # feasibility condition is enforced during selection, uniformly for every
    # architecture, so the control rule never depends on the selected pool.
    n_heads = next(iter(scores_by_language.values()))["BRS"].shape[1]
    per_layer_cap = n_heads // 2
    enforce_feasibility = gp.get("enforce_disjoint_control_feasibility", True)
    by_layer = {}
    for coord in general:
        by_layer.setdefault(coord[0], []).append(coord)

    constrained = []
    dropped = []
    for layer, coords in sorted(by_layer.items()):
        def rank(coord):
            values = []
            for scores in scores_by_language.values():
                value = float(scores["BRS"][coord[0], coord[1]].item())
                if value != float("-inf"):
                    values.append(value)
            return (counts[coord], sum(values) / len(values), -coord[1])

        ordered = sorted(coords, key=rank, reverse=True)
        keep = per_layer_cap if enforce_feasibility else len(ordered)
        constrained.extend(ordered[:keep])
        dropped.extend(ordered[keep:])

    if dropped:
        dropped_by_layer = Counter(layer for layer, _head in dropped)
        print(
            "  [selection control-feasibility] "
            f"dropped {len(dropped)} heads across layers {dict(sorted(dropped_by_layer.items()))}; "
            f"per-layer cap={per_layer_cap}/{n_heads}"
        )
    selected = sorted(constrained)
    diagnostics = {
        "unconstrained_candidate_count": unconstrained_count,
        "selected_count_after_control_feasibility": len(selected),
        "dropped_count": len(dropped),
        "dropped_by_layer": {
            str(layer): count
            for layer, count in sorted(Counter(L for L, _H in dropped).items())
        },
        "n_query_heads": n_heads,
        "per_layer_cap": per_layer_cap,
        "enforced": bool(enforce_feasibility),
    }
    return (selected, diagnostics) if return_diagnostics else selected
