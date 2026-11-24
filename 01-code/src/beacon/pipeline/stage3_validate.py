"""Stage 3: diagnostics for the candidate head pool.

  - Jaccard overlap of pools across languages / seeds (stability).
  - Mean-ablation: the increase in two-hop answer NLL when the pool's o_proj inputs are replaced by
    their mean, reported against a configured reference band and against a strictly disjoint,
    layer- and size-matched random pool.
These diagnostics never resize or prune the pool.
"""
from __future__ import annotations

import torch

from beacon.hooks import _decoder_layers, attention_head_dim
from beacon.data.prompting import encode, encode_inputs
from beacon.pipeline.patching import random_pool_like


def _heads_by_layer(pool):
    by = {}
    for (layer, head) in pool:
        by.setdefault(int(layer), []).append(int(head))
    return by


@torch.no_grad()
def _two_hop_nll(loaded, items, chat: bool = False) -> float:
    """Mean teacher-forced NLL of the gold answer over two_hop items (no ablation)."""
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
        gold = full[0, s:]
        nlls.append(-logp[range(len(gold)), gold].mean().item())
    return float(sum(nlls) / len(nlls)) if nlls else float("nan")


@torch.no_grad()
def _pool_mean_slices(loaded, items, pool, chat: bool = False) -> dict:
    """Per pooled layer, the mean o_proj-input vector over items x positions (for mean-ablation)."""
    layers = _decoder_layers(loaded.model)
    needed = sorted(_heads_by_layer(pool))
    sums, count = {L: None for L in needed}, 0
    grabbed = {}

    def mk(L):
        def hook(_m, args):
            grabbed[L] = args[0].detach()
        return hook

    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(mk(L)) for L in needed]
    tok = loaded.tokenizer
    try:
        for it in items:
            enc = encode_inputs(loaded, it["prompt"], chat)
            loaded.model(**enc, use_cache=False)
            for L in needed:
                a = grabbed[L][0]                       # (S, d_in)
                sums[L] = a.sum(0).float() if sums[L] is None else sums[L] + a.sum(0).float()
            count += enc["input_ids"].shape[1]
    finally:
        for h in handles:
            h.remove()
    return {L: (sums[L] / max(1, count)) for L in needed}


def _ablation_prehook(heads, mean_vec, d_head):
    def hook(_module, args):
        a_in = args[0].clone()
        for h in heads:
            sl = mean_vec[h * d_head:(h + 1) * d_head].to(device=a_in.device, dtype=a_in.dtype)
            a_in[..., h * d_head:(h + 1) * d_head] = sl     # device-safe under model sharding
        return (a_in,) + tuple(args[1:])
    return hook


@torch.no_grad()
def ablation_dnll(loaded, items, pool, chat: bool = False) -> float:
    """Mean-ablate the pool and return the two-hop NLL increase over the clean run."""
    two_hop = [it for it in items if it["variant"] == "two_hop"]
    if not two_hop or not pool:
        return float("nan")
    means = _pool_mean_slices(loaded, two_hop, pool, chat)
    clean = _two_hop_nll(loaded, two_hop, chat)

    layers = _decoder_layers(loaded.model)
    handles = [layers[L].self_attn.o_proj.register_forward_pre_hook(
        _ablation_prehook(
            heads, means[L], attention_head_dim(layers[L], loaded.spec.n_heads)
        )) for L, heads in _heads_by_layer(pool).items()]
    try:
        ablated = _two_hop_nll(loaded, two_hop, chat)
    finally:
        for h in handles:
            h.remove()
    return ablated - clean


def jaccard_stability(pools_by_seed) -> float:
    """Mean pairwise Jaccard overlap of the pool across seeds / languages."""
    pools = [set(map(tuple, p)) for p in pools_by_seed]
    if len(pools) < 2:
        return float("nan")
    sims = []
    for i in range(len(pools)):
        for j in range(i + 1, len(pools)):
            u = pools[i] | pools[j]
            sims.append(len(pools[i] & pools[j]) / len(u) if u else 0.0)
    return float(sum(sims) / len(sims))


def finalize_pool(loaded, candidate_pool, items, configs, chat: bool = False):
    """Measure mean-ablation and return the unchanged selected pool plus diagnostics.

    No head-level pruning is applied. The reference band (configs/pools.yaml, calibrated on Llama)
    is reported as a diagnostic only, especially for other architectures.
    """
    band = configs.pools["validation"]["general_dnll_band"]
    dnll = ablation_dnll(loaded, items, candidate_pool, chat)
    random_seed = 7
    random_error = None
    try:
        random_pool = random_pool_like(
            candidate_pool, loaded.spec.n_layers, loaded.spec.n_heads, seed=random_seed)
    except ValueError as exc:
        random_pool = None
        random_error = str(exc)
    if random_pool is None:
        random_overlap = None
        random_dnll = None
        contrast = None
    else:
        random_overlap = len(set(candidate_pool) & set(random_pool))
        random_dnll = ablation_dnll(loaded, items, random_pool, chat)
        contrast = dnll - random_dnll
    passed = (dnll == dnll) and (band[0] <= dnll <= band[1])
    print(f"  [stage3 diagnostic] mean-ablation dNLL={dnll:.3f} | "
          f"Llama reference band {band} -> {'IN BAND' if passed else 'OUT OF BAND'}")
    if random_error is None:
        print(f"  [stage3 control] selected={dnll:.3f} random={random_dnll:.3f} "
              f"selected-random={contrast:+.3f} overlap={random_overlap}/{len(candidate_pool)} "
              f"(strict disjoint; seed={random_seed})")
    else:
        print(f"  [stage3 control] INFEASIBLE under strict disjoint layer/size matching: "
              f"{random_error}")
    pool = [(int(L), int(H), f"L{L}H{H}") for (L, H) in candidate_pool]
    return pool, {"mean_ablation_dnll": dnll, "reference_band": list(band),
                  "within_llama_reference_band": bool(passed),
                  "layer_matched_random_dnll": random_dnll,
                  "selected_minus_random_dnll": contrast,
                  "random_pool_overlap": random_overlap,
                  "random_control_status": "ok" if random_error is None else "infeasible",
                  "random_control_rule": "strict disjoint exact layer and size matching",
                  "random_control_error": random_error,
                  "random_seed": random_seed,
                  "role": "held-out positive control; never used to resize the pool"}
