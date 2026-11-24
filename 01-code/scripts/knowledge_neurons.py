"""C10: measure OUR claims with SUN'S instrument (method triangulation).

Sun et al. (AAAI-26) identify an answer's KNOWLEDGE-NEURON group by vocab projection and validate the shared
store by activation/amplification. We reuse that exact instrument (the validated _entity_group below) to
re-measure our two pillars on cross-lingual two-hop FAILURES:

C10a --- the failure 2x2 (per failing item, per language):
        neuron-group ACTIVE on the failing prompt?  x  floor contains the entity (any language)?
        active + no-entity   = RENDER-side failure with the store lit (our wall, Sun-instrumented)
        silent + no-entity   = ACCESS failure (Sun's regime)
        -> "what fraction of CJK failures are store-lit but unspoken" is a number neither paper has.
C10b --- neuron-level FORM PROFILE: the group's summed vocab projection onto the answer's first token PER
        LANGUAGE, minus a script-matched random-token baseline (Delta-style control). Compare with the
        logit-lens form profile (C7): same skew from two independent instruments = triangulated.

    python scripts/knowledge_neurons.py --model gemma4_31b --solvmatrix 02-data/results/solvability/solvability_gemma4_31b.json
    python scripts/knowledge_neurons.py --model qwen36_27b --chat --solvmatrix 02-data/results/solvability/solvability_qwen36_27b.json
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import _form_matches, _normalize, entity_match
from beacon.data.prompting import encode, generate_answer
from beacon.hooks import _decoder_layers
from beacon.models import load_model
from beacon.pipeline.patching import first_answer_token_id

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

_rs = importlib.util.spec_from_file_location("rs", str(REPO_ROOT / "scripts/solvability_rescore.py"))
rs = importlib.util.module_from_spec(_rs); _rs.loader.exec_module(rs)
LANGS = ["en", "es", "ko", "ja", "zh"]


def _entity_group(loaded, ent_tok_ids, deep_layers, K):
    """Sun-style group discovery (no pre-discovered pool): rank deep-band MLP neurons by how strongly they PROMOTE
    the correct entity under vocab projection (unembed[t] . down_proj[:, i]); global top-K (layer, neuron).
    Cross-lingual by pooling the entity's first tokens across pivot+target."""
    dl = _decoder_layers(loaded.model)
    unemb = (loaded.model.get_output_embeddings() or loaded.model.get_input_embeddings()).weight
    ids = [t for t in ent_tok_ids if t is not None and t >= 0]
    if not ids:
        return {}
    u = unemb[torch.tensor(ids, device=unemb.device)].sum(0).float()
    cand = []
    for L in deep_layers:
        W = dl[int(L)].mlp.down_proj.weight
        s = u.to(W.device) @ W.float()
        topv, topi = torch.topk(s, min(K, s.shape[0]))
        cand += [(float(v), int(L), int(i)) for v, i in zip(topv.tolist(), topi.tolist())]
    cand.sort(reverse=True)
    by = {}
    for _v, L, i in cand[:K]:
        by.setdefault(L, []).append(i)
    return by


def _label_hit(pred, label):
    pn = _normalize(pred)
    return bool(pn and label) and _form_matches(pn, pn.split(), label)

def _script_bucket(tok: str) -> str:
    """First strong script in the token: hangul/kana/han; Latin only as a POST-loop fallback (an early
    per-char Latin branch would misclassify mixed-script surfaces). Local variant of 43's helper, which
    has no Latin bucket and no 'en' key."""
    for ch in tok:
        if "가" <= ch <= "힣":
            return "hangul"
        if "぀" <= ch <= "ヿ":
            return "kana"
        if "一" <= ch <= "鿿":
            return "han"
    if any(ch.isascii() and ch.isalpha() for ch in tok):
        return "latin"
    return "other"

LANG_BUCKET = {"ko": "hangul", "ja": "kana", "zh": "han", "en": "latin", "es": "latin"}

def _bucket_token_ids(loaded, cap=4000):
    """Vocab token ids grouped by script bucket, for script-matched Delta baselines."""
    tk = loaded.tokenizer
    pools = {"hangul": [], "kana": [], "han": [], "latin": []}
    for t in range(tk.vocab_size):
        s = tk.convert_ids_to_tokens(t)
        if s is None:
            continue
        try:
            s = tk.convert_tokens_to_string([s])
        except Exception:
            continue
        b = _script_bucket(s)
        if b in pools and len(pools[b]) < cap:
            pools[b].append(t)
            if all(len(v) >= cap for v in pools.values()):
                break
    return pools

def _matched_random_group(loaded, group, seed):
    """Random neuron group with per-layer counts MATCHED to the entity group's layer histogram (the entity
    group's global top-K concentrates in few layers; a uniform spread would under-control those layers)."""
    dl = _decoder_layers(loaded.model)
    g = torch.Generator().manual_seed(seed)
    return {int(L): torch.randperm(dl[int(L)].mlp.down_proj.weight.shape[1], generator=g)[:len(ns)].tolist()
            for L, ns in group.items()}

@torch.no_grad()
def _group_activation(loaded, prompt, chat, group, rand_group=None):
    """Mean |down_proj-input| activation of the group's neurons at the LAST token (one forward). If
    rand_group is given, its mean is read off the SAME forward (matched-control normalization: raw means
    across two prompts in different languages are language calorimetry, not a store-lit measurement)."""
    dl = _decoder_layers(loaded.model)
    acts = {}

    def mk(L):
        def hook(_m, args):
            acts[L] = args[0][0, -1].detach().float().abs().cpu()
        return hook

    layers = set(group) | set(rand_group or {})
    hs = [dl[int(L)].mlp.down_proj.register_forward_pre_hook(mk(L)) for L in layers]
    try:
        loaded.model(input_ids=encode(loaded, prompt, chat), use_cache=False)
    finally:
        for h in hs:
            h.remove()

    def _mean(gr):
        vals = [float(acts[L][i]) for L, ns in gr.items() for i in ns]
        return sum(vals) / max(1, len(vals))

    return _mean(group) if rand_group is None else (_mean(group), _mean(rand_group))

@torch.no_grad()
def _form_projection(loaded, group, labs, rand_pools):
    """C10b: group's summed vocab projection onto the answer's first token per language, baseline-subtracted.
    The baseline is SCRIPT-MATCHED to the actual answer token's bucket (ja answers are often pure kanji):
    a uniform-vocab baseline subtracts a language-invariant constant, so the cross-language skew would
    just inherit script-level unembedding statistics."""
    dl = _decoder_layers(loaded.model)
    unemb = (loaded.model.get_output_embeddings() or loaded.model.get_input_embeddings()).weight
    prof = {}
    for L in LANGS:
        t = first_answer_token_id(loaded, labs.get(L)) if labs.get(L) else -1
        if t < 0:
            prof[L] = None
            continue
        b = _script_bucket(loaded.tokenizer.decode([t]))
        ids = rand_pools.get(b) or rand_pools.get(LANG_BUCKET[L]) or rand_pools["uniform"]
        s, base = 0.0, 0.0
        u = unemb[t].float()
        ub = unemb[torch.tensor(ids, device=unemb.device)].float().mean(0)
        for lay, ns in group.items():
            W = dl[int(lay)].mlp.down_proj.weight.float()
            idx = torch.tensor(ns, device=W.device)
            s += float((u.to(W.device) @ W[:, idx]).sum())
            base += float((ub.to(W.device) @ W[:, idx]).sum())
        prof[L] = s - base                                                                        
    return prof

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--solvmatrix", help="default: 02-data/results/solvability/solvability_{model}.json")
    ap.add_argument("--targets", nargs="*", default=["es", "ko", "ja", "zh"])
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--n-per-lang", type=int, default=60)
    ap.add_argument("--k-neurons", type=int, default=120, help="Sun-style group size per item")
    ap.add_argument("--active-ratio", type=float, default=0.5,
                    help="active if (act_T/rand_act_T) >= ratio * (act_en/rand_act_en)")
    ap.add_argument("--chat", action="store_true")
    ap.add_argument("--out", default="02-data/results/access")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    pc = json.load(open(_path(a.solvmatrix or f"02-data/results/solvability/solvability_{a.model}.json")))["per_chain"]
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[si] loading {a.model} (chat={a.chat})", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    nL = loaded.spec.n_layers
    deep = list(range(int(0.5 * nL), nL - 1))

                                                                                 
    g = torch.Generator().manual_seed(7)
    rand_pools = {"uniform": torch.randint(0, loaded.tokenizer.vocab_size, (64,), generator=g).tolist()}
    for b, ids in _bucket_token_ids(loaded).items():
        if ids:
            pick = torch.randperm(len(ids), generator=g)[:min(64, len(ids))].tolist()
            rand_pools[b] = [ids[i] for i in pick]

    res = {"model": a.model, "k": a.k_neurons, "deep": [deep[0], deep[-1]],
           "protocol": {"active_ratio": a.active_ratio, "n_per_lang": a.n_per_lang, "chat": a.chat,
                        "solvmatrix": a.solvmatrix, "chains": a.chains, "targets": a.targets,
                        "rand_seed": 7, "rand_n": 64, "baseline": "script-matched-to-answer-token",
                        "floor_source": "fresh greedy decode", "floor_max_new": 32, "entity_free_floor": True,
                        "gate": "strict_x_correct T-fail + entity_match(strict_forms) en-solve",
                        "active_norm": "layer-matched random-group ratio"},
           "by_lang": {}}
    for T in a.targets:

                                                                                                         

        items = []
        for c in pc:
            gtext = pc[c]["gen"]
            if c not in C or T not in gtext or "en" not in gtext:
                continue
            labs = C[c]["answer_label_by_lang"]
            if not labs.get(T) or labs.get(T) == labs.get("en"):
                continue
            f = rs.strict_forms(c)
            if rs.strict_x_correct(gtext[T], c, T) is False and entity_match(gtext["en"], f[0] if f else "", f):
                items.append(c)
            if len(items) >= a.n_per_lang:
                break
        rows, skipped = [], 0
        for i, c in enumerate(items):
            labs = C[c]["answer_label_by_lang"]

                                                                                                                 
            floor = generate_answer(loaded, C[c]["prompts"][T], a.chat, max_new=32)
            ent_in_floor = any(_label_hit(floor, labs.get(L)) for L in LANGS if labs.get(L))
            if ent_in_floor:
                skipped += 1
                continue
            ent_ids = [first_answer_token_id(loaded, labs.get(L)) for L in LANGS if labs.get(L)]
            group = _entity_group(loaded, ent_ids, deep, a.k_neurons)                                          
            if not group:
                continue
            rand_group = _matched_random_group(loaded, group, seed=hash(c) & 0x7FFFFFFF)
            act_T, rand_act_T = _group_activation(loaded, C[c]["prompts"][T], a.chat, group, rand_group)
            act_en, rand_act_en = _group_activation(loaded, C[c]["prompts"]["en"], a.chat, group, rand_group)
            eps = 1e-9
            active = (act_T / max(rand_act_T, eps)) >= a.active_ratio * (max(act_en, eps) / max(rand_act_en, eps))
            prof = _form_projection(loaded, group, labs, rand_pools)
            rows.append({"chain": c, "act_T": act_T, "act_en": act_en, "rand_act_T": rand_act_T,
                         "rand_act_en": rand_act_en, "active": bool(active), "floor": floor,
                         "entity_in_floor": bool(ent_in_floor), "form_profile": prof})
            if (i + 1) % 20 == 0:
                print(f"    {T} [{i+1}/{len(items)}]", flush=True)
        n = len(rows)
        q = {"lit_unspoken": sum(1 for r in rows if r["active"] and not r["entity_in_floor"]),
             "lit_wronglang": sum(1 for r in rows if r["active"] and r["entity_in_floor"]),
             "silent": sum(1 for r in rows if not r["active"])}
        res["by_lang"][T] = {"n": n, "pseudo_fail_skipped": skipped, "quadrants": q, "items": rows}
        print(f"[si] {T}: n={n} (pseudo-fail skipped={skipped}) | store-LIT but unspoken={q['lit_unspoken']} "
              f"| lit+entity-wrong-lang={q['lit_wronglang']} | store-SILENT (Sun's access regime)={q['silent']}",
              flush=True)

        agg = {L: sum(r["form_profile"][L] for r in rows if r["form_profile"].get(L) is not None) / max(1, n)
               for L in LANGS}
        print(f"      neuron-level form availability (Delta-controlled): " +
              " ".join(f"{L}={agg[L]:.1f}" for L in LANGS), flush=True)
        res["by_lang"][T]["form_agg"] = agg
        (out_dir / f"suninstrument_{a.model}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[si] DONE --- compare form_agg with the logit-lens C7 profile: same skew = triangulated.", flush=True)

if __name__ == "__main__":
    main()
