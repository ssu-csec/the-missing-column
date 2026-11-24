"""Second-hop content-head (SCH) amplification with output-language adjudication.

SCH amplification scales the SCH o_proj-input slices x(1+alpha) at the answer colon (or the assistant generation
marker under --chat) on EN-success/X-fail two-hop items. Heads come from the SCH pool file given by --heads
(general or language-specific); decoding uses colon_decode. Every decode is scored on two criteria:

  render_T   : output_language == T AND target-language label/native-alias match (designated-form realization)
  entity_any : the answer entity appears in any language (content)
  out-lang distribution + first-token language adjudication (which language's gold label the colon argmax hits)
  random-head control: layer-matched random pool (patching.random_pool_like), same alpha.

Only items whose untreated 32-token floor is still failing and entity-free are kept.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import hashlib
import json
import platform

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import _form_matches, _normalize
from beacon.data.langid import output_language
from beacon.models import load_model
from beacon.pipeline.patching import colon_decode, colon_logits, first_answer_token_id, random_pool_like

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

LANGS = ["en", "es", "ko", "ja", "zh"]


def _label_hit(pred, label):
    pn = _normalize(pred)
    return bool(pn and label) and _form_matches(pn, pn.split(), label)


def _script_ok(s, lang):
    """Does the string carry the target language's script? (native-alias filter; hangul/kana/han buckets)"""
    has = {"hangul": False, "kana": False, "han": False}
    for ch in s:
        if "가" <= ch <= "힣":
            has["hangul"] = True
        elif "぀" <= ch <= "ヿ":
            has["kana"] = True
        elif "一" <= ch <= "鿿":
            has["han"] = True
    if lang == "ko":
        return has["hangul"]
    if lang == "ja":
        return has["kana"] or has["han"]
    if lang == "zh":
        return has["han"] and not has["kana"]
    return not any(has.values())                              # es: latin forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama31_70")
    ap.add_argument("--targets", nargs="*", default=["ko", "ja", "zh", "es"])
    ap.add_argument(
        "--heads",
        required=True,
        help="general or language-specific SCH pool JSON",
    )
    ap.add_argument("--set-type", default="general", choices=["general", "specific"],
                    help="general = the cross-lingual transfer arm; specific = per-language pools")
    ap.add_argument("--subset-dir", default="02-data/inputs/sch_subsets",
                    help="per-language failure subsets subset_<model>_<T>_failure.jsonl (not distributed)")
    ap.add_argument("--dataset", default="02-data/inputs/dataset_campaign.jsonl", help="campaign prompt set (not distributed)")
    ap.add_argument("--rich-aliases", default="02-data/inputs/aliases_rich.json", help="expanded alias inventory (not distributed)")
    ap.add_argument("--alphas", nargs="*", type=float, default=[0.25, 0.5, 1.0])
    ap.add_argument("--max-new", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="cap items per language (0 = all)")
    ap.add_argument("--chat", action="store_true",
                    help="use the checkpoint chat template (Qwen: enable_thinking=False)")
    ap.add_argument("--out", default="02-data/results/interventions")
    a = ap.parse_args()
    load_env(); cfg = load_configs()
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    # Model-specific discovered heads. Accept either a flat list or a discovery manifest.
    head_doc = json.load(open(_path(a.heads)))
    if isinstance(head_doc, dict):
        status = head_doc.get("pool_status")
        validation = head_doc.get("validation", {})
        if status in {"control_infeasible", "positive_control_failed"}:
            raise SystemExit(f"[sch] pool is not eligible for rendering: status={status}")
        if status is None and validation.get("selected_minus_random_dnll") is not None \
                and validation["selected_minus_random_dnll"] <= 0:
            raise SystemExit("[sch] pool is not eligible for rendering: non-positive causal control")
    heads_all = head_doc.get("heads", []) if isinstance(head_doc, dict) else head_doc
    if a.set_type == "general":
        pool_by_lang = {T: [(h["layer"], h["head"]) for h in heads_all
                            if h["model"] == a.model and h["set_type"] == "general"] for T in a.targets}
    else:
        pool_by_lang = {T: [(h["layer"], h["head"]) for h in heads_all
                            if h["model"] == a.model and h["set_type"] != "general" and h["lang"] == T]
                        for T in a.targets}
    for T in a.targets:
        if not pool_by_lang[T]:
            raise SystemExit(f"[sch] no heads for model={a.model} set={a.set_type} lang={T} in {a.heads}")

    # per-chain per-language labels from the two-hop dataset (subset rows carry only their own language's label)
    labels = {}
    for line in open(_path(a.dataset)):
        r = json.loads(line)
        if r.get("variant") != "two_hop":
            continue
        labels.setdefault(r["chain_id"], {})[r["lang"]] = r["answer_label"]
    rich = json.load(open(_path(a.rich_aliases)))

    print(f"[sch] loading {a.model} ...", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")
    res = {"model": a.model, "set_type": a.set_type, "alphas": a.alphas,
           "protocol": {"mechanism": "o_proj-input slice x(1+alpha) at assistant generation marker"
                                     if a.chat else
                                     "o_proj-input slice x(1+alpha) at answer colon",
                        "max_new": a.max_new, "floor": "32-tok clean re-decode, entity-free",
                        "chat": a.chat,
                        "enable_thinking": False if a.chat else None,
                        "matcher": "legacy runtime matcher; preserve txt for revised post-hoc scoring",
                        "heads": a.heads,
                        "heads_sha256": hashlib.sha256(open(_path(a.heads), "rb").read()).hexdigest(),
                        "random_seed": 7,
                        "hf_id": loaded.spec.hf_id,
                        "checkpoint_revision": getattr(loaded.model.config, "_commit_hash", None),
                        "software": {"python": platform.python_version(),
                                     "torch": __import__("torch").__version__,
                                     "transformers": __import__("transformers").__version__}},
           "by_lang": {}}

    for T in a.targets:
        pool = pool_by_lang[T]
        rand_pool = random_pool_like(pool, loaded.spec.n_layers, loaded.spec.n_heads, seed=7)
        subset = [json.loads(l) for l in open(_path(a.subset_dir) / f"subset_{a.model}_{T}_failure.jsonl")]
        if a.limit:
            subset = subset[:a.limit]
        print(f"[sch] {T}: pool={len(pool)} heads | subset={len(subset)} EN-success/{T}-fail items", flush=True)

        def native_forms(item):
            labs = labels.get(item["chain_id"], {})
            forms = [labs.get(T)] + [s for s in rich.get(item.get("answer_qid", ""), []) if _script_ok(s, T)]
            return [f for f in forms if f]

        def scored(pred, item):
            pred = pred.replace("�", "").strip()
            labs = labels.get(item["chain_id"], {})
            ol = output_language(pred, labs)
            forms = native_forms(item)
            render = ol == T and any(_form_matches(_normalize(pred), _normalize(pred).split(), f) for f in forms)
            ent = any(_label_hit(pred, labs.get(L)) for L in LANGS if labs.get(L)) or \
                any(_label_hit(pred, al) for al in (item.get("answer_aliases") or [])[:40])
            return {"out": ol, "render": bool(render), "entity": bool(ent), "txt": pred}

        # first-token gold labels per language (first-token language adjudication)
        def ft_lang(item, tok):
            labs = labels.get(item["chain_id"], {})
            for L in LANGS:
                if labs.get(L) and first_answer_token_id(loaded, labs[L]) == tok:
                    return L
            return "other"

        rows, kept = [], 0
        for i, item in enumerate(subset):
            # untreated floor: 32-token decode; keep only still-failing, entity-free items
            floor = colon_decode(loaded, item["prompt"], a.max_new, chat=a.chat)
            if floor.lstrip().startswith("<think>"):
                raise RuntimeError("thinking-mode SCH floor detected despite --chat")
            f0 = scored(floor, item)
            if f0["render"] or f0["entity"]:
                continue
            kept += 1
            row = {"chain": item["chain_id"], "floor": f0}
            for al in a.alphas:
                amp = colon_decode(
                    loaded, item["prompt"], a.max_new, pool=pool, alpha=al,
                    chat=a.chat
                )
                rnd = colon_decode(
                    loaded, item["prompt"], a.max_new, pool=rand_pool, alpha=al,
                    chat=a.chat
                )
                if amp.lstrip().startswith("<think>") or rnd.lstrip().startswith("<think>"):
                    raise RuntimeError("thinking-mode SCH arm detected despite --chat")
                ft = int(colon_logits(loaded, item["prompt"], chat=a.chat).argmax()) \
                    if al == a.alphas[0] else None
                row[f"amp_a{al}"] = scored(amp, item)
                row[f"rand_a{al}"] = scored(rnd, item)
                if ft is not None:
                    row["ft_lang_unamped"] = ft_lang(item, ft)
            rows.append(row)
            if kept % 20 == 0:
                print(f"    {T} [{i + 1}/{len(subset)}] kept={kept}", flush=True)

        n = max(1, len(rows))
        res["by_lang"][T] = {"n_subset": len(subset), "n_genuine": len(rows), "items": rows}
        print(f"\n[sch] ===== {T} (genuine n={len(rows)} of {len(subset)}) =====", flush=True)
        print(f"  {'arm':>10} {'alpha':>5} {'RENDER-in-' + T:>12} {'entity_any':>11} {'out=en':>7} {'out=' + T:>7}", flush=True)
        for al in a.alphas:
            for arm in ("amp", "rand"):
                k = f"{arm}_a{al}"
                rr = sum(r[k]["render"] for r in rows) / n
                ee = sum(r[k]["entity"] for r in rows) / n
                oe = sum(r[k]["out"] == "en" for r in rows) / n
                ot = sum(r[k]["out"] == T for r in rows) / n
                print(f"  {arm:>10} {al:>5} {rr:>11.1%} {ee:>10.1%} {oe:>6.1%} {ot:>6.1%}", flush=True)
        (out_dir / f"sch_amplification_{a.model}_{a.set_type}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("[sch] DONE", flush=True)


if __name__ == "__main__":
    main()
