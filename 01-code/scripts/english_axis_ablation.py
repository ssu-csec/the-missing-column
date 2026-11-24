"""HUB-vs-AGNOSTIC --- STEP 2, the DECISIVE causal test (red-teamed). The one experiment Wendler's mean-patching
STRUCTURALLY CANNOT run: English-subspace ablation on NON-ENGLISH -> NON-ENGLISH two-hop transfer.

Logic: patch a concept from a non-English SOURCE S into a non-English TARGET T (S!=T, e.g. ko->es), a transfer
that NEVER contains English. Then ABLATE the English axis from the donor. If removing English breaks the transfer
MORE than removing a random equal-rank subspace, English is a NECESSARY intermediary = HUB. If ablating English is
no worse than random = AGNOSTIC. Metric = entity_present (language-agnostic alias union) so no single unembedding
column decides. centering_index = [succ(raw) - succ(EnOut)] / [succ(raw) - succ(RandOut)]; 0=agnostic, 1=English-
routed, bootstrap CI over concepts. HONEST: may confirm agnostic; likely partial (>0,<1), es-pairs only; report scope.

    python scripts/english_axis_ablation.py --model llama31_70_it --S ko --T es --layers 40,52,64 --k 4 --chat --n 80
    python scripts/english_axis_ablation.py --verify           # $0 CPU arithmetic smoke, no model
"""
from __future__ import annotations

import argparse
from pathlib import Path
import importlib.util
import json

import torch

from beacon.config import REPO_ROOT, load_configs, load_env
from beacon.data.evaluate import entity_match
from beacon.data.langid import output_language
from beacon.data.prompting import encode
from beacon.models import load_model
from beacon.pipeline.patching import _decoder_layers, _unwrap_layer_out

ROOT = REPO_ROOT.parent  # repository root; REPO_ROOT is 01-code/


def _path(p):
    """Resolve a command-line path against the repository root (absolute paths pass through)."""
    p = Path(p)
    return p if p.is_absolute() else ROOT / p

_rs = importlib.util.spec_from_file_location("rs", str(REPO_ROOT / "scripts/solvability_rescore.py"))
rs = importlib.util.module_from_spec(_rs); _rs.loader.exec_module(rs)
LANGS = ["en", "es", "ko", "zh", "ja"]

def _entity_present(pred, meta):
    union = []
    for al in meta["answer_aliases_by_lang"].values():
        union += al or []
    for lab in meta["answer_label_by_lang"].values():
        if lab:
            union.append(lab)
    return bool(entity_match(pred, union[0] if union else "", union))

def proj_out(h, E):
    """remove the span(E) component: (I - E E^T) h, E columns orthonormal."""
    return h - E @ (E.t() @ h)

def rand_basis(d, k, gen):
    R = torch.randn(d, k, generator=gen)
    return torch.linalg.qr(R)[0][:, :k]

def _verify():
    """$0 CPU arithmetic check of projection + norm-match + LangOut axis + multi-R + the two decisive CIs, no model."""
    torch.manual_seed(0); d, k = 64, 4
    E = torch.linalg.qr(torch.randn(d, k))[0][:, :k]
    h = torch.randn(d)
    ho = proj_out(h, E)
    assert float((E.t() @ ho).norm()) < 1e-4, "E-component not removed"
    assert ho.norm() < h.norm(), "projection did not shrink norm"
    g = torch.Generator().manual_seed(1)
    Rs = [rand_basis(d, k, g) for _ in range(8)]
    assert len(Rs) == 8 and all(abs(float((R.t() @ R - torch.eye(k)).abs().max())) < 1e-4 for R in Rs), "R not orthonormal"
    E_S = torch.linalg.qr(torch.randn(d, k))[0][:, :k]                                  
    hlo = proj_out(h, E_S)
    tn = 10.0
    a = h * tn / h.norm(); b = ho * tn / ho.norm(); c = hlo * tn / hlo.norm()
    for R in Rs:
        hr = proj_out(h, R)
        assert abs(float((hr * tn / hr.norm()).norm()) - tn) < 1e-3, "RandOut norm-match failed"
    assert abs(float(a.norm()) - tn) < 1e-3 and abs(float(b.norm()) - tn) < 1e-3 and abs(float(c.norm()) - tn) < 1e-3, "norm-match failed"

    gen = torch.Generator().manual_seed(2)
    rows = [{"enout": (i % 2 == 0), "langout": (i % 5 == 0), "rand_mean": 0.15} for i in range(120)]
    def boot(delta):
        ds = []
        for _ in range(2000):
            idx = torch.randint(0, len(rows), (len(rows),), generator=gen).tolist()
            ds.append(sum(delta(rows[j]) for j in idx) / len(rows))
        ds.sort(); return [ds[50], ds[1949]]
    ci_evr = boot(lambda r: int(r["enout"]) - r["rand_mean"])
    ci_elo = boot(lambda r: int(r["enout"]) - int(r["langout"]))
    assert ci_evr[0] > 0 and ci_elo[0] > 0, "toy PASS case should have both CI lower bounds > 0"
    print(f"[hub2] VERIFY OK: proj removes E-subspace, LangOut axis + 8 R's norm-matched, "
          f"decisive CIs plumbed (toy EnOut-Rand CI={[round(x,3) for x in ci_evr]}, EnOut-Lang CI={[round(x,3) for x in ci_elo]}).", flush=True)

@torch.no_grad()
def _readout(loaded, prompt, chat):
    ids = encode(loaded, prompt, chat)
    g = loaded.model.generate(input_ids=ids, max_new_tokens=1, do_sample=False, pad_token_id=loaded.tokenizer.eos_token_id)
    return g[:, :ids.shape[1] + 1], ids.shape[1]

@torch.no_grad()
def _resid_at(loaded, prompt, chat, layer):
    full, col = _readout(loaded, prompt, chat)
    hs = loaded.model(input_ids=full, use_cache=False, output_hidden_states=True).hidden_states
    return hs[layer + 1][0, col].detach().float().cpu()

@torch.no_grad()
def _patched_decode(loaded, prompt, chat, layer, donor, max_new=10):
    dl = _decoder_layers(loaded.model)
    ids = encode(loaded, prompt, chat)
    g = loaded.model.generate(input_ids=ids, max_new_tokens=1, do_sample=False, pad_token_id=loaded.tokenizer.eos_token_id)
    ids = g[:, :ids.shape[1] + 1]; col0 = ids.shape[1] - 1
    out = []
    for _ in range(max_new):
        def hook(_m, _i, o):
            h, rest = _unwrap_layer_out(o); h = h.clone()
            h[0, col0] = donor.to(h.device, h.dtype)
            return (h,) + tuple(rest) if rest is not None else h
        hd = dl[int(layer)].register_forward_hook(hook)
        try:
            logits = loaded.model(input_ids=ids, use_cache=False).logits
        finally:
            hd.remove()
        nxt = int(logits[0, -1].argmax()); out.append(nxt)
        if nxt == loaded.tokenizer.eos_token_id:
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device=ids.device)], dim=1)
    return loaded.tokenizer.decode(out, skip_special_tokens=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--model"); ap.add_argument("--S", default="ko"); ap.add_argument("--T", default="es")
    ap.add_argument("--layers", default="40,52,64"); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n", type=int, default=80); ap.add_argument("--n-axis", type=int, default=120)
    ap.add_argument("--n-rand", type=int, default=8)                                                                    
    ap.add_argument("--boot", type=int, default=3000)                                                       
    ap.add_argument("--min-headroom", type=float, default=0.25)                                                       
    ap.add_argument("--chains", default="02-data/inputs/chain_lang_prompts.json")
    ap.add_argument("--chat", action="store_true"); ap.add_argument("--out", default="02-data/results/rivals")
    a = ap.parse_args()
    if a.verify:
        _verify(); return
    assert a.model, "--model required"
    load_env(); cfg = load_configs()
    C = json.load(open(_path(a.chains)))
    rs.C = C                                                                                          
    out_dir = _path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    S, T = a.S, a.T
    layers = [int(x) for x in a.layers.split(",")]
    print(f"[hub2] loading {a.model} | S={S} T={T} layers={layers} k={a.k}", flush=True)
    loaded = load_model(a.model, cfg, framework="hf_hooks", attn_implementation="eager")

    all_ch = [c for c in C if all(C[c]["prompts"].get(L) and C[c]["answer_label_by_lang"].get(L) for L in LANGS)]
    axis_ch = all_ch[:a.n_axis]; xfer_ch = all_ch[a.n_axis:a.n_axis + a.n]
    print(f"[hub2] axis concepts={len(axis_ch)} transfer concepts={len(xfer_ch)}", flush=True)
    res = {"model": a.model, "S": S, "T": T, "k": a.k, "by_layer": {}}
    gen = torch.Generator().manual_seed(0)

    def _axis(H, pivot):
        """top-k PCA of {h_pivot(c) - mean_{L!=pivot} h_L(c)} over axis concepts; orthonormal [d,k]."""
        diffs = []
        for c in axis_ch:
            hp = H[c][pivot]
            hno = torch.stack([H[c][L2] for L2 in LANGS if L2 != pivot]).mean(0)
            diffs.append(hp - hno)
        Dm = torch.stack(diffs); Dm = Dm - Dm.mean(0)
        Ax = torch.linalg.svd(Dm, full_matrices=False)[2][:a.k].t()
        return torch.linalg.qr(Ax)[0][:, :a.k]

    def _boot(rows, delta):
        ds = []
        n = len(rows)
        for _ in range(a.boot):
            idx = torch.randint(0, n, (n,), generator=gen).tolist()
            ds.append(sum(delta(rows[j]) for j in idx) / n)
        ds.sort()
        return [ds[int(0.025 * len(ds))], ds[int(0.975 * len(ds))]]

    for L in layers:

        H = {c: {L2: _resid_at(loaded, C[c]["prompts"][L2], a.chat, L) for L2 in LANGS} for c in axis_ch}
        E = _axis(H, "en")                                                                   
        E_S = _axis(H, S)                                                                                                
        d = E.shape[0]
        Rs = [rand_basis(d, a.k, gen) for _ in range(a.n_rand)]                                                     

        rows = []
        for i, c in enumerate(xfer_ch):
            meta = C[c]
            hS = _resid_at(loaded, C[c]["prompts"][S], a.chat, L)
            tn = float(hS.norm())
            def scaled(v):
                return (v * tn / (v.norm() + 1e-6)).to(loaded.device)                                                      
            donor_raw = scaled(hS)
            donor_en = scaled(proj_out(hS, E))
            donor_lo = scaled(proj_out(hS, E_S))
            cd = xfer_ch[(i + 1) % len(xfer_ch)]
            donor_diff = scaled(_resid_at(loaded, C[cd]["prompts"][S], a.chat, L))
            o_floor = _readout(loaded, C[c]["prompts"][T], a.chat)                                  
            floor_txt = loaded.tokenizer.decode(o_floor[0][0, o_floor[1]:], skip_special_tokens=True)
            o_raw = _patched_decode(loaded, C[c]["prompts"][T], a.chat, L, donor_raw)
            o_en = _patched_decode(loaded, C[c]["prompts"][T], a.chat, L, donor_en)
            o_lo = _patched_decode(loaded, C[c]["prompts"][T], a.chat, L, donor_lo)
            o_diff = _patched_decode(loaded, C[c]["prompts"][T], a.chat, L, donor_diff)
            rand_bools = []
            for R in Rs:
                rand_bools.append(_entity_present(_patched_decode(loaded, C[c]["prompts"][T], a.chat, L, scaled(proj_out(hS, R))), meta))
            rows.append({"chain": c,
                         "raw": _entity_present(o_raw, meta), "enout": _entity_present(o_en, meta),
                         "langout": _entity_present(o_lo, meta), "floor": _entity_present(floor_txt, meta),
                         "diff": _entity_present(o_diff, meta),
                         "rand": [bool(b) for b in rand_bools],
                         "rand_mean": sum(int(b) for b in rand_bools) / max(1, len(rand_bools))})
            if (i + 1) % 20 == 0:
                print(f"    L{L} [{i+1}/{len(xfer_ch)}]", flush=True)
        n = max(1, len(rows))
        rate = {kk: sum(int(bool(r[kk])) for r in rows) / n for kk in ("raw", "enout", "langout", "floor", "diff")}
        rate["randout"] = sum(r["rand_mean"] for r in rows) / n                                                          
        rand_rate_per_R = [sum(int(r["rand"][j]) for r in rows) / n for j in range(a.n_rand)]                    

                                                                                                  

                                                                                                                     
        ci_evr = _boot(rows, lambda r: int(bool(r["enout"])) - r["rand_mean"])
        ci_elo = _boot(rows, lambda r: int(bool(r["enout"])) - int(bool(r["langout"])))
        headroom = rate["raw"] - rate["floor"]
        passed = (headroom >= a.min_headroom) and (ci_evr[0] > 0) and (ci_elo[0] > 0)
        res["by_layer"][str(L)] = {"rate": rate, "n": n, "n_rand": a.n_rand, "rand_rate_per_R": rand_rate_per_R,
                                   "headroom": headroom,
                                   "EnOut_minus_raw": rate["enout"] - rate["raw"],
                                   "EnOut_minus_RandOut": rate["enout"] - rate["randout"], "ci_EnOut_RandOut": ci_evr,
                                   "EnOut_minus_LangOut": rate["enout"] - rate["langout"], "ci_EnOut_LangOut": ci_elo,
                                   "passed": bool(passed), "items": rows}
        (out_dir / f"hubablation_{a.model}_{S}_{T}.json").write_text(json.dumps(res, indent=2))
        verdict = ("STOP:NO-HEADROOM (raw-floor<%.2f)" % a.min_headroom if headroom < a.min_headroom
                   else "PASS:INTERFERENCE (En>Rand AND En>Lang, both CI>0)" if (ci_evr[0] > 0 and ci_elo[0] > 0)
                   else "STOP:En~=Rand (EnOut-RandOut CI incl 0)" if ci_evr[0] <= 0
                   else "STOP:En~=Lang (EnOut-LangOut CI incl 0; any dominant-lang axis helps)")
        print(f"[hub2] L{L}: floor={rate['floor']:.0%} raw={rate['raw']:.0%} (headroom={headroom:+.0%}) "
              f"EnOut={rate['enout']:.0%} RandOut={rate['randout']:.0%} LangOut={rate['langout']:.0%} diff={rate['diff']:.0%} | "
              f"EnOut-Rand CI[{ci_evr[0]:+.2f},{ci_evr[1]:+.2f}] EnOut-Lang CI[{ci_elo[0]:+.2f},{ci_elo[1]:+.2f}] -> {verdict}", flush=True)
    print("[hub2] DONE", flush=True)

if __name__ == "__main__":
    main()
