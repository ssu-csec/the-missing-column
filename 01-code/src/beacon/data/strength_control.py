"""Frozen splits, scalar calibration, and paired analysis (stdlib only)."""
import hashlib
import math
from .untreated_strata import cluster_interval

SEEDS = [7, 17, 27, 37, 47]
GRID = [0., .125, .25, .5, 1., 2., 4., 8.]


def content_positions(tokenizer, rendered_text, body, add_special_tokens, expected_ids):
    """Prompt positions whose characters overlap the user-message body.

    Chat templates add role headers (and, for Llama-3.1-Instruct, a dated system
    block) around the question; these positions are excluded. The offsets
    tokenization must reproduce the frozen prompt IDs exactly.
    """
    start = rendered_text.find(body)
    if start < 0 or rendered_text.find(body, start + 1) >= 0:
        raise ValueError('User message body not uniquely located in rendered prompt')
    end = start + len(body)
    enc = tokenizer(rendered_text, add_special_tokens=add_special_tokens, return_offsets_mapping=True)
    if list(enc['input_ids']) != list(expected_ids):
        raise ValueError('Offsets tokenization differs from frozen prompt IDs')
    positions = [i for i, (a, b) in enumerate(enc['offset_mapping']) if b > a and a < end and b > start]
    if not positions:
        raise ValueError('Empty user-message span')
    return positions


def split_clusters(chains, frozen, n_calib=30, n_valid=15):
    cluster = lambda c: chains[c]['answer_label_by_lang']['en']
    excluded = {cluster(c) for c in frozen['eval_chains'] + frozen['calib_chains']}
    rank = lambda value: hashlib.sha256(('strength-v1/' + value).encode()).hexdigest()
    pools = {}
    for c in chains:
        if cluster(c) not in excluded and chains[c]['prompts'].get('es'):
            pools.setdefault(cluster(c), []).append(c)
    order = sorted(pools, key=rank)
    if len(order) < n_calib + n_valid:
        raise ValueError(f'Only {len(order)} independent clusters; need {n_calib+n_valid}')
    reps = [min(pools[k], key=rank) for k in order]
    return dict(calibration=reps[:n_calib], validation=reps[n_calib:n_calib+n_valid],
                available_clusters=len(order), excluded_clusters=sorted(excluded),
                cluster_by_chain={c: cluster(c) for c in chains},
                discovery_entity_overlap='generic discovery corpus has no entity IDs; not certified',
                ordering='SHA256(strength-v1/label), one SHA256-ranked chain per cluster')


def fit_energy(target, evaluate, increment, steps=8):
    """evaluate returns mean squared energy. Never tunes on validation or outcomes."""
    history = {}
    def energy(a):
        if a not in history:
            e = float(evaluate(a))
            if not math.isfinite(e) or e < 0:
                raise ValueError('Nonfinite/negative measured energy')
            history[a] = e
        return history[a]
    if not math.isfinite(target) or target < 0:
        raise ValueError('Invalid target energy')
    for scale in GRID:
        energy(increment * scale)
    if target == 0:
        return dict(increment=0., target_energy=target, zero_target=True,
                    history=[dict(increment=a, energy=e) for a,e in sorted(history.items())])
    xs = sorted(history)
    monotone = all(history[a] <= history[b] + 1e-6*max(1.,history[b])
                   for a,b in zip(xs,xs[1:]))
    bracket = next(((a,b) for a,b in zip(xs,xs[1:])
                    if history[a] <= target <= history[b]), None)
    if monotone and bracket:
        lo,hi = bracket
        for _ in range(steps):
            mid = (lo+hi)/2
            e = energy(mid)
            if not history[lo]-1e-6*max(1.,history[lo]) <= e <= history[hi]+1e-6*max(1.,history[hi]):
                monotone = False
                break
            if e < target: lo=mid
            else: hi=mid
    candidates = list(history) if monotone else [increment*s for s in GRID]
    loss = lambda a: abs(math.log(math.sqrt(history[a]/target))) if history[a] > 0 else float('inf')
    best = min(candidates, key=lambda a: (loss(a), a))
    return dict(increment=best, target_energy=target, fitted_energy=history[best],
                ratio=math.sqrt(history[best]/target), monotone=monotone,
                bracketed=bracket is not None, zero_target=False,
                history=[dict(increment=a,energy=e) for a,e in sorted(history.items())])


def balance(target, control):
    if set(target) != set(control):
        raise ValueError('Layer coverage differs')
    total=sum(target.values()); rt=sum(control.values())
    ratios={k:math.sqrt(control[k]/e) if e>0 else None for k,e in target.items()}
    ratio=math.sqrt(rt/total) if total else None
    coverage=sum(target[k] for k,v in ratios.items() if v is not None and .8<=v<=1.2)/total if total else 0.
    return dict(total_ratio=ratio, layer_ratios=ratios, matched_energy_fraction=coverage,
                zero_target_layers=[k for k,e in target.items() if e==0],
                achieved=ratio is not None and .9<=ratio<=1.1 and coverage>=.9)


def analyze(records, plan, draws=10000):
    lookup={(r['chain'],r['target'],r['arm']):r for r in records}
    arms=['U','I']+[f'{kind}{s}' for kind in ('R','M') for s in plan['seeds']]
    expected={(c,t,a) for c in plan['evaluation'] for t in plan['targets'] for a in arms}
    if set(lookup)!=expected or len(lookup)!=len(records):
        raise ValueError('Missing/extra/duplicate test outputs')
    result={}
    for t in plan['targets']:
        cs=plan['evaluation']; clusters=[plan['split']['cluster_by_chain'][c] for c in cs]
        get=lambda c,a,m: int(lookup[c,t,a]['score']['content'])-int(lookup[c,t,a]['score']['form']) if m=='delta' else int(lookup[c,t,a]['score'][m])
        def estimate(values):
            ci=cluster_interval(values,clusters,draws=draws)
            return dict(pp=100*sum(values)/len(values),ci95_pp=[100*x for x in ci])
        effects={}
        for metric in ('content','form','delta','degenerate'):
            selected=[get(c,'I',metric) for c in cs]
            controls={k:[sum(get(c,f'{k}{s}',metric) for s in plan['seeds'])/len(plan['seeds']) for c in cs] for k in ('R','M')}
            effects[metric]={k:estimate([i-r for i,r in zip(selected,controls[k])]) for k in ('R','M')}
            effects[metric]['change_in_effect_M_minus_R']=estimate([r-m for r,m in zip(controls['R'],controls['M'])])
        by_arm={a:{m:100*sum(get(c,a,m) for c in cs)/len(cs) for m in ('content','form','degenerate')} for a in arms}
        result[t]=dict(n=len(cs),clusters=len(set(clusters)),effects=effects,arm_rates=by_arm)
    return dict(results=result,bootstrap_draws=draws,
                scope='Descriptive paired answer-cluster intervals, conditional on fixed calibration and five groups; no multiplicity correction')
