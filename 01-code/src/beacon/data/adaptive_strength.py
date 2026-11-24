"""Adaptive dose bracketing for energy-matched controls, used on calibration data only.

`fit_energy` finds the intervention increment whose measured energy best matches a target, with fixed
stopping rules for non-finite values, non-monotone responses, and saturation. `fresh_neuron_split`
builds a held-out validation split for strength fitting.
"""
import hashlib
import math

RULE=dict(initial_scales=[0,.125,.25,.5,1,2,4,8],max_scale=128,
          expansion_factor=2,bisection_steps=12,saturation_relative_gain=.01,
          saturation_consecutive_steps=2,nonmonotone='stop expansion; nearest finite observed norm',
          nonfinite='stop search; mark failure; keep finite measurements',
          validation_tuning=False)


# Expand the increment over a fixed geometric schedule until the target energy is bracketed, then
# bisect. Returns the finite observed increment whose energy is closest to the target (in log-norm
# ratio), together with the full search history and the reason the search stopped.
def fit_energy(target,evaluate,increment,steps=12):
    if not math.isfinite(target) or target<0 or increment<=0:raise ValueError('Invalid target/increment')
    history={};failure=None
    def measure(a):
        nonlocal failure
        if a not in history:
            e=float(evaluate(a))
            if not math.isfinite(e) or e<0:
                failure='nonfinite';return None
            history[a]=e
        return history[a]
    if measure(0)!=0:raise ValueError('Zero dose must have zero energy')
    if target==0:return dict(increment=0.,target_energy=0.,fitted_energy=0.,ratio=None,zero_target=True,stop='zero_target',history=[])
    previous=0.;plateau=0;stop='cap';bracket=None
    scales=RULE['initial_scales'][1:]+[16,32,64,128]
    for scale in scales:
        a=increment*scale;e=measure(a)
        if e is None:stop='nonfinite';break
        old=history[previous]
        if e+1e-6*max(1,old)<old:stop='nonmonotone';break
        if old<=target<=e:bracket=(previous,a);stop='bracketed';break
        if scale>8:
            plateau=plateau+1 if (e-old)/max(old,1e-30)<RULE['saturation_relative_gain'] else 0
            if plateau>=RULE['saturation_consecutive_steps']:stop='saturated';break
        previous=a
    if bracket:
        lo,hi=bracket
        for _ in range(steps):
            mid=(lo+hi)/2;e=measure(mid)
            if e is None:stop='nonfinite';break
            if not history[lo]-1e-6*max(1,history[lo])<=e<=history[hi]+1e-6*max(1,history[hi]):
                stop='nonmonotone';break
            if e<target:lo=mid
            else:hi=mid
    positive=[a for a,e in history.items() if e>0]
    best=min(positive,key=lambda a:(abs(math.log(math.sqrt(history[a]/target))),a)) if positive else 0.
    return dict(increment=best,target_energy=target,fitted_energy=history[best],ratio=math.sqrt(history[best]/target),
                zero_target=False,bracketed=bool(bracket),stop=stop,search_failed=stop!='bracketed',
                history=[dict(increment=a,energy=e) for a,e in sorted(history.items())])


# Select one chain per answer cluster (English answer label) that is disjoint from the evaluation
# chains and from the existing calibration/validation split; selection order is a fixed hash, so the
# split is deterministic. Also records any overlap with the neuron dose-calibration clusters.
def fresh_neuron_split(chains,frozen,previous):
    cluster=lambda c:chains[c]['answer_label_by_lang']['en']
    excluded={cluster(c) for c in frozen['eval_chains']}
    used={cluster(c) for k in ('calibration','validation') for c in previous['split'][k]}
    candidates={}
    for c in chains:
        if cluster(c) not in excluded|used and chains[c]['prompts'].get('es'):
            candidates.setdefault(cluster(c),[]).append(c)
    rank=lambda x:hashlib.sha256(('strength-v2/'+x).encode()).hexdigest()
    ids=[min(candidates[k],key=rank) for k in sorted(candidates,key=rank)]
    if len(ids)<5:raise ValueError('Fewer than five fresh validation clusters')
    return dict(calibration=previous['split']['calibration'],validation=ids,
                old_validation_not_reused=previous['split']['validation'],
                available_clusters=len(ids),cluster_by_chain={c:cluster(c) for c in chains},
                limitation='New to strength fitting/validation and disjoint from test; may overlap original neuron dose calibration entities',
                original_dose_calibration_overlap=sorted({cluster(c) for c in ids}&{cluster(c) for c in frozen['calib_chains']}))
