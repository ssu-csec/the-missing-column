#!/usr/bin/env python3
"""Output-matched random control for language-neuron amplification, zh/ko; all draws/outcomes retained.

For each random group (seed) and target, one increment applied to every layer of the group is fitted so that
the mean KL divergence of the next-token distribution at the generation position, relative to the frozen
Spanish-suppression-only state S, equals that of the selected intervention on 30 calibration clusters.
The fit uses a nonmonotone-safe log-grid search (KL is not monotone in dose); 15 held-out clusters validate it.
Arms: U, S, I, five construction-matched random groups R at the selected dose and five output-matched random
groups L. The plan and fitted increments are fixed before test generation.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import math
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'01-code';RESULTS=ROOT/'02-data/results'
sys.path.insert(0,str(CODE/'src'))
from beacon.data.strength_control import SEEDS,split_clusters
from beacon.data.tokenizer_checks import validate_pilot_files,tokenizer_identity
from beacon.data.intervention_cohorts import save_json

ROSTER=('gemma4_31b','gemma4_31b_it','llama31_70','llama31_70_it','qwen36_27b')
BALANCE=[.9,1.1]  # on sqrt(mean KL ratio), parallel to the residual RMS-ratio criterion
# Output KL is not monotone in the random-group increment, so a bracketing search can stop at a local dip.
# The search therefore evaluates a dense log grid of relative scales and refines around the best point
# without assuming monotonicity.
GRID_RULE=dict(relative_scales='2**(k/4), k=-8..28 (0.25x..128x the original increment)',
               selection='minimum |log(mean KL / target)| over all evaluated doses',
               refinement='6 bisection steps inside the bracket formed with the neighbouring grid point '
                          'on the other side of the target, if any; best evaluated dose kept',
               supersedes='v1 adaptive bracketing (stopped as nonmonotone in every v1 fit)')


def fit_grid(target,evaluate,increment,steps=6):
    """Nonmonotone-safe scalar fit of evaluate(dose) to target; returns every evaluated point."""
    if not math.isfinite(target) or target<=0 or increment<=0:raise ValueError('Invalid target/increment')
    history={}
    def measure(d):
        if d not in history:
            v=float(evaluate(d))
            if not math.isfinite(v) or v<0:raise ValueError('Nonfinite KL')
            history[d]=v
        return history[d]
    grid=[increment*2**(k/4) for k in range(-8,29)]
    for d in grid:measure(d)
    loss=lambda d:abs(math.log(history[d]/target)) if history[d]>0 else float('inf')
    best=min(grid,key=lambda d:(loss(d),d));i=grid.index(best)
    bracket=None
    for j in (i-1,i+1):
        if 0<=j<len(grid) and (history[grid[j]]-target)*(history[best]-target)<0:
            bracket=tuple(sorted((best,grid[j])));break
    if bracket:
        lo,hi=bracket;up=history[hi]>history[lo]
        for _ in range(steps):
            mid=(lo+hi)/2;v=measure(mid)
            if (v<target)==up:lo=mid
            else:hi=mid
    best=min(history,key=lambda d:(loss(d),d))
    return dict(increment=best,target_energy=target,fitted_energy=history[best],ratio=math.sqrt(history[best]/target),
                bracketed=bracket is not None,stop='grid_refined' if bracket else 'grid_nearest',search_failed=False,
                history=[dict(increment=d,energy=v) for d,v in sorted(history.items())])


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def log(s):print(s,flush=True)


def prepare(a):
    f=read(a.frozen);g=read(a.groups);pilot=validate_pilot_files(a.pilot)
    if f['model'] not in ROSTER:raise ValueError('Unknown checkpoint')
    if g['model']!=f['model'] or pilot['model']!=f['model'] or g['chat']!=pilot['chat']:raise ValueError('Model/chat mismatch')
    if g['frozen_sha256']!=sha(a.frozen) or g['model_revision']!=pilot['resolved_revision']:raise ValueError('Group provenance')
    if g['tokenizer_identity_sha256']!=pilot['identity_sha256']:raise ValueError('Tokenizer identity')
    expected={'frozen_run':a.frozen,'pairs':ROOT/'02-data/inputs/parallel_pairs.jsonl',
              'chains':ROOT/'02-data/inputs/chain_lang_prompts.json','prompting':CODE/'src/beacon/data/prompting.py',
              'tokenizer_checks':CODE/'src/beacon/data/tokenizer_checks.py','neuron_discovery':CODE/'scripts/language_neurons.py'}
    if any(pilot['input_sha256'][k]!=sha(p) for k,p in expected.items()):raise ValueError('Changed pilot source')
    if g['pairs_sha256']!=sha(expected['pairs']) or g['suppression']!=g['selected']['es']:raise ValueError('Discovery/suppression mismatch')
    if any(str(s) not in g['random'] for s in SEEDS):raise ValueError('Missing fixed random draw')
    chains=read(expected['chains']);split=split_clusters(chains,f)
    paths=[a.frozen,a.groups,a.pilot,*expected.values(),Path(__file__),
           CODE/'src/beacon/standard_strength_runtime.py',CODE/'src/beacon/data/strength_control.py',
           CODE/'src/beacon/data/adaptive_strength.py',CODE/'src/beacon/data/untreated_strata.py',
           CODE/'scripts/language_neuron_groups.py',CODE/'src/beacon/models.py',CODE/'configs/models.yaml',
           ROOT/'01-code/analysis'/'paired_effects.py',ROOT/'01-code/analysis'/'verify_numbers.py',ROOT/'01-code/analysis'/'degenerate_outputs.py']
    plan=dict(schema_version=1,mode='logit_matched_prospective',model=f['model'],chat=pilot['chat'],
        targets=['ko','zh'],seeds=SEEDS,evaluation=f['eval_chains'],split=split,
        original_increment={t:f['best'][t]['amp'] for t in ('ko','zh')},
        suppression={t:f['best'][t]['supp'] for t in ('ko','zh')},
        max_new_tokens=32,search_rule=GRID_RULE,balance=dict(sqrt_kl_ratio=BALANCE),
        expected_records=len(f['eval_chains'])*26,expected_physical_generations=len(f['eval_chains'])*26,
        model_revision=pilot['resolved_revision'],pilot_sha256=sha(a.pilot),
        hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths},
        measurement='KL(arm || S) of the next-token distribution at the last prompt position (generation position), '
                    'float64 over the full vocabulary; one increment shared by all layers of each random group; '
                    'mean over calibration prompts fitted by a nonmonotone-safe log-grid search with local refinement',
        selection='All original eval IDs; no rescreening; all five draws retained irrespective of matching; coefficients frozen before test generations',
        scope='Output-level functional matching at the generation position only; not full-trajectory equivalence')
    if f['protocol']['max_new']!=32 or not plan['evaluation']:raise ValueError('Unexpected evaluation')
    return plan,f,g,pilot,chains


def analyze(rows,plan,draws=10000):
    from beacon.data.untreated_strata import cluster_interval
    index={(r['chain'],r['target'],r['arm']):r for r in rows}
    arms=['U','S','I']+[f'{k}{s}' for k in ('R','L') for s in SEEDS]
    if set(index)!={(c,t,a) for c in plan['evaluation'] for t in plan['targets'] for a in arms} or len(index)!=len(rows):
        raise ValueError('Missing/extra/duplicate test outputs')
    out={}
    for t in plan['targets']:
        cs=plan['evaluation'];clusters=[plan['split']['cluster_by_chain'][c] for c in cs]
        def value(c,arm,metric):
            s=index[c,t,arm]['score']
            return int(s['content'])-int(s['form']) if metric=='delta' else int(s[metric])
        def estimate(values):
            ci=cluster_interval(values,clusters,draws=draws)
            return dict(pp=100*sum(values)/len(values),ci95_pp=[100*x for x in ci])
        effects={}
        for metric in ('content','form','delta','degenerate'):
            sel=[value(c,'I',metric) for c in cs]
            ctrl={k:[sum(value(c,f'{k}{s}',metric) for s in SEEDS)/len(SEEDS) for c in cs] for k in ('R','L')}
            e={k:estimate([i-r for i,r in zip(sel,ctrl[k])]) for k in ('R','L')}
            e['change_in_effect_L_minus_R']=estimate([r-l for r,l in zip(ctrl['R'],ctrl['L'])])
            for ref in ('U','S'):e[ref]=estimate([i-value(c,ref,metric) for i,c in zip(sel,cs)])
            effects[metric]=e
        rates={a:{m:100*sum(value(c,a,m) for c in cs)/len(cs) for m in ('content','form','degenerate')} for a in arms}
        out[t]=dict(n=len(cs),clusters=len(set(clusters)),effects=effects,arm_rates=rates)
    return dict(results=out,bootstrap_draws=draws,
                scope='Descriptive paired answer-cluster intervals, conditional on fixed calibration and five groups; no multiplicity correction')


def execute(a,plan,f,g,pilot,chains,manifest):
    import torch
    from transformers import AutoTokenizer
    from beacon.models import load_model
    from beacon.config import load_configs
    from beacon.data.prompting import encode
    from beacon.standard_strength_runtime import validate_layers,intervention,synthetic_test
    synthetic_test()
    if not torch.cuda.is_available():raise ValueError('GPU required')
    if importlib.metadata.version('torch').split('+')[0]!='2.8.0':raise ValueError('Expected torch 2.8.0')
    for k in ('transformers','tokenizers'):
        if importlib.metadata.version(k)!=pilot['versions'][k]:raise ValueError('Pilot runtime changed')
    tokenizer=AutoTokenizer.from_pretrained(a.pilot.parent/'snapshot',local_files_only=True)
    if tokenizer_identity(tokenizer)['sha256']!=pilot['identity_sha256']:raise ValueError('Tokenizer mismatch')
    config=load_configs()
    if config.models['models'][plan['model']]['hf_id']!=pilot['repo_id']:raise ValueError('Model source')
    log('[load] '+plan['model'])
    loaded=load_model(plan['model'],config,framework='hf_hooks',attn_implementation='eager',revision=plan['model_revision'],tokenizer=tokenizer)
    loaded.model.eval()
    neuron_module=module('output_matched_neuron_module',CODE/'scripts/language_neurons.py')
    runner=module('logit_group_validation',CODE/'scripts/language_neuron_groups.py')
    g=runner.normalize_and_validate_groups(g,loaded,neuron_module,list(f['eval']),SEEDS)
    layers=neuron_module._decoder_layers(loaded.model);validate_layers(layers)
    if getattr(loaded.model.config,'_commit_hash',None) not in (None,plan['model_revision']):raise ValueError('Model commit mismatch')
    actual=loaded.model.generation_config.to_dict();expected_config=read(a.pilot.parent/'effective_generation_config.json')
    for k in ('eos_token_id','bos_token_id','repetition_penalty','no_repeat_ngram_size','min_length','min_new_tokens',
              'forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens','bad_words_ids','stop_strings'):
        if actual.get(k)!=expected_config.get(k):raise ValueError('Generation config mismatch '+k)
    expected={};pilot_rows=[json.loads(x) for x in (a.pilot.parent/'prompt_tokens.jsonl').read_text().splitlines()]
    for r in pilot_rows:
        if encode(loaded,r['text'],plan['chat'])[0].tolist()!=r['input_ids']:raise ValueError('Prompt tensor mismatch')
        if r['source']=='chains' and r['language']=='es':expected[r['key']]=r['input_ids']
    def tokens(c):
        ids=encode(loaded,chains[c]['prompts']['es'],plan['chat'])
        if ids[0].tolist()!=expected[c]:raise ValueError('Prompt changed')
        return ids
    manifest['runtime']=dict(versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','accelerate','huggingface_hub')},
        cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],dtype=str(loaded.model.dtype),
        generation_config=actual,model_revision=plan['model_revision'],prompt_tensor_checks=len(pilot_rows))
    save_json(a.out/'manifest.json',manifest);save_json(a.out/'groups.json',g)
    save_json(a.out/'evaluation_items.json',{c:chains[c] for c in plan['evaluation']})
    (a.out/'block_source.py.txt').write_text(inspect.getsource(type(layers[0])))

    @torch.inference_mode()
    def last_logprobs(c,groups,inc,factor):
        with intervention(layers,groups,inc,g['suppression'],factor):
            logits=loaded.model(input_ids=tokens(c),use_cache=False,logits_to_keep=1).logits[0,-1]
        lp=torch.log_softmax(logits.double(),-1)
        if not torch.isfinite(lp).all():raise ValueError('Nonfinite logits')
        return lp
    def kl(lp,base):
        return float((lp.exp()*(lp-base)).sum().clamp_min(0))
    calibration={};validation={}
    for t in plan['targets']:
        factor=plan['suppression'][t];amp=plan['original_increment'][t]
        sel=g['selected'][t]
        base={s:{c:last_logprobs(c,{},{},factor) for c in plan['split'][s]} for s in ('calibration','validation')}
        log(f'[baseline] {t} S logprobs for {sum(len(v) for v in base.values())} prompts')
        def mean_kl(split,groups,dose):
            inc={l:dose for l in groups}
            return sum(kl(last_logprobs(c,groups,inc,factor),base[split][c]) for c in plan['split'][split])/len(plan['split'][split])
        zero_check=max(kl(last_logprobs(c,sel,{l:0. for l in sel},factor),base['calibration'][c]) for c in plan['split']['calibration'][:3])
        if zero_check>1e-6:raise ValueError(f'Zero increment changes logits: KL={zero_check}')
        target=mean_kl('calibration',sel,amp)
        val_I=mean_kl('validation',sel,amp)
        log(f'[target] {t} calibration KL(I||S)={target:.6g} validation={val_I:.6g}')
        calibration[t]={'target_kl':target,'zero_increment_max_kl':zero_check,'seeds':{}};validation[t]={'I':val_I,'seeds':{}}
        for seed in SEEDS:
            rand=g['random'][str(seed)][t]
            fit=fit_grid(target,lambda dose:mean_kl('calibration',rand,dose),amp)
            fit['ratio_definition']='sqrt(fitted KL / target KL)'
            calibration[t]['seeds'][str(seed)]=fit
            R=mean_kl('validation',rand,amp);L=mean_kl('validation',rand,fit['increment'])
            ratio=math.sqrt(L/val_I) if val_I>0 else None
            validation[t]['seeds'][str(seed)]=dict(R=R,L=L,sqrt_ratio_L=ratio,sqrt_ratio_R=math.sqrt(R/val_I) if val_I>0 else None,
                achieved=ratio is not None and BALANCE[0]<=ratio<=BALANCE[1])
            save_json(a.out/'calibration.json',calibration);save_json(a.out/'validation.json',validation)
            log(f'[fit] {t} seed={seed} increment={fit["increment"]:.5g} stop={fit["stop"]} '
                f'val KL I={val_I:.4g} R={R:.4g} L={L:.4g} sqrt_ratio={ratio if ratio is None else round(ratio,3)}')
    save_json(a.out/'calibration_lock.json',dict(calibration_sha256=sha(a.out/'calibration.json'),validation_sha256=sha(a.out/'validation.json'),
        frozen_before_test_generation=True,validation_used_for_tuning=False))

    arms=['U','S','I']+[f'{k}{s}' for k in ('R','L') for s in SEEDS]
    def settings(t,arm):
        if arm in ('U','S'):return {},{},1. if arm=='U' else plan['suppression'][t]
        groups=g['selected'][t] if arm=='I' else g['random'][arm[1:]][t]
        dose=calibration[t]['seeds'][arm[1:]]['increment'] if arm.startswith('L') else plan['original_increment'][t]
        return groups,{l:dose for l in groups},plan['suppression'][t]
    @torch.inference_mode()
    def decode(c,t,arm,trace=True,max_new=32):
        ids=tokens(c);groups,inc,factor=settings(t,arm)
        with intervention(layers,groups,inc,g['suppression'],factor,trace) as trajectory:
            output=loaded.model.generate(input_ids=ids,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        tok=output[0,ids.shape[1]:].tolist();eos=actual['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
        text=tokenizer.decode(tok,skip_special_tokens=True)
        return dict(text=text,text_with_specials=tokenizer.decode(tok,skip_special_tokens=False),token_ids=tok,
            prompt_token_ids=ids[0].tolist(),prompt_sha256=hashlib.sha256(chains[c]['prompts']['es'].encode()).hexdigest(),
            n_generated_tokens=len(tok),stop_reason='eos' if tok and tok[-1] in eos else 'max_new_tokens' if len(tok)==max_new else 'other'),trajectory
    plumbing={}
    for t in plan['targets']:
        c=plan['split']['validation'][0];groups,inc,factor=settings(t,'L7')
        with torch.inference_mode(),intervention(layers,groups,inc,g['suppression'],factor):
            ids=tokens(c);steps=[]
            for _ in range(3):
                nxt=loaded.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1].argmax(-1,keepdim=True)
                steps.append(int(nxt.item()));ids=torch.cat([ids,nxt],1)
                eos=actual['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
                if steps[-1] in eos:break
        cached=decode(c,t,'L7',False,3)[0]['token_ids'];traced=decode(c,t,'L7',True,3)[0]['token_ids']
        if cached!=steps or cached!=traced:raise ValueError('Cached/full/telemetry parity failure')
        with torch.inference_mode(),intervention(layers,groups,{l:0. for l in groups},g['suppression'],factor):
            zero=loaded.model.generate(input_ids=tokens(c),max_new_tokens=3,do_sample=False,pad_token_id=tokenizer.eos_token_id)[0,tokens(c).shape[1]:].tolist()
        if zero!=decode(c,t,'S',False,3)[0]['token_ids']:raise ValueError('Zero-dose vs S mismatch')
        plumbing[t]=dict(cached_full_parity=True,telemetry_parity=True,zero_increment_equals_S=True,
                         zero_increment_max_kl=calibration[t]['zero_increment_max_kl'])
    save_json(a.out/'plumbing_checks.json',plumbing)
    scoring=module('logit_scoring',ROOT/'01-code/analysis'/'paired_effects.py').V;deg=module('logit_deg',ROOT/'01-code/analysis'/'degenerate_outputs.py')
    if scoring.MATCHER_POLICY!='attribution_strict':raise ValueError('Locked matcher required')
    rows=[]
    with (a.out/'generations.jsonl').open('x') as journal,(a.out/'trajectory.jsonl').open('x') as tracefile:
        for i,c in enumerate(plan['evaluation']):
            for t in plan['targets']:
                if not scoring.scoreable(c,t):raise ValueError('Unexpected exclusion')
                for arm in arms:
                    gen,trace=decode(c,t,arm)
                    row=dict(chain=c,target=t,arm=arm,model=plan['model'],answer_cluster=chains[c]['answer_label_by_lang']['en'],
                        scored_line=deg.first_line(gen['text']),generation=gen,reused=False,
                        score=dict(content=bool(scoring.content(gen['text'],c)),form=bool(scoring.ilr(gen['text'],c,t)),
                                   degenerate=bool(deg.is_degenerate(deg.first_line(gen['text'])))))
                    rows.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
                    tracefile.write(json.dumps(dict(chain=c,target=t,arm=arm,steps=trace))+'\n');tracefile.flush()
            log(f'[generate] {i+1}/{len(plan["evaluation"])} records={len(rows)}')
    if len(rows)!=plan['expected_records']:raise ValueError('Record count mismatch')
    save_json(a.out/'analysis.json',analyze(rows,plan))
    manifest.update(status='complete',records=len(rows),physical_generations=len(rows),finished_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('frozen','groups','pilot'):p.add_argument('--'+k,type=Path,required=True)
    for k in ('plan','plan-out','out'):p.add_argument('--'+k,type=Path)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    plan,f,g,pilot,chains=prepare(a)
    if a.plan and read(a.plan)!=plan:raise ValueError('Locked plan/source changed')
    if a.plan_out:
        with a.plan_out.open('x') as out:json.dump(plan,out,indent=2)
    if a.dry_run:log(json.dumps(dict(model=plan['model'],calibration=len(plan['split']['calibration']),validation=len(plan['split']['validation']),
                                     test=len(plan['evaluation']),records=plan['expected_records'])));return
    if not a.plan or not a.out:raise ValueError('Plan and new output required')
    a.out.mkdir(parents=True,exist_ok=False);save_json(a.out/'plan.json',plan)
    manifest=dict(status='running',model=plan['model'],started_unix=time.time(),plan_sha256=sha(a.out/'plan.json'))
    try:execute(a,plan,f,g,pilot,chains,manifest)
    except BaseException as exc:manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:save_json(a.out/'manifest.json',manifest)


if __name__=='__main__':main()
