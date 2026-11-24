#!/usr/bin/env python3
"""Energy-matched random control for language-neuron amplification, zh/ko (Llama, Qwen, Gemma-4-It).

Arms: U (untreated), S (Spanish suppression only), I (selected neurons), R (construction-matched random groups at
the selected dose) and M (energy-matched random groups). Random-group increments are fitted on a calibration
split and frozen before test generation; all five random draws and all outcomes are retained.
Write the plan with --dry-run --plan-out plan.json, then run with --plan plan.json --out DIR.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2];CODE=ROOT/'01-code';RESULTS=ROOT/'02-data/results'
sys.path.insert(0,str(CODE/'src'))
from beacon.data.strength_control import SEEDS,split_clusters,balance,analyze,content_positions
from beacon.data.adaptive_strength import RULE,fit_energy
from beacon.data.tokenizer_checks import validate_pilot_files,tokenizer_identity,render_prompt
from beacon.data.intervention_cohorts import save_json


# Gemma-4-31B base uses the separate S=U runner (energy_matched_control_gemma.py).
ROSTER=('llama31_70','qwen36_27b','llama31_70_it','gemma4_31b_it')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def log(s):print(s,flush=True)


def prepare(a):
    f=read(a.frozen);g=read(a.groups);pilot=validate_pilot_files(a.pilot)
    if f['model'] not in ROSTER:raise ValueError('Frozen extension roster only')
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
           CODE/'src/beacon/data/adaptive_strength.py',CODE/'scripts/language_neuron_groups.py',
           CODE/'src/beacon/models.py',CODE/'configs/models.yaml',ROOT/'01-code/analysis'/'paired_effects.py',
           ROOT/'01-code/analysis'/'verify_numbers.py',ROOT/'01-code/analysis'/'degenerate_outputs.py']
    plan=dict(schema_version=1,mode='checkpoint_strength_prospective',model=f['model'],chat=pilot['chat'],
        targets=['ko','zh'],seeds=SEEDS,evaluation=f['eval_chains'],split=split,
        original_increment={t:f['best'][t]['amp'] for t in ('ko','zh')},
        suppression={t:f['best'][t]['supp'] for t in ('ko','zh')},
        max_new_tokens=32,search_rule=RULE,
        balance=dict(total_ratio=[.9,1.1],layer_ratio=[.8,1.2],minimum_energy_fraction=.9),
        expected_records=len(f['eval_chains'])*26,expected_physical_generations=len(f['eval_chains'])*26,
        model_revision=pilot['resolved_revision'],pilot_sha256=sha(a.pilot),
        hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths},
        fit_positions='user_message_span' if pilot['chat'] else 'all_prompt_positions',
        measurement='Dense block output; local amplification effect relative to frozen Spanish suppression-only states; '
                    +('user-message-span mean energy fitted (chat template positions excluded); all-position and last-token held-out diagnostics'
                      if pilot['chat'] else 'prompt-mean energy fitted, last-token held-out diagnostic'),
        selection='All original eval IDs; no rescreening; all five draws retained irrespective of matching; coefficients frozen before test generations',
        scope='Prospective checkpoint extension, not full-trajectory strength equivalence')
    if f['protocol']['max_new']!=32 or not plan['evaluation']:raise ValueError('Unexpected evaluation')
    return plan,f,g,pilot,chains


def execute(a,plan,f,g,pilot,chains,manifest):
    import torch
    from transformers import AutoTokenizer
    from beacon.models import load_model
    from beacon.config import load_configs
    from beacon.data.prompting import encode
    from beacon.standard_strength_runtime import validate_layers,capture,LayerReplay,intervention,synthetic_test
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
    neuron_module=module('standard_neuron_module',CODE/'scripts/language_neurons.py')
    runner=module('standard_group_validation',CODE/'scripts/language_neuron_groups.py')
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
    def fit_positions(c):
        if plan['fit_positions']=='all_prompt_positions':return None
        prompt=chains[c]['prompts']['es'];rendered=render_prompt(tokenizer,prompt,True)
        return content_positions(tokenizer,rendered['text'],prompt.rsplit('\nAnswer:',1)[0].strip(),
                                 rendered['add_special_tokens'],expected[c])
    manifest['runtime']=dict(versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','accelerate','huggingface_hub')},
        cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],dtype=str(loaded.model.dtype),
        generation_config=actual,model_revision=plan['model_revision'],prompt_tensor_checks=len(pilot_rows))
    save_json(a.out/'manifest.json',manifest);save_json(a.out/'groups.json',g)
    save_json(a.out/'evaluation_items.json',{c:chains[c] for c in plan['evaluation']})
    (a.out/'block_source.py.txt').write_text(inspect.getsource(type(layers[0])))
    calibration={};validation={}
    for t in plan['targets']:
        active=sorted(g['selected'][t]);factor=plan['suppression'][t]
        caches={};spans={}
        for split in ('calibration','validation'):
            caches[split]={};spans[split]={}
            for i,c in enumerate(plan['split'][split]):
                caches[split][c]=capture(loaded.model,layers,tokens(c),active,g['suppression'],factor)
                spans[split][c]=fit_positions(c)
                log(f'[capture] {t} {split} {i+1}/{len(plan["split"][split])}')
        span_rows=lambda split:None if plan['fit_positions']=='all_prompt_positions' else list(spans[split].values())
        if span_rows('calibration') is not None:
            save_json(a.out/f'fit_positions_{t}.json',{s:{c:p for c,p in spans[s].items()} for s in spans})
        calibration[t]={};validation[t]={}
        for seed in SEEDS:
            fits={};measurements={}
            for l,ix in g['selected'][t].items():
                cal=LayerReplay(layers[l],[row[l] for row in caches['calibration'].values()],g['suppression'].get(l,[]),factor,span_rows('calibration'))
                amp=plan['original_increment'][t];target=cal.measure(ix,amp);random_ix=g['random'][str(seed)][t][l]
                fit=fit_energy(target['mean'],lambda dose:cal.measure(random_ix,dose)['mean'],amp)
                fit.update(target_all=target['all'],target_last=target['last'],replay_relative_rms=cal.replay_relative_rms)
                del cal
                val=LayerReplay(layers[l],[row[l] for row in caches['validation'].values()],g['suppression'].get(l,[]),factor,span_rows('validation'))
                measurements[str(l)]=dict(I=val.measure(ix,amp),R=val.measure(random_ix,amp),M=val.measure(random_ix,fit['increment']),
                                         replay_relative_rms=val.replay_relative_rms,baseline_residual_energy=val.residual_energy)
                del val
                fits[str(l)]=fit
                log(f'[fit] {t} seed={seed} layer={l} increment={fit["increment"]:.5g} stop={fit["stop"]}')
            calibration[t][str(seed)]=fits
            validation[t][str(seed)]=dict(layers=measurements,balance={k:balance({l:v['I'][k] for l,v in measurements.items()},
                {l:v['M'][k] for l,v in measurements.items()}) for k in ('mean','all','last')})
            save_json(a.out/'calibration.json',calibration);save_json(a.out/'validation.json',validation)
            log(f'[balance] {t} seed={seed} {validation[t][str(seed)]["balance"]}')
        del caches
    save_json(a.out/'calibration_lock.json',dict(calibration_sha256=sha(a.out/'calibration.json'),validation_sha256=sha(a.out/'validation.json'),
        frozen_before_test_generation=True,validation_used_for_tuning=False))
    arms=['U','S','I']+[f'{kind}{seed}' for kind in ('R','M') for seed in SEEDS]
    def settings(t,arm):
        if arm in ('U','S'):return {},{},1. if arm=='U' else plan['suppression'][t]
        groups=g['selected'][t] if arm=='I' else g['random'][arm[1:]][t]
        inc={l:calibration[t][arm[1:]][str(l)]['increment'] if arm.startswith('M') else plan['original_increment'][t] for l in groups}
        return groups,inc,plan['suppression'][t]
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
        c=plan['split']['validation'][0];groups,inc,factor=settings(t,'R7')
        with torch.inference_mode(),intervention(layers,groups,inc,g['suppression'],factor):
            ids=tokens(c);steps=[]
            for _ in range(3):
                nxt=loaded.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1].argmax(-1,keepdim=True)
                steps.append(int(nxt.item()));ids=torch.cat([ids,nxt],1)
                eos=actual['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
                if steps[-1] in eos:break
        cached=decode(c,t,'R7',False,3)[0]['token_ids'];traced=decode(c,t,'R7',True,3)[0]['token_ids']
        if cached!=steps or cached!=traced:raise ValueError('Cached/full/telemetry parity failure')
        with torch.inference_mode(),intervention(layers,groups,{l:0. for l in groups},g['suppression'],factor):
            zero=loaded.model.generate(input_ids=tokens(c),max_new_tokens=3,do_sample=False,pad_token_id=tokenizer.eos_token_id)[0,tokens(c).shape[1]:].tolist()
        if zero!=decode(c,t,'S',False,3)[0]['token_ids']:raise ValueError('Zero-dose vs S mismatch')
        plumbing[t]=dict(cached_full_parity=True,telemetry_parity=True,zero_increment_equals_S=True)
    save_json(a.out/'plumbing_checks.json',plumbing)
    scoring=module('standard_scoring',ROOT/'01-code/analysis'/'paired_effects.py').V;deg=module('standard_deg',ROOT/'01-code/analysis'/'degenerate_outputs.py')
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
    analysis=analyze([r for r in rows if r['arm']!='S'],plan)
    from beacon.data.untreated_strata import cluster_interval
    index={(r['chain'],r['target'],r['arm']):r for r in rows}
    exact=[];transitions=[]
    for t in plan['targets']:
        ids=plan['evaluation'];clusters=[plan['split']['cluster_by_chain'][c] for c in ids]
        for arm in arms:
            exact.append(dict(target=t,arm=arm,n=len(ids),clusters=len(set(clusters)),
                **{k:sum(int(index[c,t,arm]['score'][k]) for c in ids) for k in ('content','form','degenerate')}))
            if arm=='I':continue
            state=lambda row:2 if row['score']['form'] else 1 if row['score']['content'] else 0
            matrix=[[0]*3 for _ in range(3)]
            for c in ids:matrix[state(index[c,t,arm])][state(index[c,t,'I'])]+=1
            transitions.append(dict(target=t,reference_arm=arm,reference_to_I=matrix,
                content_gains=sum(matrix[0][1:]),content_losses=matrix[1][0]+matrix[2][0],
                gains_with_form=matrix[0][2],gains_without_form=matrix[0][1]))
        for metric in ('content','form','delta','degenerate'):
            def value(c,arm):
                s=index[c,t,arm]['score']
                return int(s['content'])-int(s['form']) if metric=='delta' else int(s[metric])
            for reference in ('U','S'):
                values=[value(c,'I')-value(c,reference) for c in ids]
                ci=cluster_interval(values,clusters,draws=10000)
                analysis['results'][t]['effects'][metric][reference]=dict(pp=100*sum(values)/len(values),ci95_pp=[100*x for x in ci])
    save_json(a.out/'analysis.json',analysis)
    save_json(a.out/'counts_and_transitions.json',dict(arms=exact,transitions=transitions,
        state_order=['no_content','content_without_form','designated_form'],matrix_orientation='reference rows, I columns'))
    save_json(a.out/'balance_summary.json',{t:{s:v['balance'] for s,v in draws.items()} for t,draws in validation.items()})
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
    if a.dry_run:log(json.dumps(dict(model=plan['model'],calibration=30,validation=15,test=len(plan['evaluation']),records=plan['expected_records'])));return
    if not a.plan or not a.out:raise ValueError('Plan and new output required')
    a.out.mkdir(parents=True,exist_ok=False);save_json(a.out/'plan.json',plan)
    manifest=dict(status='running',model=plan['model'],started_unix=time.time(),plan_sha256=sha(a.out/'plan.json'))
    try:execute(a,plan,f,g,pilot,chains,manifest)
    except BaseException as exc:manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}');raise
    finally:save_json(a.out/'manifest.json',manifest)


if __name__=='__main__':main()
