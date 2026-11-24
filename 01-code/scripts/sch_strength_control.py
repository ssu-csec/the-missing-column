#!/usr/bin/env python3
"""Energy-matched random-head control for second-hop content-head (SCH) amplification.

Uses the frozen general SCH pool and the sch_amplification.py evaluation items. Per random seed and layer, the
random-head increment is fitted so that its attention-residual energy matches that of the selected heads.
Default: calibration/validation only. --evaluate-all records U/I/R/M for every evaluation item and every seed,
including unmatched controls. Items are not rescreened and the qualified head pool is not changed.
Write the plan with --dry-run --plan-out plan.json, then run with --plan plan.json --out DIR.
"""
import argparse,hashlib,importlib.util,importlib.metadata,json,sys,time
from pathlib import Path
CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent;RESULTS=ROOT/'02-data/results'
sys.path.insert(0,str(CODE/'src'))
from beacon.data.adaptive_strength import RULE,fit_energy
from beacon.data.strength_control import SEEDS,balance,analyze
from beacon.data.tokenizer_checks import validate_pilot_files,tokenizer_identity
from beacon.data.intervention_cohorts import save_json

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def prepare(a):
 heads_path=RESULTS/f'interventions/sch_pool_{a.model}_general.json'
 render_path=RESULTS/f'interventions/sch_amplification_{a.model}_general.json'
 h=read(heads_path);r=read(render_path);chains=read(ROOT/'02-data/inputs/chain_lang_prompts.json');pilot=validate_pilot_files(a.tokenizer_pilot)
 if h['pool_status']!='positive_control_supported':raise ValueError('Original SCH pool not qualified')
 if pilot['model']!=a.model or pilot['chat']!=r['protocol']['chat']:raise ValueError('SCH chat/model differs from pilot')
 if r['protocol'].get('checkpoint_revision') not in (None,pilot['resolved_revision']):raise ValueError('Checkpoint revision differs')
 targets=['es','zh','ja','ko'];evaluation={t:[x['chain'] for x in r['by_lang'][t]['items']] for t in targets}
 prompts={};paths=[heads_path,render_path,ROOT/'02-data/inputs/chain_lang_prompts.json',Path(__file__),
  CODE/'src/beacon/sch_strength_runtime.py',CODE/'src/beacon/data/adaptive_strength.py',CODE/'src/beacon/data/strength_control.py',
  CODE/'src/beacon/data/untreated_strata.py',CODE/'src/beacon/data/intervention_cohorts.py',
  CODE/'src/beacon/pipeline/patching.py',CODE/'src/beacon/hooks.py',CODE/'src/beacon/models.py',
  CODE/'src/beacon/data/prompting.py',CODE/'src/beacon/data/tokenizer_checks.py',CODE/'configs/models.yaml',
  ROOT/'01-code/analysis'/'paired_effects.py',ROOT/'01-code/analysis'/'verify_numbers.py',ROOT/'01-code/analysis'/'degenerate_outputs.py']
 for t in targets:
  path=ROOT/f'02-data/inputs/sch_subsets/subset_{a.model}_{t}_failure.jsonl';paths.append(path)
  lookup={x['chain_id']:x for x in map(json.loads,path.read_text().splitlines())}
  prompts[t]={c:lookup[c]['prompt'] for c in evaluation[t]}
  if len(set(evaluation[t]))!=len(evaluation[t]):raise ValueError('Duplicate evaluation item')
 inventory={(x['text'],x['language']) for x in map(json.loads,(a.tokenizer_pilot.parent/'prompt_tokens.jsonl').read_text().splitlines()) if x['source']=='chains'}
 if any((text,t) not in inventory for t,items in prompts.items() for text in items.values()):
  raise ValueError('Exact evaluation prompt absent from tokenizer pilot')
 cluster=lambda c:chains[c]['answer_label_by_lang']['en']
 excluded={cluster(c) for ids in evaluation.values() for c in ids}
 pool={}
 for c in chains:
  if cluster(c) not in excluded and all(chains[c]['prompts'].get(t) for t in targets):pool.setdefault(cluster(c),[]).append(c)
 rank=lambda x:hashlib.sha256(('sch-strength-v1/'+x).encode()).hexdigest()
 ranked=[min(pool[k],key=rank) for k in sorted(pool,key=rank)]
 if len(ranked)<15:raise ValueError('Insufficient test-disjoint answer clusters')
 # Use a smaller fixed split when fewer than 45 test-disjoint clusters are available.
 ncal,nval=(30,15) if len(ranked)>=45 else (10,5 if len(ranked)==15 else 6)
 split=dict(calibration=ranked[:ncal],validation=ranked[ncal:ncal+nval],available_clusters=len(ranked),
            cluster_by_chain={c:cluster(c) for c in chains},discovery_overlap='Original discovery IDs not certified; no claim of discovery-disjoint fitting')
 plan=dict(schema_version=1,model=a.model,chat=pilot['chat'],revision=pilot['resolved_revision'],targets=targets,
  evaluation=evaluation,evaluation_prompts=prompts,split=split,seeds=SEEDS,search_rule=RULE,
  selected_heads=[[x['layer'],x['head']] for x in h['heads']],n_heads=h['architecture']['n_heads'],alpha=1.,max_new_tokens=32,
  evaluate_all=a.evaluate_all,pilot_sha256=sha(a.tokenizer_pilot),
  expected_records=sum(map(len,evaluation.values()))*12 if a.evaluate_all else 0,
  hashes={str(p.resolve().relative_to(ROOT)):sha(p) for p in paths},
  measurement='Fixed answer-colon/assistant-marker attention residual junction before MLP input normalization; includes attention postnorm where present, excludes downstream MLP and block-end scaling',
  scope='Prospective pinned-runtime comparison; no rescreening, no outcome tuning or historical identity claim',
  thresholds=dict(total_ratio=[.9,1.1],layer_ratio=[.8,1.2],minimum_energy_fraction=.9))
 return plan,chains,pilot


def execute(a,p,chains,pilot,manifest):
 import torch
 from transformers import AutoTokenizer
 from beacon.models import load_model
 from beacon.config import load_configs
 from beacon.hooks import _decoder_layers
 from beacon.pipeline.patching import random_pool_like,_heads_by_layer,colon_decode
 from beacon.sch_strength_runtime import capture,AttentionReplay,coordinates,decode,synthetic_test,head_hooks
 synthetic_test()
 if not torch.cuda.is_available():raise ValueError('GPU required')
 if importlib.metadata.version('torch').split('+')[0]!='2.8.0':raise ValueError('Expected torch 2.8.0')
 for name in ('transformers','tokenizers'):
  if importlib.metadata.version(name)!=pilot['versions'][name]:raise ValueError('Pilot runtime mismatch')
 tok=AutoTokenizer.from_pretrained(a.tokenizer_pilot.parent/'snapshot',local_files_only=True)
 if tokenizer_identity(tok)['sha256']!=pilot['identity_sha256']:raise ValueError('Tokenizer identity mismatch')
 cfg=load_configs()
 if cfg.models['models'][a.model]['hf_id']!=pilot['repo_id']:raise ValueError('Model source mismatch')
 loaded=load_model(a.model,cfg,framework='hf_hooks',attn_implementation='eager',revision=p['revision'],tokenizer=tok);loaded.model.eval()
 if getattr(loaded.model.config,'_commit_hash',None) not in (None,p['revision']):raise ValueError('Model commit mismatch')
 if loaded.spec.n_heads!=p['n_heads']:raise ValueError('Head count mismatch')
 layers=_decoder_layers(loaded.model);selected=_heads_by_layer(p['selected_heads'])
 random={str(s):_heads_by_layer(random_pool_like(p['selected_heads'],loaded.spec.n_layers,p['n_heads'],seed=s)) for s in SEEDS}
 save_json(a.out/'groups.json',dict(selected=selected,random=random))
 manifest['runtime']=dict(versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','accelerate')},
  cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],dtype=str(loaded.model.dtype))
 save_json(a.out/'manifest.json',manifest)
 # Validate every exact evaluation prompt and calibration prompt against the tokenizer preflight inventory.
 from beacon.data.prompting import encode
 inventory={(x['text'],x['language']):x['input_ids'] for x in map(json.loads,(a.tokenizer_pilot.parent/'prompt_tokens.jsonl').read_text().splitlines()) if x['source']=='chains'}
 for t in p['targets']:
  texts=list(p['evaluation_prompts'][t].values())+[chains[c]['prompts'][t] for k in ('calibration','validation') for c in p['split'][k]]
  for text in texts:
   if encode(loaded,text,p['chat'])[0].tolist()!=inventory[text,t]:raise ValueError('Prompt token identity mismatch')
 calibration={};validation={};checks={}
 for t in p['targets']:
  calibration[t]={};validation[t]={}
  cache={}
  with torch.inference_mode():
   for split in ('calibration','validation'):
    cache[split]=[capture(loaded,chains[c]['prompts'][t],p['chat'],selected) for c in p['split'][split]]
   # Nonzero actual-model junction parity with only one active layer, before fitting.
   l=next(iter(selected));probe=chains[p['split']['calibration'][0]]['prompts'][t]
   from beacon.pipeline.patching import _encoded_col
   _,col=_encoded_col(loaded,probe,p['chat'])
   reference=cache['calibration'][0][l]
   replay=AttentionReplay(layers[l],[reference])
   expected=replay.measure(coordinates(layers[l],selected[l],p['n_heads']),1.)
   with head_hooks(layers,{l:selected[l]},{l:1.},col,p['n_heads']):
    actual=capture(loaded,probe,p['chat'],[l])[l]
   observed=float((actual['observed'].float()-reference['observed'].float()).square().sum(-1).mean())
   if abs(observed-expected)>max(1e-5,.02*max(observed,expected)):
    raise ValueError(f'Nonzero attention replay disagrees with actual hook: {observed} vs {expected}')
   checks[t]=dict(nonzero_energy_parity=True,observed_energy=observed,replayed_energy=expected)
   del replay
   for seed in SEEDS:
    fits={};measurements={}
    for l,heads in selected.items():
     cal=AttentionReplay(layers[l],[row[l] for row in cache['calibration']]);ix=coordinates(layers[l],heads,p['n_heads']);rx=coordinates(layers[l],random[str(seed)][l],p['n_heads'])
     target=cal.measure(ix,p['alpha']);fit=fit_energy(target,lambda dose:cal.measure(rx,dose),p['alpha'])
     fit['replay_max_relative_rms']=max(cal.errors)
     fit['single_token_replay_max_relative_rms']=max(cal.single_token_errors)
     print('[replay]',t,seed,l,'full_sequence_rms',max(cal.errors),'single_token_rms',max(cal.single_token_errors),flush=True)
     fits[str(l)]=fit;del cal
     val=AttentionReplay(layers[l],[row[l] for row in cache['validation']])
     measurements[str(l)]=dict(I=val.measure(ix,p['alpha']),R=val.measure(rx,p['alpha']),M=val.measure(rx,fit['increment']),replay_max_relative_rms=max(val.errors));del val
    calibration[t][str(seed)]=fits
    validation[t][str(seed)]=dict(layers=measurements,balance=balance({l:v['I'] for l,v in measurements.items()},{l:v['M'] for l,v in measurements.items()}))
    save_json(a.out/'calibration.json',calibration);save_json(a.out/'validation.json',validation)
    print('[validation]',t,seed,validation[t][str(seed)]['balance'],flush=True)
  # Check that nonzero-dose decoding matches the SCH implementation used by sch_amplification.py (colon_decode).
  text=chains[p['split']['validation'][0]]['prompts'][t]
  new=decode(loaded,text,p['chat'],selected,{l:1. for l in selected},3)
  old=colon_decode(loaded,text,3,pool=p['selected_heads'],alpha=1.,chat=p['chat'])
  if new['text']!=old:raise ValueError('SCH decoding parity failed')
  zero=decode(loaded,text,p['chat'],selected,{l:0. for l in selected},3)
  plain=decode(loaded,text,p['chat'],{},{},3)
  if zero['token_ids']!=plain['token_ids']:raise ValueError('Zero head hook changed tokens')
  checks[t].update(legacy_text_parity=True,zero_token_parity=True)
 save_json(a.out/'plumbing_checks.json',checks)
 if not p['evaluate_all']:
  manifest.update(status='complete',records=0,validation_only=True);return
 scoring=module('sch_strength_scoring',ROOT/'01-code/analysis'/'paired_effects.py').V
 deg=module('sch_strength_deg',ROOT/'01-code/analysis'/'degenerate_outputs.py')
 if scoring.MATCHER_POLICY!='attribution_strict':raise ValueError('Primary matcher required')
 records=[]
 with (a.out/'generations.jsonl').open('x') as journal:
  for t in p['targets']:
   for c in p['evaluation'][t]:
    arms=['U','I']+[f'{k}{s}' for k in ('R','M') for s in SEEDS]
    for arm in arms:
     g={} if arm=='U' else selected if arm=='I' else random[arm[1:]]
     doses={l:calibration[t][arm[1:]][str(l)]['increment'] if arm.startswith('M') else 1. for l in g}
     gen=decode(loaded,p['evaluation_prompts'][t][c],p['chat'],g,doses)
     eligible=bool(scoring.scoreable(c,t));text=gen['text']
     row=dict(chain=c,target=t,arm=arm,generation=gen,score=dict(eligible=eligible,content=bool(scoring.content(text,c)),
        form=bool(scoring.ilr(text,c,t)) if eligible else None,degenerate=bool(deg.is_degenerate(deg.first_line(text)))))
     records.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
    print('[generate]',t,c,len(records),'/',p['expected_records'],flush=True)
 if len(records)!=p['expected_records']:raise ValueError('Record count mismatch')
 summaries={}
 for t in p['targets']:
  ids=[c for c in p['evaluation'][t] if scoring.scoreable(c,t)]
  if not ids:summaries[t]={'n':0};continue
  q=dict(p,targets=[t],evaluation=ids)
  summaries[t]=analyze([r for r in records if r['target']==t and r['chain'] in ids],q)
 save_json(a.out/'analysis.json',summaries);manifest.update(status='complete',records=len(records))


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True,choices=['gemma4_31b','llama31_70','llama31_70_it','qwen36_27b'])
 p.add_argument('--tokenizer-pilot',type=Path,required=True);p.add_argument('--dry-run',action='store_true');p.add_argument('--evaluate-all',action='store_true')
 p.add_argument('--plan',type=Path);p.add_argument('--plan-out',type=Path);p.add_argument('--out',type=Path)
 a=p.parse_args();plan,chains,pilot=prepare(a)
 if a.plan and read(a.plan)!=plan:raise ValueError('Locked plan changed')
 if a.plan_out:
  a.plan_out.parent.mkdir(parents=True,exist_ok=True)
  with a.plan_out.open('x') as f:json.dump(plan,f,indent=2)
 if a.dry_run:
  print(json.dumps(dict(model=a.model,n_eval={t:len(v) for t,v in plan['evaluation'].items()},split={k:len(plan['split'][k]) for k in ('calibration','validation')},records=plan['expected_records']),indent=2));return
 if not a.plan or not a.out:raise ValueError('GPU execution requires locked --plan and new --out')
 a.out.mkdir(parents=True,exist_ok=False);save_json(a.out/'plan.json',plan)
 manifest=dict(status='running',started=time.time(),plan_sha256=sha(a.out/'plan.json'));save_json(a.out/'manifest.json',manifest)
 try:execute(a,plan,chains,pilot,manifest)
 except BaseException as e:manifest.update(status='failed',error=f'{type(e).__name__}: {e}');raise
 finally:manifest['finished']=time.time();save_json(a.out/'manifest.json',manifest)
if __name__=='__main__':main()
