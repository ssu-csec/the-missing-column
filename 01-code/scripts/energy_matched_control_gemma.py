#!/usr/bin/env python3
"""Energy-matched random control for language-neuron amplification (Gemma base, zh/ko).

Fits, per random seed and layer, the random-group increment whose residual-energy perturbation matches that of
the selected language neurons, then runs an all-outcome paired evaluation with arms U (untreated), I (selected
neurons), R (construction-matched random groups) and M (energy-matched random groups).
--adaptive-validation fits and validates the calibration only (no test generation); --frozen-calibration
reuses a completed calibration for the evaluation without refitting.

Write the plan with --dry-run --plan-out plan.json (no GPU needed), then run with --plan plan.json --out DIR.
The plan is fixed before generation. All seeds and degenerate outputs are retained; existing outputs are never
overwritten.
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
import traceback

CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent;RESULTS=ROOT/'02-data/results'
sys.path.insert(0,str(CODE/'src'))
from beacon.data.strength_control import SEEDS, GRID, split_clusters, fit_energy, balance, analyze
from beacon.data.tokenizer_checks import validate_pilot_files, tokenizer_identity
from beacon.data.intervention_cohorts import save_json


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def log(text):print(text,flush=True)


def prepare(args):
    frozen=read(args.frozen);chains=read(ROOT/'02-data/inputs/chain_lang_prompts.json')
    bundle=read(args.groups);pilot=validate_pilot_files(args.tokenizer_pilot)
    if frozen['model']!='gemma4_31b' or bundle['chat'] is not False:
        raise ValueError('This runner supports Gemma base raw prompts only; no silent recipe fallback')
    if any(frozen['best'][t]['supp']!=1 for t in ('ko','zh')):
        raise ValueError('This runner requires S=U')
    if bundle['frozen_sha256']!=sha(args.frozen) or bundle['model_revision']!=pilot['resolved_revision']:
        raise ValueError('Group/frozen/model revision mismatch')
    if bundle['tokenizer_identity_sha256']!=pilot['identity_sha256']:
        raise ValueError('Group tokenizer identity mismatch')
    if bundle['pairs_sha256']!=sha(ROOT/'02-data/inputs/parallel_pairs.jsonl'):
        raise ValueError('Discovery corpus differs')
    if not all(str(s) in bundle['random'] for s in SEEDS):raise ValueError('Missing saved seed')
    paths=[args.frozen,args.groups,ROOT/'02-data/inputs/chain_lang_prompts.json',
           ROOT/'02-data/inputs/parallel_pairs.jsonl',CODE/'scripts/language_neurons.py',
           CODE/'scripts/language_neuron_groups.py',Path(__file__),CODE/'src/beacon/strength_runtime.py',
           CODE/'src/beacon/data/strength_control.py',CODE/'src/beacon/models.py',CODE/'configs/models.yaml',
           CODE/'src/beacon/data/prompting.py',ROOT/'01-code/analysis'/'paired_effects.py',ROOT/'01-code/analysis'/'verify_numbers.py',ROOT/'01-code/analysis'/'degenerate_outputs.py']
    plan=dict(schema_version=1,model=frozen['model'],chat=False,seeds=SEEDS,targets=['ko','zh'],
              evaluation=frozen['eval_chains'],split=split_clusters(chains,frozen),
              original_increment={t:frozen['best'][t]['amp'] for t in ('ko','zh')},
              max_new_tokens=32,grid_scales=GRID,bisection_steps=8,
              balance=dict(total_ratio=[.9,1.1],layer_ratio=[.8,1.2],minimum_energy_fraction=.9),
              expected_physical_generations=len(frozen['eval_chains'])*23,
              expected_records=len(frozen['eval_chains'])*24,
              pilot_sha256=sha(args.tokenizer_pilot),model_revision=pilot['resolved_revision'],
              hashes={str(p.relative_to(ROOT)):sha(p) for p in paths},
              measurement='S-state local dense Gemma block output including postnorm, bf16 addition and layer scalar',
              zero_energy_policy='exact zero uses increment 0; all positive energies retained; no near-zero dropping',
              generation_selection='all original test IDs; all five seeds regardless of achieved balance')
    if len(plan['evaluation'])!=120 or frozen['protocol']['max_new']!=32:raise ValueError('Unexpected original test')
    if getattr(args,'adaptive_validation',False):
        from beacon.data.adaptive_strength import RULE, fresh_neuron_split
        if not args.previous_plan:raise ValueError('Adaptive validation requires --previous-plan')
        previous=read(args.previous_plan)
        if previous['evaluation']!=frozen['eval_chains']:raise ValueError('Evaluation set mismatch')
        plan.update(schema_version=2,mode='adaptive_validation_only',search_rule=RULE,
                    split=fresh_neuron_split(chains,frozen,previous),grid_scales=None,bisection_steps=12,
                    expected_physical_generations=0,expected_records=0,
                    generation_selection='No test generation; validation-only run')
        for path in [args.previous_plan,CODE/'src/beacon/data/adaptive_strength.py']:
            plan['hashes'][str(path.resolve().relative_to(ROOT))]=sha(path)
    if getattr(args,'frozen_calibration',None):
        if args.adaptive_validation:raise ValueError('Frozen evaluation cannot refit calibration')
        source=args.frozen_calibration.resolve();prior=read(source/'plan.json');status=read(source/'manifest.json')
        if status['status']!='complete' or prior.get('mode')!='adaptive_validation_only':raise ValueError('Completed adaptive calibration required')
        for key in ('model','chat','seeds','targets','evaluation','original_increment','max_new_tokens','pilot_sha256','model_revision','measurement','balance'):
            if prior[key]!=plan[key]:raise ValueError(f'Frozen calibration setting differs: {key}')
        for path,digest in prior['hashes'].items():
            if path!='01-code/scripts/energy_matched_control_gemma.py' and sha(ROOT/path)!=digest:
                raise ValueError(f'Calibration dependency changed: {path}')
        if sha(source/'plan.json')!=status['plan_sha256']:raise ValueError('Calibration plan checksum mismatch')
        cal=read(source/'calibration.json');val=read(source/'validation.json')
        for t in plan['targets']:
            for seed in SEEDS:
                for l in bundle['selected'][t]:
                    dose=cal[t][str(seed)][str(l)]['increment']
                    if not 0<=dose<float('inf'):raise ValueError('Nonfinite frozen coefficient')
                    val[t][str(seed)]['layers'][str(l)]
        plan.update(schema_version=3,mode='frozen_adaptive_evaluation',split=prior['split'],
                    grid_scales=None,bisection_steps=prior['bisection_steps'],search_rule=prior['search_rule'],
                    calibration_source=str(source.relative_to(ROOT)),calibration_plan_sha256=sha(source/'plan.json'))
        for name in ('plan.json','manifest.json','calibration.json','validation.json','groups.json'):
            path=source/name;plan['hashes'][str(path.relative_to(ROOT))]=sha(path)
    return plan,frozen,chains,bundle,pilot


def execute(args,plan,frozen,chains,bundle,pilot,manifest):
    import torch
    from transformers import AutoTokenizer
    from beacon.config import load_configs
    from beacon.data.prompting import encode
    from beacon.models import load_model
    from beacon.strength_runtime import validate_layers,capture,LayerReplay,intervention,synthetic_test
    synthetic_test()
    if not torch.cuda.is_available():raise ValueError('GPU required for generation')
    for name in ('transformers','tokenizers'):
        if importlib.metadata.version(name)!=pilot['versions'][name]:raise ValueError(f'{name} differs from pilot')
    if importlib.metadata.version('torch')!='2.8.0':
        # CUDA wheels can include a local-version suffix.
        if importlib.metadata.version('torch').split('+')[0]!='2.8.0':raise ValueError('Expected torch 2.8.0')
    tokenizer=AutoTokenizer.from_pretrained(args.tokenizer_pilot.parent/'snapshot',local_files_only=True)
    if tokenizer_identity(tokenizer)['sha256']!=pilot['identity_sha256']:raise ValueError('Tokenizer identity')
    config=load_configs()
    if config.models['models'][plan['model']]['hf_id']!=pilot['repo_id']:raise ValueError('Model repo mismatch')
    log(f"[load] {plan['model']} pinned {plan['model_revision']}")
    loaded=load_model(plan['model'],config,framework='hf_hooks',attn_implementation='eager',revision=plan['model_revision'],tokenizer=tokenizer)
    runner=module('strength_legacy_runner',CODE/'scripts/language_neuron_groups.py')
    neuron_module=module('energy_matched_neuron_module',CODE/'scripts/language_neurons.py')
    bundle=runner.normalize_and_validate_groups(bundle,loaded,neuron_module,list(frozen['eval']),SEEDS)
    layers=neuron_module._decoder_layers(loaded.model);validate_layers(layers)
    if getattr(loaded.model.config,'_commit_hash',None) not in (None,plan['model_revision']):raise ValueError('Loaded revision')
    expected_config=read(args.tokenizer_pilot.parent/'effective_generation_config.json')
    actual_config=loaded.model.generation_config.to_dict()
    for k in ('eos_token_id','bos_token_id','repetition_penalty','no_repeat_ngram_size','min_length','min_new_tokens',
              'forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens','bad_words_ids','stop_strings'):
        if expected_config.get(k)!=actual_config.get(k):raise ValueError(f'Generation setting {k}')
    expected={}
    prompt_rows=[json.loads(s) for s in (args.tokenizer_pilot.parent/'prompt_tokens.jsonl').read_text().splitlines()]
    for r in prompt_rows:
        if encode(loaded,r['text'],False)[0].tolist()!=r['input_ids']:raise ValueError('Tokenizer tensor mismatch')
        if r['source']=='chains' and r['language']=='es':expected[r['key']]=r['input_ids']
    if max(tokenizer.get_vocab().values())>=loaded.model.get_input_embeddings().weight.shape[0]:raise ValueError('Embedding bounds')
    manifest['runtime']=dict(versions={n:importlib.metadata.version(n) for n in ('torch','transformers','tokenizers','accelerate','huggingface_hub')},
                             cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                             generation_config=actual_config,dtype=str(loaded.model.dtype),
                             prompt_tensor_checks=len(prompt_rows),block_class=type(layers[0]).__name__)
    save_json(args.out/'manifest.json',manifest)
    (args.out/'block_source.py.txt').write_text(inspect.getsource(type(layers[0])))
    save_json(args.out/'groups.json',bundle)
    save_json(args.out/'group_overlap.json',{t:{str(s):{str(l):len(set(ns)&set(bundle['random'][str(s)][t][l]))
                   for l,ns in bundle['selected'][t].items()} for s in SEEDS} for t in plan['targets']})
    active=sorted(set().union(*(set(bundle['selected'][t]) for t in plan['targets'])))
    def tokens(c):
        ids=encode(loaded,chains[c]['prompts']['es'],False)
        if ids[0].tolist()!=expected[c]:raise ValueError('Changed prompt IDs')
        return ids
    scoring=module('strength_scoring',ROOT/'01-code/analysis'/'paired_effects.py').V
    deg=module('strength_degeneracy',ROOT/'01-code/analysis'/'degenerate_outputs.py')
    if scoring.MATCHER_POLICY!='attribution_strict':raise ValueError('Expected locked matcher')
    def score(text,c,t):
        if not scoring.scoreable(c,t):raise ValueError('Unexpected excluded zh/ko item')
        return dict(content=bool(scoring.content(text,c)),form=bool(scoring.ilr(text,c,t)),
                    degenerate=bool(deg.is_degenerate(deg.first_line(text))))
    @torch.inference_mode()
    def decode(c,groups=None,increments=None,trace=True,max_new=32):
        ids=tokens(c)
        with intervention(layers,groups or {},increments or {},telemetry=trace) as trajectory:
            out=loaded.model.generate(input_ids=ids,max_new_tokens=max_new,do_sample=False,pad_token_id=tokenizer.eos_token_id)
        generated=out[0,ids.shape[1]:].tolist();eos=actual_config['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
        return dict(text=tokenizer.decode(generated,skip_special_tokens=True),token_ids=generated,
                    prompt_token_ids=ids[0].tolist(),text_with_specials=tokenizer.decode(generated,skip_special_tokens=False),
                    n_generated_tokens=len(generated),stop_reason='eos' if generated and generated[-1] in eos else 'max_new_tokens' if len(generated)==max_new else 'other',
                    trajectory=trajectory)
    if plan.get('mode')=='frozen_adaptive_evaluation':
        source=ROOT/plan['calibration_source'];old_runtime=read(source/'manifest.json')['runtime']
        for key in ('versions','cuda','dtype'):
            if manifest['runtime'][key]!=old_runtime[key]:raise ValueError(f'Calibration runtime differs: {key}')
        if json.loads(json.dumps(bundle))!=read(source/'groups.json'):raise ValueError('Frozen groups differ')
        calibration=read(source/'calibration.json');validation=read(source/'validation.json')
        save_json(args.out/'calibration.json',calibration);save_json(args.out/'validation.json',validation)
        manifest['calibration_reused']=True;manifest['calibration_source']=plan['calibration_source']
        log('[calibration] reusing frozen calibration coefficients and validation; no refit or seed selection')
    else:
        caches={}
        with torch.inference_mode():
            for split in ('calibration','validation'):
                caches[split]={}
                for i,c in enumerate(plan['split'][split]):
                    caches[split][c]=capture(loaded.model,layers,tokens(c),active)
                    log(f'[capture] {split} {i+1}/{len(plan["split"][split])}')
        calibration={t:{} for t in plan['targets']}; validation={t:{} for t in plan['targets']}
        # Seed 7 is completed and validated first; the other seeds proceed without outcome-based selection.
        for seed in SEEDS:
            for t in plan['targets']:
                fits={};measurements={};amp=plan['original_increment'][t]
                for l,indices in bundle['selected'][t].items():
                    with torch.inference_mode():
                        cal=LayerReplay(layers[l],[caches['calibration'][c][l] for c in plan['split']['calibration']])
                        target=cal.measure(indices,amp)
                        random_ids=bundle['random'][str(seed)][t][l]
                        fit=fit_energy(target['mean'],lambda a:cal.measure(random_ids,a)['mean'],amp)
                        fit.update(target_last=target['last'],replay_relative_rms=cal.replay_relative_rms)
                        if cal.measure(random_ids,0)['mean']!=0:raise ValueError('Zero-dose identity failed')
                        del cal
                        val=LayerReplay(layers[l],[caches['validation'][c][l] for c in plan['split']['validation']])
                        measurements[str(l)]={
                            'I':val.measure(indices,amp),'R':val.measure(random_ids,amp),
                            'M':val.measure(random_ids,fit['increment']),
                            'baseline_residual_energy':val.residual_energy,'replay_relative_rms':val.replay_relative_rms}
                        del val
                    fits[str(l)]=fit
                    log(f'[fit] {t} seed={seed} layer={l} a={fit["increment"]:.5g} ratio={fit.get("ratio")}')
                calibration[t][str(seed)]=fits
                balance_by_metric={metric:balance({l:v['I'][metric] for l,v in measurements.items()},
                                                 {l:v['M'][metric] for l,v in measurements.items()}) for metric in ('mean','last')}
                validation[t][str(seed)]=dict(layers=measurements,balance=balance_by_metric)
                save_json(args.out/'calibration.json',calibration);save_json(args.out/'validation.json',validation)
                log(f'[validation] {t} seed={seed} {balance_by_metric["mean"]}')
            if seed==7:
                save_json(args.out/'pilot_status.json',dict(status='measurement_complete',seed=7,
                    validation={t:validation[t]['7']['balance'] for t in plan['targets']},
                    continuation='Proceed with all seeds irrespective of balance; unmatched seeds remain reported'))
        del caches
    if plan.get('mode')=='adaptive_validation_only':
        save_json(args.out/'balance_summary.json',{t:{s:v['balance'] for s,v in seeds.items()} for t,seeds in validation.items()})
        manifest.update(status='complete',physical_generations=0,records=0,finished_unix=time.time(),
                        validation_passes=sum(v['balance']['mean']['achieved'] for seeds in validation.values() for v in seeds.values()),
                        contrasts=10)
        return
    def arm_config(t,a):
        if a=='U':return {},{}
        if a=='I':return bundle['selected'][t],{l:plan['original_increment'][t] for l in bundle['selected'][t]}
        kind=a[0];seed=a[1:];g=bundle['random'][seed][t]
        return g,{l:calibration[t][seed][str(l)]['increment'] if kind=='M' else plan['original_increment'][t] for l in g}
    # Technical parity before test generation: cached vs full recomputation under a nonzero intervention.
    parity_chain=plan['split']['validation'][0];t='zh';g,inc=arm_config(t,'R7')
    with torch.inference_mode(),intervention(layers,g,inc):
        ids=tokens(parity_chain);steps=[]
        for _ in range(3):
            pred=loaded.model(input_ids=ids,use_cache=False,logits_to_keep=1).logits[:,-1].argmax(-1,keepdim=True)
            steps.append(int(pred.item()));ids=torch.cat([ids,pred],dim=1)
            eos=actual_config['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
            if steps[-1] in eos:break
    cached=decode(parity_chain,g,inc,trace=False,max_new=3)['token_ids']
    if cached!=steps:raise ValueError(f'Cached/full token parity failed: {cached} vs {steps}')
    traced=decode(parity_chain,g,inc,trace=True,max_new=3)['token_ids']
    if traced!=cached:raise ValueError('Telemetry changed generation')
    zero=decode(parity_chain,g,{l:0. for l in g},trace=False,max_new=3)['token_ids']
    untreated=decode(parity_chain,trace=False,max_new=3)['token_ids']
    if zero!=untreated:raise ValueError('Zero hook changed generation')
    save_json(args.out/'plumbing_checks.json',dict(cached_full_tokens=True,telemetry_parity=True,zero_hook_identity=True,
              calibration_test_overlap=0,postnorm_scalar_replay_checked=True))
    arms=['U','I']+[f'{kind}{s}' for kind in ('R','M') for s in SEEDS]
    # Shared validation continuations: no tuning; preserve EOS coverage and degenerate U outputs.
    with (args.out/'prefix_diagnostics.jsonl').open('x') as journal:
        for ci,c in enumerate(plan['split']['validation']):
            u=decode(c,trace=False,max_new=8);journal.write(json.dumps(dict(chain=c,untreated_generation=u))+'\n');journal.flush()
            for n in (0,1,4,8):
                if n>len(u['token_ids']):continue
                ids=tokens(c)
                if n:ids=torch.cat([ids,torch.tensor([u['token_ids'][:n]],device=ids.device)],dim=1)
                with torch.inference_mode():baseline=capture(loaded.model,layers,ids,active)
                for t in plan['targets']:
                    for arm in arms[1:]:
                        g,inc=arm_config(t,arm);values={};handles=[]
                        for l in active:
                            def hook(m,a,o,l=l):
                                ref=baseline[l]['output'].to(o.device)
                                d=(o.float()-ref.float()).square().sum(-1)
                                values[str(l)]=dict(mean=float(d.mean()),last=float(d[:,-1].mean()))
                            handles.append(layers[l].register_forward_hook(hook))
                        try:
                            with torch.inference_mode(),intervention(layers,g,inc):loaded.model(input_ids=ids,use_cache=False,logits_to_keep=1)
                        finally:
                            for h in handles:h.remove()
                        journal.write(json.dumps(dict(chain=c,target=t,arm=arm,prefix_length=n,energy=values))+'\n');journal.flush()
                del baseline
            log(f'[prefix] validation {ci+1}/{len(plan["split"]["validation"])}')
    records=[];calls=0
    with (args.out/'generations.jsonl').open('x') as journal, (args.out/'trajectory.jsonl').open('x') as telemetry:
        for ci,c in enumerate(plan['evaluation']):
            u=None
            for t in plan['targets']:
                for arm in arms:
                    if arm=='U' and u is not None:gen=u;reused=True
                    else:
                        g,inc=arm_config(t,arm);gen=decode(c,g,inc);calls+=1;reused=False
                        trajectory=gen.pop('trajectory')
                        telemetry.write(json.dumps(dict(chain=c,target=t,arm=arm,steps=trajectory))+'\n');telemetry.flush()
                        if arm=='U':u=gen
                    row=dict(chain=c,target=t,arm=arm,generation=gen,score=score(gen['text'],c,t),reused=reused)
                    records.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
            log(f'[generate] {ci+1}/120 physical={calls}/{plan["expected_physical_generations"]}')
    if calls!=plan['expected_physical_generations'] or len(records)!=plan['expected_records']:raise ValueError('Generation count')
    save_json(args.out/'analysis.json',analyze(records,plan))
    save_json(args.out/'balance_summary.json',{t:{s:v['balance'] for s,v in seeds.items()} for t,seeds in validation.items()})
    manifest.update(status='complete',physical_generations=calls,records=len(records),finished_unix=time.time())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frozen',type=Path,default=RESULTS/'interventions/language_neurons_gemma4_31b.json')
    p.add_argument('--groups',type=Path,default=RESULTS/'frozen_inputs/gemma4_31b/groups.json')
    p.add_argument('--tokenizer-pilot',type=Path,required=True)
    p.add_argument('--out',type=Path)
    p.add_argument('--plan',type=Path)
    p.add_argument('--plan-out',type=Path)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--adaptive-validation',action='store_true')
    p.add_argument('--previous-plan',type=Path)
    p.add_argument('--frozen-calibration',type=Path,help='Reuse completed adaptive calibration without fitting')
    args=p.parse_args()
    if args.adaptive_validation:
        global fit_energy
        from beacon.data.adaptive_strength import fit_energy
    plan,frozen,chains,bundle,pilot=prepare(args)
    if args.plan and read(args.plan)!=plan:raise ValueError('Locked plan/source hashes changed')
    if args.plan_out:
        if args.plan_out.exists():raise ValueError('Refuse plan overwrite')
        args.plan_out.parent.mkdir(parents=True,exist_ok=True)
        save_json(args.plan_out,plan)
    if args.dry_run:
        print(json.dumps(dict(model=plan['model'],split={k:len(plan['split'][k]) for k in ('calibration','validation')},
                         test=len(plan['evaluation']),seeds=plan['seeds'],physical_generations=plan['expected_physical_generations']),indent=2));return
    if not args.plan or not args.out:raise ValueError('GPU run requires --plan and --out')
    if args.out.exists():raise ValueError('Output exists; partial journals must not be overwritten')
    args.out.mkdir(parents=True);save_json(args.out/'plan.json',plan)
    manifest=dict(status='running',started_unix=time.time(),plan_sha256=sha(args.out/'plan.json'))
    save_json(args.out/'manifest.json',manifest)
    try:execute(args,plan,frozen,chains,bundle,pilot,manifest)
    except BaseException as e:
        manifest.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save_json(args.out/'manifest.json',manifest)


if __name__=='__main__':main()
