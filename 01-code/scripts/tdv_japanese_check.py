#!/usr/bin/env python3
"""Japanese 70B single-hop TDV implementation check with complete raw arms (App. E).

The legacy Korean-only script and published result are preserved. This reconstructs
the vector from historical two-hop pool IDs, pins runtime, and saves all 200 screened
items in U-ja/U-en/TDV/R0/R17/R27/R37/R47, including untreated successes.

    python 01-code/scripts/tdv_japanese_check.py --tokenizer-pilot <report.json> \
        --plan 02-data/results/tdv_japanese_check/data/raw/plan.json --out <new directory>

The tokenizer pilot is a tokenizer_preflight.py report for Llama-3.1-70B-Instruct (chat); the snapshot
of the original run is not redistributed. Analysis: 01-code/analysis/tdv_japanese_check_recompute.py.
"""
import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys
import time

CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent;RESULTS=ROOT/'02-data/results'
sys.path.insert(0,str(CODE/'src'))
from beacon.data.tdv_check import LANGS,SEEDS,TRANS,make_plan,analyze
from beacon.data.tokenizer_checks import validate_pilot_files,tokenizer_identity,check_prompt
from beacon.data.intervention_cohorts import save_json


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def log(s):print(s,flush=True)


def prepare(args):
    cp=ROOT/'02-data/inputs/chain_lang_prompts.json';sp=ROOT/'02-data/inputs/single_hop_prompts.jsonl'
    hp=RESULTS/'tdv/transvec_llama31_70_it.json'
    chains=read(cp);historical=read(hp);single={}
    for row in map(json.loads,sp.read_text().splitlines()):
        c,l=row['chain_id'],row['lang']
        if l in single.setdefault(c,{}):raise ValueError('Duplicate single-hop prompt')
        single[c][l]=row['prompt']
    pilot=validate_pilot_files(args.tokenizer_pilot)
    plan=make_plan(chains,single,historical)
    paths=[cp,sp,hp,Path(__file__),CODE/'scripts/translation_vector.py',CODE/'src/beacon/data/tdv_check.py',
           CODE/'src/beacon/models.py',CODE/'src/beacon/data/prompting.py',CODE/'src/beacon/data/evaluate.py',
           CODE/'src/beacon/data/langid.py',CODE/'src/beacon/data/tokenizer_checks.py',CODE/'configs/models.yaml',
           CODE/'analysis/paired_effects.py',CODE/'analysis/verify_numbers.py',CODE/'analysis/degenerate_outputs.py']
    plan.update(schema_version=1,pilot_sha256=sha(args.tokenizer_pilot),revision=pilot['resolved_revision'],
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    if pilot['model']!=plan['model'] or pilot['chat'] is not True:raise ValueError('Need Llama70B IT chat tokenizer pilot')
    return plan,chains,single,pilot


def run(args,plan,chains,single,pilot,manifest):
    import torch
    from transformers import AutoTokenizer
    from beacon.config import load_configs
    from beacon.models import load_model
    from beacon.data.prompting import encode
    from beacon.data.evaluate import entity_match
    from beacon.pipeline.patching import _decoder_layers,_unwrap_layer_out
    legacy=module('tdv_original',CODE/'scripts/translation_vector.py')
    if TRANS!=legacy.TRANS_INSTR:raise ValueError('Translation template drift')
    legacy._verify()
    for package in ('transformers','tokenizers'):
        if importlib.metadata.version(package)!=pilot['versions'][package]:raise ValueError(f'{package} version mismatch')
    if importlib.metadata.version('torch').split('+')[0]!='2.8.0':raise ValueError('Pinned torch mismatch')
    tok=AutoTokenizer.from_pretrained(args.tokenizer_pilot.parent/'snapshot',local_files_only=True)
    if tokenizer_identity(tok)['sha256']!=pilot['identity_sha256']:raise ValueError('Tokenizer identity mismatch')
    config=load_configs();spec=config.models['models'][plan['model']]
    if spec.get('local_path') or spec['hf_id']!=pilot['repo_id']:raise ValueError('Model source differs')
    mc=read(args.tokenizer_pilot.parent/'snapshot/config.json');vocab=mc.get('text_config',mc)['vocab_size']
    prompts={}
    for c in plan['construction']:
        for l in ('ko','ja','zh','es'):prompts[c,'translation',l]=TRANS[l].format(x=chains[c]['answer_label_by_lang']['en'])
        for l in LANGS:prompts[c,'recall',l]=single[c][l]
    for c in plan['screening']:
        for l in ('ja','en'):prompts[c,'screen',l]=single[c][l]
    checked={}
    with (args.out/'prompt_tokens.jsonl').open('x') as f:
        for key,text in prompts.items():
            result=check_prompt(tok,text,True,vocab);checked[key]=result['input_ids']
            f.write(json.dumps(dict(chain=key[0],kind=key[1],language=key[2],prompt=text,**result),ensure_ascii=False)+'\n')
    save_json(args.out/'tokenizer_preflight.json',dict(status='PASS',prompt_cases=len(checked),
                identity_sha256=pilot['identity_sha256'],revision=plan['revision'],original_pilot_sha256=plan['pilot_sha256']))
    log(f'[preflight] {len(checked)} single-hop / translation prompt encodings passed')
    log(f'[load] {plan["model"]} {plan["revision"]}')
    loaded=load_model(plan['model'],config,framework='hf_hooks',attn_implementation='eager',revision=plan['revision'],tokenizer=tok)
    if getattr(loaded.model.config,'_commit_hash',None) not in (None,plan['revision']):raise ValueError('Loaded revision mismatch')
    layers=_decoder_layers(loaded.model)
    if len(layers)!=80:raise ValueError('Expected 80-layer checkpoint')
    if max(tok.get_vocab().values())>=loaded.model.get_input_embeddings().weight.shape[0]:raise ValueError('Embedding bounds')
    for key,text in prompts.items():
        if encode(loaded,text,True)[0].tolist()!=checked[key]:raise ValueError('Tensor tokenizer check failed')
    actual=loaded.model.generation_config.to_dict();expected=read(args.tokenizer_pilot.parent/'effective_generation_config.json')
    for k in ('bos_token_id','eos_token_id','forced_bos_token_id','forced_eos_token_id','suppress_tokens','begin_suppress_tokens','bad_words_ids','stop_strings'):
        if actual.get(k)!=expected.get(k):raise ValueError(f'Generation config drift: {k}')
    manifest['runtime']=dict(versions={k:importlib.metadata.version(k) for k in ('torch','transformers','tokenizers','accelerate','huggingface_hub')},
          cuda=torch.version.cuda,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
          dtype=str(loaded.model.dtype),generation_config=actual,explicit_decode=plan['decode'])
    save_json(args.out/'manifest.json',manifest)
    layer=plan['layer'];sum_t=sum_r=None;nt=nr=0
    # Same ordering, float32 CPU accumulation, and hidden_states[L+1] definition as legacy.
    with torch.inference_mode():
        for i,c in enumerate(plan['construction']):
            for l in ('ko','ja','zh','es'):
                h=legacy._final_h(loaded,prompts[c,'translation',l],True,[layer])[layer]
                sum_t=h.clone() if sum_t is None else sum_t+h;nt+=1
            for l in LANGS:
                h=legacy._final_h(loaded,prompts[c,'recall',l],True,[layer])[layer]
                sum_r=h.clone() if sum_r is None else sum_r+h;nr+=1
            log(f'[construct] {i+1}/60 translation={nt} recall={nr}')
    if (nt,nr)!=(240,300):raise ValueError('Construction count differs')
    delta=sum_t/nt-sum_r/nr
    if not torch.isfinite(delta).all() or delta.norm()==0:raise ValueError('Invalid TDV')
    random={}
    for seed in SEEDS:
        raw=torch.randn(delta.shape[0],generator=torch.Generator().manual_seed(seed))
        random[seed]=raw/raw.norm()*delta.norm()
        if not torch.isclose(random[seed].norm(),delta.norm(),rtol=1e-6):raise ValueError('Random norm mismatch')
    torch.save(dict(delta=delta,random=random,translation_mean=sum_t/nt,recall_mean=sum_r/nr),args.out/'vectors.pt')
    save_json(args.out/'vector_metadata.json',dict(vector_sha256=sha(args.out/'vectors.pt'),layer=layer,
        norm=float(delta.norm()),random_norms={str(s):float(v.norm()) for s,v in random.items()},
        construction_translation=nt,construction_recall=nr,source='reconstructed; historical vector unavailable'))
    scoring=module('tdv_current_scoring',CODE/'analysis/paired_effects.py').V
    deg=module('tdv_degeneracy',CODE/'analysis/degenerate_outputs.py')
    if scoring.MATCHER_POLICY!='attribution_strict':raise ValueError('Matcher must be locked')
    @torch.inference_mode()
    def generate(c,lang,vec=None):
        text=single[c][lang];ids=encode(loaded,text,True)
        if ids[0].tolist()!=checked[c,'screen',lang]:raise ValueError('Changed input encoding')
        handle=layers[layer].register_forward_hook(legacy.AddVec(vec,1.)) if vec is not None else None
        try:out=loaded.model.generate(input_ids=ids,**plan['decode'],pad_token_id=tok.eos_token_id)
        finally:
            if handle is not None:handle.remove()
        tokens=out[0,ids.shape[1]:].tolist();eos=actual['eos_token_id'];eos=eos if isinstance(eos,list) else [eos]
        raw_text=tok.decode(tokens,skip_special_tokens=True)
        # Preserve unmodified text; legacy score used replacement-char removal/strip.
        score_text=raw_text.replace('\ufffd','').strip()
        return dict(text=raw_text,legacy_score_text=score_text,token_ids=tokens,prompt_token_ids=ids[0].tolist(),
                    text_with_specials=tok.decode(tokens,skip_special_tokens=False),n_generated_tokens=len(tokens),
                    stop_reason='eos' if tokens and tokens[-1] in eos else 'max_new_tokens' if len(tokens)==24 else 'other')
    # Check local hook add/remove on a screening prompt; this does not select a scale or sample.
    first=plan['screening'][0]
    neutral=generate(first,'ja');zero=generate(first,'ja',torch.zeros_like(delta))
    if neutral['token_ids']!=zero['token_ids']:raise ValueError('Zero-vector hook changed generation')
    save_json(args.out/'plumbing_checks.json',dict(legacy_unit_check=True,zero_vector_token_parity=True,
                  checkpoint_layers=80,primary_layer=layer,vector_norm_matching=True,
                  zero_vector_probe_chain=first,extra_probe_generation_calls=2))
    records=[];screen=[]
    with (args.out/'generations.jsonl').open('x') as journal,(args.out/'screening.jsonl').open('x') as sf:
        for i,c in enumerate(plan['screening']):
            item={}
            for arm,lang,vec in [('U_ja','ja',None),('U_en','en',None),('TDV','ja',delta)]+[(f'R{s}','ja',random[s]) for s in SEEDS]:
                gen=generate(c,lang,vec);text=gen['legacy_score_text'];meta=chains[c]
                english_label=meta['answer_label_by_lang']['en']
                english_ok=bool(legacy._label_hit(text,english_label) or entity_match(text,english_label,[english_label]+(meta.get('answer_aliases_by_lang',{}).get('en') or []))) if lang=='en' else None
                eligible=bool(scoring.scoreable(c,'ja'))
                r=dict(chain=c,arm=arm,prompt_language=lang,target_language='ja',generation=gen,
                       legacy_ilr=bool(legacy._correct_T(text,meta,lang)),english_correct=english_ok,
                       locked_score=dict(eligible=eligible,content=bool(scoring.content(gen['text'],c)),
                           form=bool(scoring.ilr(gen['text'],c,'ja')) if eligible else None),
                       degenerate=bool(deg.is_degenerate(deg.first_line(gen['text']))))
                records.append(r);item[arm]=r;journal.write(json.dumps(r,ensure_ascii=False)+'\n');journal.flush()
            row=dict(chain=c,japanese_untreated_correct=item['U_ja']['legacy_ilr'],english_untreated_correct=item['U_en']['english_correct'],
                     implementation_eligible=not item['U_ja']['legacy_ilr'] and item['U_en']['english_correct'],
                     construction_cluster_disjoint=c in plan['cluster_disjoint_screening'])
            screen.append(row);sf.write(json.dumps(row)+'\n');sf.flush()
            log(f'[check] {i+1}/200 generations={len(records)}/1600 eligible_failures={sum(r["implementation_eligible"] for r in screen)}')
    if len(records)!=plan['expected_generations']:raise ValueError('Unexpected generation count')
    summary=analyze(records,plan);save_json(args.out/'analysis.json',summary)
    lines=['# Japanese 70B TDV single-hop implementation check','',
           'Current pinned-runtime reconstruction; not exact replay of the historical vector.','',
           '| Subset | N | TDV recovery % | R0 recovery % | TDV−R0 pp [95% CI] |',
           '|---|---:|---:|---:|---|']
    for name,s in summary['subsets'].items():
        if not s['n']:lines.append(f'| {name} | 0 | — | — | inconclusive |');continue
        a=s['contrasts']['R0']
        ci=f'[{a["ci95"][0]:+.1f}, {a["ci95"][1]:+.1f}]' if a['ci95'] is not None else '[insufficient clusters]'
        lines.append(f'| {name} | {s["n"]} | {s["rates"]["TDV"]["percent_or_pp"]:.1f} | {s["rates"]["R0"]["percent_or_pp"]:.1f} | {a["percent_or_pp"]:+.1f} {ci} |')
    lines+=['','No minimum absolute recovery threshold is asserted post hoc. Report rates and uncertainty together.',
            'Cluster-disjoint sensitivity is mandatory because historical construction was item-disjoint only.',
            'New single-hop evidence does not rescore or replace the historical two-hop +19.6 pp.']
    (args.out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    manifest.update(status='complete',finished_unix=time.time(),generation_records=len(records),
                    extra_plumbing_generation_calls=2,screening_items=len(screen))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokenizer-pilot',type=Path,required=True)
    p.add_argument('--out',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--plan-out',type=Path)
    p.add_argument('--dry-run',action='store_true');a=p.parse_args()
    plan,chains,single,pilot=prepare(a)
    if a.plan and read(a.plan)!=plan:raise ValueError('Frozen plan/source changed')
    if a.plan_out:
        if a.plan_out.exists():raise ValueError('Refuse plan overwrite')
        a.plan_out.parent.mkdir(parents=True,exist_ok=True);save_json(a.plan_out,plan)
    if a.dry_run:
        print(json.dumps({k:plan[k] for k in ('model','layer','scale','seeds','expected_generations')},indent=2))
        print('construction',len(plan['construction']),'screen',len(plan['screening']),'cluster-disjoint screen',len(plan['cluster_disjoint_screening']))
        return
    if not a.plan or not a.out:raise ValueError('GPU run requires --plan and --out')
    if a.out.exists():raise ValueError('Refuse existing output directory')
    a.out.mkdir(parents=True);save_json(a.out/'plan.json',plan)
    manifest=dict(status='running',started_unix=time.time(),plan_sha256=sha(a.out/'plan.json'))
    save_json(a.out/'manifest.json',manifest)
    try:run(a,plan,chains,single,pilot,manifest)
    except BaseException as e:manifest.update(status='failed',error=f'{type(e).__name__}: {e}');raise
    finally:save_json(a.out/'manifest.json',manifest)


if __name__=='__main__':main()
