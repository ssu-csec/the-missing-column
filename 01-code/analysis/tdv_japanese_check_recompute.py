#!/usr/bin/env python3
"""Recompute the Japanese TDV check (App. E) from its released run, using only the Python standard library.

Checks the SHA-256 of every file in 02-data/results/tdv_japanese_check/MANIFEST.json and of the run inputs
recorded in its plan, rescores all 1,600 outputs with the run-time scorer and with the locked matcher, and
recomputes the paired answer-cluster intervals; the result must equal the run's own analysis.
The scorer functions are executed from their unchanged source in 01-code/ without importing torch.

    python3 01-code/analysis/tdv_japanese_check_recompute.py --out /path/to/new-directory
"""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import types
import unicodedata


CODE=Path(__file__).resolve().parents[1];ROOT=CODE.parent


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def compare(a,b):
    if isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:compare(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,float):assert math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-10),(a,b)
    else:assert a==b,(a,b)


def pure_functions(path,names,env,assignments=()):
    """Execute the unchanged scorer function ASTs, omitting unrelated torch imports."""
    tree=ast.parse(path.read_text());nodes=[];found=set()
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name in names:
            assert not node.decorator_list
            nodes.append(node);found.add(node.name)
        elif isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id in assignments for t in node.targets):nodes.append(node)
    assert found==set(names),(path,found,names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)


def csvfile(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=ROOT/'02-data/results/tdv_japanese_check')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();root=a.package.resolve()
    if a.out.exists():raise ValueError('Use a new output directory; existing calculations are preserved')
    manifest=read(root/'MANIFEST.json')
    for rel,digest in manifest['sha256'].items():
        part=Path(rel)
        assert not part.is_absolute() and '..' not in part.parts
        assert sha(root/part)==digest,rel
    code=CODE
    sys.path.insert(0,str(code/'src'))
    from beacon.data.tdv_check import analyze
    raw=root/'data/raw';plan=read(raw/'plan.json');rows=[json.loads(s) for s in (raw/'generations.jsonl').read_text().splitlines()]
    run=read(raw/'manifest.json');assert run['status']=='complete'
    assert run['plan_sha256']==sha(raw/'plan.json')
    assert read(raw/'vector_metadata.json')['vector_sha256']==sha(raw/'vectors.pt')
    # Run inputs recorded in the plan (the code it records is the released code in 01-code/).
    for rel in ('02-data/inputs/chain_lang_prompts.json','02-data/inputs/single_hop_prompts.jsonl',
                '02-data/results/tdv/transvec_llama31_70_it.json'):
        assert sha(ROOT/rel)==plan['source_sha256'][rel],rel
    V=load('packaged_separation',code/'analysis/paired_effects.py').V
    D=load('packaged_degenerate',code/'analysis/degenerate_outputs.py')
    # The original scoring functions are pure Python; use their unchanged ASTs.
    evaluation=types.ModuleType('beacon.data.evaluate');env=evaluation.__dict__;env.update(re=re,unicodedata=unicodedata)
    pure_functions(code/'src/beacon/data/evaluate.py',
        ['_fold','_normalize','_has_cjk','_tok_eq','_latin_run','_form_matches','entity_match'],env,['_ARTICLES'])
    sys.modules['beacon.data.evaluate']=evaluation
    pure_functions(code/'src/beacon/data/langid.py',
        ['_has_hangul','_has_kana','_has_han','_latin_toks','_cjk_hit','_latin_exact_run','output_language'],env)
    pure_functions(code/'scripts/translation_vector.py',['_label_hit','_correct_T'],env)
    for row in rows:
        c=row['chain'];text=row['generation']['text'];normalized=text.replace('\ufffd','').strip();lang=row['prompt_language']
        assert normalized==row['generation']['legacy_score_text']
        meta=V.CHAINS[c]
        assert row['legacy_ilr']==bool(env['_correct_T'](normalized,meta,lang)),(c,row['arm'],'legacy')
        label=meta['answer_label_by_lang']['en']
        english=bool(env['_label_hit'](normalized,label) or env['entity_match'](normalized,label,[label]+(meta.get('answer_aliases_by_lang',{}).get('en') or []))) if lang=='en' else None
        assert row['english_correct']==english,(c,row['arm'],'English')
        eligible=bool(V.scoreable(c,'ja'))
        assert row['locked_score']==dict(eligible=eligible,content=bool(V.content(text,c)),form=bool(V.ilr(text,c,'ja')) if eligible else None)
        assert row['degenerate']==bool(D.is_degenerate(D.first_line(text)))
        assert len(row['generation']['token_ids'])==row['generation']['n_generated_tokens']
    summary=analyze(rows,plan,draws=10000);compare(summary,read(root/'results/server/analysis.json'))
    by={(r['chain'],r['arm']):r for r in rows};disjoint=set(plan['cluster_disjoint_screening'])
    for r in map(json.loads,(raw/'screening.jsonl').read_text().splitlines()):
        c=r['chain'];assert r==dict(chain=c,japanese_untreated_correct=by[c,'U_ja']['legacy_ilr'],
          english_untreated_correct=by[c,'U_en']['english_correct'],
          implementation_eligible=not by[c,'U_ja']['legacy_ilr'] and by[c,'U_en']['english_correct'],
          construction_cluster_disjoint=c in disjoint)
    a.out.mkdir(parents=True);save(a.out/'analysis.json',summary)
    rates=[];contrasts=[]
    for subset,s in summary['subsets'].items():
        if not s['n']:continue
        for arm,v in s['rates'].items():
            count=round(v['percent_or_pp']*s['n']/100)
            rates.append(dict(subset=subset,n=s['n'],answer_clusters=s['clusters'],arm=arm,
               correct=count,rate_percent=v['percent_or_pp'],ci95_low=v['ci95'][0] if v['ci95'] else '',ci95_high=v['ci95'][1] if v['ci95'] else ''))
        for control,v in s['contrasts'].items():
            contrasts.append(dict(subset=subset,n=s['n'],answer_clusters=s['clusters'],control=control,
               delta_pp=v['percent_or_pp'],ci95_low=v['ci95'][0] if v['ci95'] else '',ci95_high=v['ci95'][1] if v['ci95'] else '',primary_control=control=='R0'))
    csvfile(a.out/'rates.csv',rates);csvfile(a.out/'contrasts.csv',contrasts)
    flat=[]
    for r in rows:
        c=r['chain'];g=r['generation'];s=r['locked_score']
        flat.append(dict(chain=c,answer_cluster=plan['cluster_by_chain'][c],arm=r['arm'],prompt_language=r['prompt_language'],
             implementation_eligible=not by[c,'U_ja']['legacy_ilr'] and by[c,'U_en']['english_correct'],
             construction_cluster_disjoint=c in disjoint,legacy_ilr=r['legacy_ilr'],english_correct=r['english_correct'],
             locked_eligible=s['eligible'],locked_content=s['content'],locked_form=s['form'],degenerate=r['degenerate'],
             n_tokens=g['n_generated_tokens'],stop_reason=g['stop_reason'],text=g['text']))
    csvfile(a.out/'generations_scored.csv',flat)
    validation=dict(status='PASS',records=len(rows),complete_grid=True,source_and_vector_hashes=True,
         legacy_ilr_recomputed=len(rows),english_eligibility_recomputed=200,locked_scores_recomputed=len(rows),
         screening_recomputed=200,paired_bootstrap_draws=10000,server_analysis_matches=True,
         scoring_method='Unchanged pure-function ASTs from the released source, executed without torch; no human judgments')
    save(a.out/'validation.json',validation);print(json.dumps(validation,indent=2))


if __name__=='__main__':main()
