"""Fixed-population re-generation of the language-neuron comparison for one checkpoint.

Re-generates every frozen evaluation chain under the frozen neuron groups and dose, with five random
control groups (seeds 7..47), then reports paired selected-minus-random effects with answer-cluster
bootstrap intervals and an exact-match comparison against the stored outputs. The plan file is hashed
before generation and checked afterwards, so it cannot change during execution.
"""
from collections import Counter
import hashlib
import json

from .intervention_cohorts import build_arms, first_line, save_json, summarize
from .untreated_strata import cluster_interval

SEEDS = [7, 17, 27, 37, 47]
TARGETS = ['ko', 'zh']


def make_plan(frozen, pilot_sha, frozen_sha, chat):
    ids = frozen['eval_chains']
    if not ids or len(set(ids)) != len(ids) or set(ids) & set(frozen['calib_chains']):
        raise ValueError('Invalid historical evaluation IDs')
    arms = build_arms(frozen['best'], TARGETS, SEEDS, [1.], ['original'])
    supplements = {a['suppression'] for aa in arms.values() for a in aa if a['kind']=='suppression'}
    return dict(mode='checkpoint_stability', schema_version=1, model=frozen['model'],
                chat=chat, chain_ids=list(ids), targets=TARGETS, discovery_targets=list(frozen['eval']),
                seeds=SEEDS, arms=arms, max_new_tokens=frozen['protocol']['max_new'],
                pilot_sha256=pilot_sha, frozen_sha256=frozen_sha,
                expected_physical_generations=len(ids)*(13+len(supplements)),
                expected_records=len(ids)*sum(map(len, arms.values())),
                selection='All original eval IDs, no rescreening or outcome filtering',
                scope='Prospective pinned-runtime replication and random-control sensitivity; original runtime identity not certified')


def analyze(records, chains, plan, draws=10000):
    keys={(r['chain'],r['target_language'],r['arm_id']) for r in records}
    expected={(c,t,a['arm_id']) for c in plan['chain_ids'] for t in TARGETS for a in plan['arms'][t]}
    if keys != expected or len(keys) != len(records):
        raise ValueError('Missing/duplicate/extra generation cells')
    result={}
    for t in TARGETS:
        rows=[r for r in records if r['target_language']==t]
        lookup={(r['chain'],r['kind'],r['seed']):r for r in rows}
        ids=[c for c in plan['chain_ids'] if lookup[c,'untreated',None]['score']['eligible']]
        if not ids:
            raise ValueError('No scoreable evaluation items')
        get=lambda c,k,s:lookup[c,k,s]['score']
        group=lambda c:chains[c]['answer_label_by_lang'].get('en') or c
        effects={}
        for metric in ('content','form','delta','degenerate'):
            def value(sc):
                return int(sc['content'])-int(sc['form']) if metric=='delta' else int(sc[metric])
            differences=[]; by_seed={str(s):[] for s in SEEDS}; iu=[]; ru=[]
            for c in ids:
                i=value(get(c,'selected',None)); u=value(get(c,'untreated',None))
                controls=[value(get(c,'random',s)) for s in SEEDS]
                differences.append(i-sum(controls)/len(SEEDS));iu.append(i-u);ru.append(sum(controls)/len(SEEDS)-u)
                for seed,r in zip(SEEDS,controls):by_seed[str(seed)].append(i-r)
            mean=lambda values:100*sum(values)/len(values)
            ci=cluster_interval(differences,[group(c) for c in ids],draws=draws)
            effects[metric]=dict(I_minus_R_mean_pp=mean(differences),I_minus_U_pp=mean(iu),R_minus_U_pp=mean(ru),
                conditional_cluster_ci95_pp=[100*x for x in ci] if ci else None,
                I_minus_R_by_seed_pp={s:mean(v) for s,v in by_seed.items()})
        bridge={}; states={}
        for kind,seed in [('untreated',None),('selected',None),('random',7)]:
            subset=[lookup[c,kind,seed] for c in ids]
            available=[r for r in subset if r['historical_text'] is not None]
            label=kind if seed is None else f'{kind}_{seed}'
            bridge[label]=dict(n=len(available),raw_exact=sum(r['generation']['text']==r['historical_text'] for r in available),
                storage_normalized_exact=sum(r['generation']['text'].replace('\ufffd','').strip()==r['historical_text'] for r in available),
                first_line_exact=sum(first_line(r['generation']['text'])==first_line(r['historical_text']) for r in available),
                content_score_changed=sum(r['score']['content']!=r['historical_score']['content'] for r in available),
                form_score_changed=sum(r['score']['form']!=r['historical_score']['form'] for r in available))
        for seed in SEEDS:
            counts=Counter(); common=[]
            for c in ids:
                i=get(c,'selected',None);r=get(c,'random',seed)
                state=f"{int(not i['degenerate'])}{int(not r['degenerate'])}"
                counts[state]+=1
                if state=='11':common.append(int(i['content'])-int(r['content']))
            states[str(seed)]=dict(counts=dict(counts),both_attempt_n=len(common),
                                  both_attempt_content_pp=100*sum(common)/len(common) if common else None)
        result[t]=dict(n=len(ids),original_n=len(plan['chain_ids']),clusters=len({group(c) for c in ids}),
                       effects=effects,historical_bridge=bridge,attempt_states=states)
    return dict(checkpoints={plan['model']:result},bootstrap_draws=draws,
                interpretation='CI is conditional on five fixed random neuron groups. Seeds are not independent questions. Model-specific eval sets/doses differ; no pure architecture effect is identified. Historical seed7 differences and matcher limitations must accompany conclusions.')


def run(*, chains, frozen, plan, decode, score, output, progress=print):
    save_json(output/'stability_plan.json',plan)
    lock=hashlib.sha256((output/'stability_plan.json').read_bytes()).hexdigest()
    save_json(output/'plan_lock.json',dict(sha256=lock))
    records=[]; physical=0
    with (output/'generations.jsonl').open('x',encoding='utf-8') as journal:
        for index,chain in enumerate(plan['chain_ids']):
            cache={}
            for t in TARGETS:
                for arm in plan['arms'][t]:
                    shared=arm['kind'] in ('untreated','suppression')
                    key=(arm['kind'],arm['suppression']) if shared else (t,arm['arm_id'])
                    reused=key in cache
                    if not reused:
                        cache[key]=decode(chain,t,arm);physical+=1
                    generation=cache[key]
                    prior=(frozen.get('floors',{}).get(chain) if arm['kind']=='untreated' else
                           frozen['eval'][t]['steer_items'][index].get('txt','') if arm['kind']=='selected' else
                           frozen['eval'][t]['rand_items'][index].get('txt','') if arm['kind']=='random' and arm['seed']==7 else None)
                    row=dict(model=frozen['model'],cohort='historical_eval_fixed',chain=chain,target_language=t,
                             prompt_language='es',**arm,generation=generation,
                             generation_id=chain+'/'+json.dumps(key),reused_generation=reused,
                             score=score(generation['text'],chain,t),
                             baseline_score=score(cache['untreated',1.]['text'],chain,t),historical_text=prior,
                             historical_score=score(prior,chain,t) if prior is not None else None)
                    records.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
            progress(f"[stability] items={index+1}/{len(plan['chain_ids'])} generations={physical}/{plan['expected_physical_generations']}")
    if physical!=plan['expected_physical_generations'] or len(records)!=plan['expected_records']:
        raise ValueError('Generation count differs from plan')
    if hashlib.sha256((output/'stability_plan.json').read_bytes()).hexdigest()!=lock:
        raise ValueError('Plan changed during execution')
    save_json(output/'summary.json',summarize(records))
    save_json(output/'stability_summary.json',analyze(records,chains,plan))
    return dict(n_records=len(records),replay=dict(physical_generation_calls=physical),
                cohorts=dict(selected_successes=0,success_shortfall=0))
