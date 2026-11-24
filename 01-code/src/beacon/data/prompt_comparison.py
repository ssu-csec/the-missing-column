"""Spanish-prompt vs target-language-prompt comparison under frozen language-neuron groups.

Every frozen evaluation chain is generated under both prompt conditions with the same arms (untreated,
suppression, selected neurons, five random controls) and the same frozen Spanish suppression. The
population, groups, and doses are fixed: no re-screening or re-calibration. Outputs paired per-item
vectors, answer-cluster bootstrap intervals, and content/form transition tables.
"""
from collections import defaultdict
import csv
import hashlib
import json
import random
from .checkpoint_stability import make_plan as original_plan
from .intervention_cohorts import save_json


def make_plan(frozen, pilot_sha, frozen_sha, chat):
    p = original_plan(frozen, pilot_sha, frozen_sha, chat)
    p.update(mode='prompt_comparison', prompt_conditions=['es', 'target'],
             suppression_policy='Frozen Spanish suppression in BOTH prompt conditions',
             primary_control='mean of five fixed random draws; seed 7 sensitivity',
             scope='Prospective prompt comparison using reconstructed frozen groups; not exact historical replication',
             inference='Unadjusted descriptive cluster intervals; no original 12-cell verdict inherited')
    # No cache sharing, even for identical Spanish untreated prompts. Transparent paired records.
    p['expected_records'] = len(p['chain_ids']) * 2 * sum(map(len, p['arms'].values()))
    p['expected_physical_generations'] = p['expected_records']
    return p


def expected_keys(plan):
    return {(c,t,lg,a['arm_id']) for c in plan['chain_ids'] for t in plan['targets']
            for lg in ('es',t) for a in plan['arms'][t]}


def vectors(records, chains, plan):
    lookup = {(r['chain'],r['target_language'],r['prompt_language'],r['arm_id']): r for r in records}
    if len(lookup) != len(records) or set(lookup) != expected_keys(plan):
        raise ValueError('Missing, duplicate, or extra prompt/arm observations')
    output = {}
    for t in plan['targets']:
        rows=[]
        for c in plan['chain_ids']:
            arms = plan['arms'][t]
            scores = {lg: {a['arm_id']: lookup[c,t,lg,a['arm_id']]['score'] for a in arms} for lg in ('es',t)}
            eligible = {s['eligible'] for by_arm in scores.values() for s in by_arm.values()}
            if len(eligible) != 1:
                raise ValueError('Eligibility must be item-level and symmetric')
            if not next(iter(eligible)):
                raise ValueError('Unexpected ineligible zh/ko item; do not silently change the population')
            if any(s['form'] and not s['content'] for arm in scores.values() for s in arm.values()):
                raise ValueError('Form must imply content')
            values={}
            for lg in ('es',t):
                by_kind={(a['kind'],a['seed']):scores[lg][a['arm_id']] for a in arms}
                for metric in ('content','form','delta','degenerate'):
                    def val(sc):
                        return int(sc['content'])-int(sc['form']) if metric=='delta' else int(sc[metric])
                    i=val(by_kind['selected',None]); u=val(by_kind['untreated',None])
                    controls=[val(by_kind['random',seed]) for seed in plan['seeds']]
                    values[f'{lg}:{metric}:I-R']=i-sum(controls)/len(controls)
                    values[f'{lg}:{metric}:I-U']=i-u
                    for seed,r in zip(plan['seeds'],controls):
                        values[f'{lg}:{metric}:I-R{seed}']=i-r
                        values[f'{lg}:{metric}:R{seed}']=r
                    for kind in ('untreated','selected'):
                        values[f'{lg}:{metric}:{kind}']=val(by_kind[kind,None])
                    values[f'{lg}:{metric}:random_mean']=sum(controls)/len(controls)
                for seed in plan['seeds']:
                    si=by_kind['selected',None]; sr=by_kind['random',seed]
                    values[f'{lg}:gain:R{seed}']=int(si['content'] and not sr['content'])
                    values[f'{lg}:loss:R{seed}']=int(sr['content'] and not si['content'])
                    values[f'{lg}:gain_without_form:R{seed}']=int(si['content'] and not sr['content'] and not si['form'])
            values['gamma']=values[f'{t}:delta:I-R']-values['es:delta:I-R']
            for metric in ('content', 'form', 'delta', 'degenerate'):
                for contrast in ('I-R', 'I-U'):
                    values[f'prompt_difference:{metric}:{contrast}'] = (
                        values[f'{t}:{metric}:{contrast}']-values[f'es:{metric}:{contrast}'])
            for seed in plan['seeds']:
                values[f'gamma:R{seed}']=values[f'{t}:delta:I-R{seed}']-values[f'es:delta:I-R{seed}']
            rows.append(dict(chain=c,cluster=chains[c]['answer_label_by_lang']['en'],values=values))
        output[t]=rows
    return output


def aggregate(datasets, draws=4000):
    """Equal-checkpoint means; shared cluster resamples for all metrics within checkpoint."""
    rng=random.Random(20260917); result={}
    for t in ('zh','ko'):
        pools=[]; keys=list(datasets[0][t][0]['values'])
        for dataset in datasets:
            groups=defaultdict(list)
            for row in dataset[t]: groups[row['cluster']].append(row['values'])
            if len(groups)<2: raise ValueError('Need at least two answer clusters')
            pools.append(list(groups.values()))
        def mean_sample(selected):
            n=sum(len(g) for g in selected)
            return {k:sum(row[k] for g in selected for row in g)/n for k in keys}
        points=[mean_sample(pool) for pool in pools]
        samples={k:[] for k in keys}
        for _ in range(draws):
            estimates=[mean_sample(rng.choices(pool,k=len(pool))) for pool in pools]
            for k in keys: samples[k].append(100*sum(e[k] for e in estimates)/len(estimates))
        stats={}
        for k in keys:
            values=sorted(samples[k]); stats[k]=dict(estimate_pp=100*sum(p[k] for p in points)/len(points),
                ci95_pp=[values[int((draws-1)*.025)],values[int((draws-1)*.975)]])
        result[t]=dict(n_by_checkpoint=[len(d[t]) for d in datasets],
                      clusters_by_checkpoint=[len(p) for p in pools],metrics=stats)
    return dict(checkpoints=len(datasets),draws=draws,results=result,
                bootstrap_seed=20260917, weighting='equal checkpoint; items within checkpoint',
                interpretation='Descriptive unadjusted intervals conditional on fixed groups and roster; gamma is target minus Spanish delta. No outcome filtering. Clusters use English answer labels as in the original audit.')


def descriptive_report(records, chains, plan):
    """Exact denominators and transitions; never pool random draws as new items."""
    vectors(records, chains, plan)  # completeness and scorer invariants
    index={(r['chain'],r['target_language'],r['prompt_language'],r['arm_id']):r
           for r in records}
    state=lambda r: 2 if r['score']['form'] else 1 if r['score']['content'] else 0
    arms=[]; transitions=[]; prompt_transitions=[]
    for t in plan['targets']:
        selected=next(a for a in plan['arms'][t] if a['kind']=='selected')
        for lg in ('es',t):
            for arm in plan['arms'][t]:
                rows=[index[c,t,lg,arm['arm_id']] for c in plan['chain_ids']]
                counts={k:sum(bool(r['score'][k]) for r in rows)
                        for k in ('content','form','degenerate','eligible')}
                arms.append(dict(target=t,prompt_language=lg,arm_id=arm['arm_id'],
                    kind=arm['kind'],seed=arm['seed'],n=len(rows),
                    answer_clusters=len({chains[r['chain']]['answer_label_by_lang']['en'] for r in rows}),
                    **counts, content_without_form=counts['content']-counts['form'],
                    state_counts=[sum(state(r)==s for r in rows) for s in range(3)],
                    stop_reasons={reason:sum(r.get('generation',{}).get('stop_reason','unrecorded')==reason for r in rows)
                                  for reason in sorted({r.get('generation',{}).get('stop_reason','unrecorded') for r in rows})}))
                if arm['kind']=='selected': continue
                matrix=[[0]*3 for _ in range(3)]; attempt=[[0]*2 for _ in range(2)]
                sums={k:0 for k in ('content','form','delta')}; common_n=0
                for c in plan['chain_ids']:
                    ref=index[c,t,lg,arm['arm_id']]; sel=index[c,t,lg,selected['arm_id']]
                    matrix[state(ref)][state(sel)]+=1
                    rd=bool(ref['score']['degenerate']); sd=bool(sel['score']['degenerate'])
                    attempt[int(rd)][int(sd)]+=1
                    if not rd and not sd:
                        common_n+=1
                        dc=int(sel['score']['content'])-int(ref['score']['content'])
                        df=int(sel['score']['form'])-int(ref['score']['form'])
                        for k,v in (('content',dc),('form',df),('delta',dc-df)):sums[k]+=v
                transitions.append(dict(target=t,prompt_language=lg,reference_arm=arm['arm_id'],
                    n=len(plan['chain_ids']),reference_to_selected=matrix,
                    content_gains=sum(matrix[0][1:]),content_losses=matrix[1][0]+matrix[2][0],
                    gains_with_form=matrix[0][2],gains_without_form=matrix[0][1],
                    degeneration_pairs=attempt,common_attempt_n=common_n,
                    common_attempt_effects_pp={k:100*v/common_n if common_n else None for k,v in sums.items()}))
        for arm in plan['arms'][t]:
            matrix=[[0]*3 for _ in range(3)]
            for c in plan['chain_ids']:
                matrix[state(index[c,t,'es',arm['arm_id']])][state(index[c,t,t,arm['arm_id']])]+=1
            prompt_transitions.append(dict(target=t,arm_id=arm['arm_id'],
                                           n=len(plan['chain_ids']),spanish_to_target=matrix))
    return dict(state_order=['no_content','content_without_form','designated_form'],
                matrix_orientation='rows reference/Spanish; columns selected/target',
                degeneration_order=['nondegenerate','degenerate'],arms=arms,
                transitions=transitions,prompt_transitions=prompt_transitions,
                conditional_scope='Common-attempt effects are post-treatment descriptive diagnostics, not causal corrections. No conditional CI or robustness verdict.',
                draw_scope='Random-arm counts are per draw. Do not sum denominators across draws as independent items.')


def export_tables(output, summary, report):
    save_json(output/'counts_and_transitions.json',report)
    with (output/'arm_rates.csv').open('w',newline='') as f:
        fields=['target','prompt_language','arm_id','kind','seed','n','answer_clusters',
                'content','form','degenerate','eligible','content_without_form',
                'content_pct','form_pct','degenerate_pct']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for arm in report['arms']:
            row={k:arm[k] for k in fields if k in arm}
            row.update({k+'_pct':100*arm[k]/arm['n'] for k in ('content','form','degenerate')})
            writer.writerow(row)
    with (output/'effects.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['target','metric','estimate_pp','ci95_low','ci95_high'])
        for target,result in summary['results'].items():
            for metric,value in result['metrics'].items():
                writer.writerow([target,metric,value['estimate_pp'],*value['ci95_pp']])


def run(*, chains, frozen, plan, decode, score, output, progress=print):
    save_json(output/'prompt_comparison_plan.json',plan)
    save_json(output/'plan_lock.json',dict(sha256=hashlib.sha256((output/'prompt_comparison_plan.json').read_bytes()).hexdigest()))
    save_json(output/'evaluation_items.json',{c:chains[c] for c in plan['chain_ids']})
    records=[]
    with (output/'generations.jsonl').open('x',encoding='utf-8') as journal:
        for index,c in enumerate(plan['chain_ids']):
            for t in plan['targets']:
                for lg in ('es',t):
                    for arm in plan['arms'][t]:
                        generation=decode(c,t,dict(arm,_prompt_language=lg))
                        row=dict(model=frozen['model'],chain=c,target_language=t,prompt_language=lg,
                                 answer_cluster=chains[c]['answer_label_by_lang']['en'],
                                 scored_line=next((s.strip() for s in generation['text'].splitlines() if s.strip()),''),
                                 **arm,generation=generation,score=score(generation['text'],c,t))
                        records.append(row);journal.write(json.dumps(row,ensure_ascii=False)+'\n');journal.flush()
            progress(f'[prompt comparison] items={index+1}/{len(plan["chain_ids"])} records={len(records)}/{plan["expected_records"]}')
    data=vectors(records,chains,plan);save_json(output/'paired_vectors.json',data)
    summary=aggregate([data]);save_json(output/'summary.json',summary)
    export_tables(output,summary,descriptive_report(records,chains,plan))
    return dict(n_records=len(records),cohorts=dict(selected_successes=0,success_shortfall=0),
                replay={'historical_identity':'not certified; prospective comparison only'})
