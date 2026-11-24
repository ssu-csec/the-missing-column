"""Japanese 70B TDV check: frozen item reconstruction and paired summaries."""
from .untreated_strata import cluster_interval

LANGS=['en','es','ko','zh','ja']
SEEDS=[0,17,27,37,47]
TRANS={
    'ko':'Instruction: Translate the phrase into Korean.\nPhrase: {x}\nTranslation:',
    'ja':'Instruction: Translate the phrase into Japanese.\nPhrase: {x}\nTranslation:',
    'zh':'Instruction: Translate the phrase into Chinese.\nPhrase: {x}\nTranslation:',
    'es':'Instruction: Translate the phrase into Spanish.\nPhrase: {x}\nTranslation:',
}


def make_plan(chains, single, historical, screen_n=200):
    if historical['model']!='llama31_70_it' or historical['protocol']['primary_layer']!=60:
        raise ValueError('Expected original 70B layer-60 TDV artifact')
    if historical['protocol']['construction']!='240T/300R forwards, langs-averaged, raw addition, continuous last-pos injection':
        raise ValueError('Unexpected original construction')
    all_ch=[c for c in chains if all(chains[c]['prompts'].get(l) and chains[c]['answer_label_by_lang'].get(l) for l in LANGS)]
    old_pools={t:[r['chain'] for r in historical['by_target'][t]['items']] for t in ('ko','ja')}
    if any(len(v)!=len(set(v)) for v in old_pools.values()):raise ValueError('Duplicate historical evaluation IDs')
    used=set().union(*[set(v) for v in old_pools.values()])
    if used-set(all_ch):raise ValueError('Historical pool missing from inventory')
    construction=[c for c in all_ch if c not in used][:60]
    if len(construction)!=60:raise ValueError('Not enough construction items')
    if any(c not in single or any(l not in single[c] for l in LANGS) for c in construction):
        raise ValueError('Missing construction recall prompts')
    screen=[c for c in all_ch if c not in construction and c in single and single[c].get('ja') and single[c].get('en')][:screen_n]
    if len(screen)!=screen_n:raise ValueError('Insufficient screening items')
    clusters={c:chains[c]['answer_label_by_lang']['en'] for c in all_ch}
    construction_clusters={clusters[c] for c in construction}
    disjoint=[c for c in screen if clusters[c] not in construction_clusters]
    return dict(model='llama31_70_it',target='ja',chat=True,layer=60,scale=1.,seeds=SEEDS,
                construction=construction,screening=screen,cluster_by_chain=clusters,
                cluster_disjoint_screening=disjoint,construction_clusters=sorted(construction_clusters),
                historical_two_hop_pools=old_pools,
                decode=dict(max_new_tokens=24,min_new_tokens=2,do_sample=False,repetition_penalty=1.3,no_repeat_ngram_size=3),
                expected_forwards=dict(translation=240,recall=300),expected_generations=8*screen_n,
                primary='Japanese single-hop U-fail and English-U-correct; TDV scale 1 vs seed-0 norm-matched random',
                secondary='Mean of 5 random directions, construction-cluster-disjoint subset, all-screen and U-success retention',
                minimum_absolute_recovery_threshold=None,
                historical_identity='Construction reconstructed from preserved two-hop pool IDs; original vector/runtime not saved')


def analyze(records,plan,draws=10000):
    by={(r['chain'],r['arm']):r for r in records}
    arms=['U_ja','U_en','TDV']+[f'R{s}' for s in plan['seeds']]
    expected={(c,a) for c in plan['screening'] for a in arms}
    if set(by)!=expected or len(records)!=len(expected):raise ValueError('Incomplete/duplicate generation grid')
    fail=[c for c in plan['screening'] if not by[c,'U_ja']['legacy_ilr'] and by[c,'U_en']['english_correct']]
    groups=dict(all_screen=plan['screening'],implementation_failures=fail,
                implementation_failures_cluster_disjoint=[c for c in fail if c in set(plan['cluster_disjoint_screening'])],
                untreated_ja_success=[c for c in plan['screening'] if by[c,'U_ja']['legacy_ilr']])
    result={}
    for name,ids in groups.items():
        clusters=[plan['cluster_by_chain'][c] for c in ids]
        if not ids:
            result[name]=dict(n=0,status='no_eligible_items',rates=None,contrasts=None);continue
        def estimate(xs):
            ci=cluster_interval(xs,clusters,draws=draws)
            return dict(percent_or_pp=100*sum(xs)/len(xs),ci95=[100*x for x in ci] if ci is not None else None)
        get=lambda c,a:float(by[c,a]['legacy_ilr'])
        rates={a:estimate([get(c,a) for c in ids]) for a in arms if a!='U_en'}
        contrasts={a:estimate([get(c,'TDV')-get(c,a) for c in ids]) for a in ['U_ja']+[f'R{s}' for s in plan['seeds']]}
        contrasts['R_mean5']=estimate([get(c,'TDV')-sum(get(c,f'R{s}') for s in plan['seeds'])/len(plan['seeds']) for c in ids])
        current={}
        for metric in ('content','form'):
            eligible=[c for c in ids if by[c,'TDV']['locked_score']['eligible']]
            current[metric]=dict(n=len(eligible),counts={a:sum(by[c,a]['locked_score'][metric] for c in eligible)
                       for a in arms if a!='U_en'})
        result[name]=dict(n=len(ids),clusters=len(set(clusters)),rates=rates,contrasts=contrasts,
                         locked_policy_diagnostic=current,
                         degenerate_counts={a:sum(by[c,a]['degenerate'] for c in ids) for a in arms},
                         n_TDV_recovered=sum(by[c,'TDV']['legacy_ilr'] for c in ids))
    return dict(subsets=result,bootstrap_draws=draws,
                screen_counts=dict(total=len(plan['screening']),eligible_failures=len(fail),
                  japanese_U_correct=sum(by[c,'U_ja']['legacy_ilr'] for c in plan['screening']),
                  english_U_correct=sum(by[c,'U_en']['english_correct'] for c in plan['screening']),
                  construction_cluster_disjoint=len(plan['cluster_disjoint_screening'])),
                interpretation='Descriptive paired answer-cluster intervals conditional on reconstructed vector and fixed random directions. Primary seed is 0; no any-seed/any-scale PASS rule. Absolute rates must accompany differences.')
