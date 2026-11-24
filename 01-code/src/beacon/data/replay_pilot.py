"""Fixed paired re-generation sample used to check that frozen inputs reproduce stored outputs.

Selects a small, outcome-independent sample of evaluation chains (lowest SHA-256 of [seed, chain ID]),
re-generates the untreated (U), suppression (S), selected-neuron (I), and random-control (R) arms under
the frozen neuron groups and doses, and compares each output with the stored text. Mismatches are
recorded, never filtered. Standard-library only.
"""
from collections import Counter
from pathlib import Path
import json

from .intervention_cohorts import first_line, save_json, summarize
from .tokenizer_checks import digest


def make_plan(frozen, frozen_sha, pilot_sha, chat, n=24, seed=53):
    ids = frozen['eval_chains']
    if not 1 <= n <= len(ids) or len(set(ids)) != len(ids):
        raise ValueError('Invalid replay sample size or duplicate evaluation IDs')
    selected = sorted(ids, key=lambda c: (digest([seed, c]), c))[:n]
    targets = ['ko', 'zh']
    arms = {}
    for t in targets:
        b = frozen['best'][t]
        arms[t] = [
            dict(arm_id='U', kind='untreated', amp=0., suppression=1., dose=0., seed=None),
            dict(arm_id='S', kind='suppression', amp=0., suppression=b['supp'], dose=0., seed=None,
                 alias_of='U' if b['supp'] == 1 else None),
            dict(arm_id='I', kind='selected', amp=b['amp'], suppression=b['supp'], dose=1., seed=None),
            dict(arm_id='R', kind='random', amp=b['amp'], suppression=b['supp'], dose=1., seed=7)]
    return dict(schema_version=1, model=frozen['model'], frozen_sha256=frozen_sha,
                tokenizer_pilot_sha256=pilot_sha, chat=chat, sample_seed=seed, n=n,
                sampling='lowest SHA256([seed, chain ID]) from historical eval IDs; no output input',
                chain_ids=selected, targets=targets, discovery_targets=list(frozen['eval']),
                arms=arms, max_new_tokens=frozen['protocol']['max_new'],
                historical_chat_status='unrecorded; chosen mode is a prospective replay setting',
                scope='diagnostic replay, not an effect estimate or historical identity certificate')


def validate_plan(plan, frozen, frozen_sha, pilot_sha, chat):
    expected = make_plan(frozen, frozen_sha, pilot_sha, chat, plan['n'], plan['sample_seed'])
    if plan != expected:
        raise ValueError('Replay plan changed or does not match frozen run/tokenizer/chat')


def repetition(tokens):
    longest = run = 0
    previous = None
    for token in tokens:
        run = run + 1 if token == previous else 1
        longest = max(longest, run)
        previous = token
    trigrams = [tuple(tokens[i:i+3]) for i in range(max(0, len(tokens)-2))]
    return dict(token_count=len(tokens), unique_tokens=len(set(tokens)),
                longest_identical_token_run=longest,
                repeated_trigram_occurrences=len(trigrams)-len(set(trigrams)))


def run_replay(*, chains, frozen, plan, decode, score, output, progress=print):
    output = Path(output)
    save_json(output/'replay_plan.json', plan)  # Written before ANY generation.
    save_json(output/'arms.json', plan['arms'])
    records = []
    physical_calls = 0
    with (output/'generations.jsonl').open('x', encoding='utf-8') as fh:
        for chain in plan['chain_ids']:
            baseline = decode(chain, None, plan['arms'][plan['targets'][0]][0])
            physical_calls += 1
            for target in plan['targets']:
                baseline_score = score(baseline['text'], chain, target)
                idx = frozen['eval_chains'].index(chain)
                for arm in plan['arms'][target]:
                    reused = arm['kind'] == 'untreated' or arm.get('alias_of') == 'U'
                    generation = baseline if reused else decode(chain, target, arm)
                    physical_calls += not reused
                    prior = (frozen.get('floors', {}).get(chain) if reused else
                             frozen['eval'][target]['steer_items'][idx]['txt'] if arm['kind'] == 'selected' else
                             frozen['eval'][target]['rand_items'][idx]['txt'] if arm['kind'] == 'random' else None)
                    row = dict(model=frozen['model'], cohort='historical_replay_sample', chain=chain,
                               target_language=target, prompt_language=frozen['protocol']['fail_lang'],
                               **arm, generation=generation, reused_untreated_generation=reused,
                               generation_id=f'{chain}/U' if reused else f'{chain}/{target}/{arm["arm_id"]}',
                               score=score(generation['text'], chain, target), baseline_score=baseline_score,
                               repetition=repetition(generation.get('token_ids', [])),
                               historical_text=prior,
                               historical_score=score(prior, chain, target) if prior is not None else None,
                               historical_text_exact=generation['text'] == prior if prior is not None else None,
                               historical_first_line_exact=first_line(generation['text']) == first_line(prior)
                               if prior is not None else None)
                    records.append(row)
                    fh.write(json.dumps(row, ensure_ascii=False)+'\n'); fh.flush()
                progress(f'[replay] {chain} {target}')
    pairs = []
    for target in plan['targets']:
        lookup = {(r['chain'], r['arm_id']): r for r in records if r['target_language'] == target}
        states = Counter()
        for chain in plan['chain_ids']:
            i, r = lookup[chain, 'I'], lookup[chain, 'R']
            states[f"{int(not i['score']['degenerate'])}{int(not r['score']['degenerate'])}"] += 1
        pairs.append(dict(target_language=target, n=len(plan['chain_ids']),
                          attempt_states_I_R={k: states[k] for k in ('11', '10', '01', '00')}))
    compared = [r for r in records if r['historical_text_exact'] is not None and r['arm_id'] in ('I', 'R')]
    replay = dict(compared_intervention_records=len(compared),
                  text_mismatches=sum(not r['historical_text_exact'] for r in compared),
                  first_line_mismatches=sum(not r['historical_first_line_exact'] for r in compared),
                  physical_generation_calls=physical_calls, paired_attempts=pairs,
                  note='Mismatches are retained; no filtering or recalibration. U/S aliases are not independent draws.')
    save_json(output/'summary.json', summarize(records))
    save_json(output/'replay_validation.json', replay)
    return dict(n_records=len(records), replay=replay, cohorts=dict(selected_successes=0, success_shortfall=0))
