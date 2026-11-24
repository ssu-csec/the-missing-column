"""Generation-length check for a fixed paired sample under frozen neuron groups.

Compares a reference set of 32-token generations against a re-generated 32-token budget (exact
token reproduction) and a 64-token budget (the 32-token output must be a prefix of the 64-token
output, or identical when generation stopped at EOS). Content, form, and degeneration are scored
with the unchanged first-line scorer, so a longer budget cannot replace the first answer.
"""
from collections import Counter
import json
from pathlib import Path

from .intervention_cohorts import save_json


def read_records(path, plan):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    indexed = {(r['chain'], r['target_language'], r['arm_id']): r for r in rows}
    expected = {(c,t,a['arm_id']) for c in plan['chain_ids'] for t in plan['targets'] for a in plan['arms'][t]}
    if len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError('Missing, duplicated or unexpected comparison records')
    unique = {}
    for r in rows:
        gid = r['generation_id']
        if gid in unique and unique[gid] != r['generation']:
            raise ValueError('Shared U/S generation IDs have inconsistent outputs')
        unique[gid] = r['generation']
    return indexed


def compare_length_runs(reference, output, plan):
    output = Path(output)
    old = read_records(Path(reference)/'generations.jsonl', plan)
    short = read_records(output/'length_32/generations.jsonl', plan)
    long = read_records(output/'length_64/generations.jsonl', plan)
    compared = []
    for key, s in short.items():
        o, l = old[key], long[key]
        a, b, c = (x['generation']['token_ids'] for x in (o,s,l))
        if o['generation']['prompt_token_ids'] != s['generation']['prompt_token_ids'] or s['generation']['prompt_token_ids'] != l['generation']['prompt_token_ids']:
            raise ValueError('Prompt IDs changed between comparison runs')
        # If the short run stopped early at EOS, the long run must stop there too.
        prefix = (c == b if s['generation']['stop_reason'] == 'eos' else c[:len(b)] == b)
        compared.append(dict(chain=key[0],target_language=key[1],arm_id=key[2],
            generation_id=s['generation_id'],
            reference_32_token_exact=a == b,
            reference_32_text_exact=o['generation']['text'] == s['generation']['text'],
            length_64_preserves_32_prefix=prefix,
            n_tokens_32=len(b),n_tokens_64=len(c),
            stop_32=s['generation']['stop_reason'],stop_64=l['generation']['stop_reason'],
            reference_score=o['score'],score_32=s['score'],score_64=l['score']))
    unique = {r['generation_id']:r for r in compared}
    summaries = []
    for t in plan['targets']:
        for arm in plan['arms'][t]:
            a = arm['arm_id']
            rs = [r for r in compared if r['target_language']==t and r['arm_id']==a]
            per_budget = {}
            for name, source in [('reference',old),('32',short),('64',long)]:
                records = [r for r in source.values() if r['target_language']==t and r['arm_id']==a]
                eligible = [r for r in records if r['score']['eligible']]
                per_budget[name] = dict(n=len(records),n_eligible=len(eligible),
                    content=sum(r['score']['content'] for r in eligible),
                    form=sum(bool(r['score']['form']) for r in eligible),
                    degenerate=sum(r['score']['degenerate'] for r in records),
                    stops=dict(Counter(r['generation']['stop_reason'] for r in records)))
            summaries.append(dict(target_language=t,arm_id=a,n=len(rs),
                reference_32_token_exact=sum(r['reference_32_token_exact'] for r in rs),
                length_64_prefix_exact=sum(r['length_64_preserves_32_prefix'] for r in rs),
                score_changes_32_to_64=sum(r['score_32']!=r['score_64'] for r in rs),
                budgets=per_budget))
    result = dict(status='complete',n_records=len(compared),n_unique_generations_per_budget=len(unique),
        reference_32_exact_unique=sum(r['reference_32_token_exact'] for r in unique.values()),
        length_64_prefix_exact_unique=sum(r['length_64_preserves_32_prefix'] for r in unique.values()),
        all_reference_32_exact=all(r['reference_32_token_exact'] for r in unique.values()),
        all_length_prefixes_exact=all(r['length_64_preserves_32_prefix'] for r in unique.values()),
        summary=summaries,
        interpretation='Exact token comparisons use raw stored outputs from the preceding GPU pilot. C/F/degeneration use the unchanged first-line scorer; later answers do not replace the first answer. U/S aliases are not independent generations.')
    with (output/'paired_comparison.jsonl').open('x',encoding='utf-8') as fh:
        for row in compared:
            fh.write(json.dumps(row,ensure_ascii=False)+'\n')
    save_json(output/'length_check_summary.json',result)
    return result
