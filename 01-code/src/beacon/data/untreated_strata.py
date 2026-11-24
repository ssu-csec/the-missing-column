"""Untreated-output strata, paired intervention collection, and descriptive cluster intervals.

Every inventory item is screened once without intervention and assigned to stratum A (content
correct), B (content incorrect, non-degenerate), or C (degenerate; capped, hash-ordered sample).
Strata are hashed and locked before any intervention. Selected-neuron (I) and random-control (R)
arms are then generated for each stratum, and paired I-vs-R effects are summarized with
answer-cluster bootstrap intervals (`cluster_interval`, fixed RNG seed).
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import random

from .intervention_cohorts import save_json

TARGETS = ("ko", "zh")
SEEDS = (7, 17, 27, 37, 47)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_strata(chains, frozen, baseline, cap=54, seed=53):
    if set(baseline) != set(chains):
        raise ValueError("Screen must cover every inventory item exactly once")
    cluster = lambda c: chains[c]["answer_label_by_lang"].get("en") or c
    excluded_clusters = {cluster(c) for c in frozen["calib_chains"]}
    pools = {k: [] for k in "ABC"}
    excluded = []
    for chain in sorted(chains):
        if cluster(chain) in excluded_clusters:
            excluded.append(chain)
            continue
        s = baseline[chain]["scores"]["ko"]
        other = baseline[chain]["scores"]["zh"]
        if any(s[k] != other[k] for k in ("content", "degenerate")):
            raise ValueError("Content/degeneration must be target-independent")
        pools["C" if s["degenerate"] else "A" if s["content"] else "B"].append(chain)
    ordered = sorted(pools["C"], key=lambda c: (
        hashlib.sha256(json.dumps([seed, c], ensure_ascii=False,
                                 separators=(",", ":")).encode()).hexdigest(), c))
    selected = dict(A=pools["A"], B=pools["B"], C=ordered[:cap])
    return dict(selected=selected, pools=pools, excluded=excluded,
                cluster_by_chain={c: cluster(c) for c in chains},
                historical_eval=list(frozen["eval_chains"]),
                counts={k: len(v) for k, v in selected.items()},
                sampling=dict(cap=cap, seed=seed, order='SHA256(UTF8(JSON([seed,chain],ensure_ascii=False,separators=(",",":"))))'),
                selection="Untreated only; exclude calibration answer clusters; retain historical eval IDs")


def extend_random_groups(bundle, widths, draw, seeds=SEEDS):
    """Re-draw each seed's random group in the full target/layer RNG order, checking any saved IDs."""
    for seed in seeds:
        generated = draw(seed, bundle["selected"], bundle["targets"], widths)
        key = str(seed)
        if key in bundle["random"]:
            if bundle["random"][key] != generated:
                raise ValueError(f"Saved random group differs for seed {seed}")
        else:
            bundle["random"][key] = generated
    return bundle


def read_screen(directory, chains, context):
    manifest = json.loads((directory / "screen_manifest.json").read_text())
    if (manifest["status"] != "complete" or manifest["context"] != context
            or manifest["sha256"] != digest(directory / "screening.jsonl")):
        raise ValueError("Screening context/hash/status differs")
    records = [json.loads(line) for line in (directory / "screening.jsonl").read_text().splitlines()]
    result = {r["chain"]: r for r in records}
    if len(result) != len(records) or set(result) != set(chains):
        raise ValueError("Incomplete/duplicate screening items")
    return result


def collect(*, chains, frozen, arms, decode, score, output, context,
            screen_input=None, screen_only=False, progress=print):
    u = dict(arm_id="U", kind="untreated", amp=0., suppression=1., seed=None)
    if screen_input:
        baseline = read_screen(screen_input, chains, context)
    else:
        baseline = {}
    with (output / "screening.jsonl").open("x") as journal:
        for i, chain in enumerate(sorted(chains), 1):
            if chain not in baseline:
                generation = decode(chain, "ko", u)
                baseline[chain] = dict(chain=chain, arm_id="U", **generation,
                    scores={t: score(generation["text"], chain, t) for t in TARGETS})
            journal.write(json.dumps(baseline[chain], ensure_ascii=False) + "\n")
            journal.flush()
            if i % 10 == 0 or i == len(chains):
                progress(f"[screen] {i}/{len(chains)}")
    save_json(output / "screen_manifest.json", dict(status="complete", context=context,
              n=len(baseline), sha256=digest(output / "screening.jsonl"),
              reused_from=str(screen_input) if screen_input else None))
    cohorts = select_strata(chains, frozen, baseline)
    save_json(output / "cohorts.json", cohorts)
    cohort_hash = digest(output / "cohorts.json")
    save_json(output / "cohort_lock.json", dict(sha256=cohort_hash))
    progress(f"[cohorts] {cohorts['counts']}; frozen before intervention")
    if screen_only:
        return dict(n_records=len(baseline), cohorts=dict(selected_successes=cohorts['counts']['A'], success_shortfall=0), replay=None)
    records = []
    expected = sum(cohorts["counts"].values()) * 12
    with (output / "interventions.jsonl").open("x") as journal:
        for stratum, ids in cohorts["selected"].items():
            for chain in ids:
                for target in TARGETS:
                    for arm in arms[target]:
                        if arm["kind"] == "untreated":
                            continue
                        generation = decode(chain, target, arm)
                        row = dict(chain=chain, stratum=stratum, target=target,
                                   arm=arm, **generation, score=score(generation["text"], chain, target))
                        records.append(row)
                        journal.write(json.dumps(row, ensure_ascii=False) + "\n")
                        journal.flush()
                        if len(records) % 10 == 0 or len(records) == expected:
                            progress(f"[interventions] {len(records)}/{expected}")
    if digest(output / "cohorts.json") != cohort_hash:
        raise ValueError("Cohort manifest changed during intervention")
    analyze(baseline, records, cohorts, output)
    return dict(n_records=len(baseline) + len(records),
                cohorts=dict(selected_successes=cohorts['counts']['A'], success_shortfall=0), replay=None)


def cluster_interval(values, clusters, draws=10000):
    grouped = defaultdict(list)
    for value, cluster in zip(values, clusters):
        grouped[cluster].append(value)
    if len(grouped) < 2:
        return None
    groups = [(sum(v), len(v)) for v in grouped.values()]
    rng = random.Random(20260916)
    samples = []
    for _ in range(draws):
        selected = rng.choices(groups, k=len(groups))
        samples.append(sum(s for s, _ in selected) / sum(n for _, n in selected))
    samples.sort()
    return [samples[int((draws-1)*.025)], samples[int((draws-1)*.975)]]


def analyze(baseline, records, cohorts, output):
    index = {}
    for row in records:
        arm = row["arm"]
        label = "I" if arm["kind"] == "selected" else f"R{arm['seed']}"
        key = (row["chain"], row["target"], label)
        if key in index:
            raise ValueError("Duplicate intervention cell")
        index[key] = row
    expected = {(c, t, a) for ids in cohorts["selected"].values() for c in ids
                for t in TARGETS for a in ("I", *[f"R{s}" for s in SEEDS])}
    if set(index) != expected:
        raise ValueError("Missing or extra intervention cells")
    summary, paired = {}, []
    def get(c, t, a):
        if a == "U":
            return baseline[c]["scores"][t], baseline[c]["stop_reason"]
        row = index[c, t, a]
        return row["score"], row["stop_reason"]
    for sensitivity in (False, True):
        table = {}
        for stratum, original in cohorts["selected"].items():
            ids = [c for c in original if not sensitivity or c not in cohorts["historical_eval"]]
            for target in TARGETS:
                cells = {}
                for arm in ("U", "I", *[f"R{s}" for s in SEEDS]):
                    scored = [get(c, target, arm) for c in ids]
                    cells[arm] = dict(n=len(ids),
                        content=sum(s["content"] for s, _ in scored),
                        degenerate=sum(s["degenerate"] for s, _ in scored),
                        eligible=sum(s["eligible"] for s, _ in scored),
                        form=sum(bool(s["form"]) for s, _ in scored),
                        token_cap=sum(stop == "max_new_tokens" for _, stop in scored),
                        DCF=dict(Counter(f"{int(s['degenerate'])}{int(s['content'])}{int(bool(s['form']))}" for s, _ in scored)))
                effects = {}
                for metric in ("content", "degenerate", "form"):
                    eligible = [c for c in ids if metric != "form" or get(c, target, "U")[0]["eligible"]]
                    values = []
                    iu, ru = [], []
                    seed_effects = {str(s): [] for s in SEEDS}
                    for c in eligible:
                        i = float(get(c, target, "I")[0][metric])
                        u = float(get(c, target, "U")[0][metric])
                        rs = [float(get(c, target, f"R{s}")[0][metric]) for s in SEEDS]
                        values.append(i - sum(rs)/len(rs))
                        iu.append(i-u); ru.append(sum(rs)/len(rs)-u)
                        for seed, r in zip(SEEDS, rs):
                            seed_effects[str(seed)].append(i-r)
                    mean = lambda xs: sum(xs)/len(xs) if xs else None
                    effects[metric] = dict(n=len(eligible), I_minus_R=mean(values), I_minus_U=mean(iu), R_minus_U=mean(ru),
                        cluster_ci=cluster_interval(values, [cohorts['cluster_by_chain'][c] for c in eligible]),
                        I_minus_R_by_seed={s: mean(v) for s, v in seed_effects.items()})
                states = {}
                for seed in SEEDS:
                    counts = Counter()
                    common = []
                    for c in ids:
                        i = get(c, target, "I")[0]; r = get(c, target, f"R{seed}")[0]
                        state = f"{int(not i['degenerate'])}{int(not r['degenerate'])}"
                        counts[state] += 1
                        counts[f"{state}_content_gain"] += int(i['content'] and not r['content'])
                        counts[f"{state}_content_loss"] += int(r['content'] and not i['content'])
                        if state == "11":
                            common.append(int(i['content'])-int(r['content']))
                        if not sensitivity:
                            paired.append(dict(chain=c, cluster=cohorts['cluster_by_chain'][c], stratum=stratum,
                                target=target, seed=seed, historical_eval=c in cohorts['historical_eval'], attempt_state=state,
                                **{f"{arm}_{k}": get(c,target,arm if arm != 'R' else f'R{seed}')[0][k]
                                   for arm in ('U','I','R') for k in ('content','form','degenerate','eligible')}))
                    states[str(seed)] = dict(counts=counts, common_attempt_n=len(common),
                        common_attempt_delta=sum(common)/len(common) if common else None)
                r_mean = {metric: sum(cells[f'R{s}'][metric] for s in SEEDS)/len(SEEDS)
                          for metric in ('content','form','degenerate','token_cap')}
                table[f"{stratum}/{target}"] = dict(n=len(ids), answer_clusters=len({cohorts['cluster_by_chain'][c] for c in ids}),
                    arms=cells, random_mean_counts=r_mean, effects=effects, paired_states=states)
        summary['exclude_historical_eval' if sensitivity else 'primary'] = table
    summary['interpretation'] = 'Descriptive, post-hoc supplement. Cluster CI conditional on five fixed random groups; no independent seed replication of items. No pooled stratum effect. Nondegenerate is detector-defined.'
    save_json(output / "summary.json", summary)
    with (output / "paired_items.csv").open("w", newline="") as handle:
        fields = list(paired[0]) if paired else ['chain','target','seed']
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(paired)
    lines = ['# Untreated strata results', '', summary['interpretation'], '',
             '| Stratum/target | N | Clusters | Arm | Content | Form | Degenerate | Token cap |',
             '|---|---:|---:|---|---:|---:|---:|---:|']
    for key, row in summary['primary'].items():
        for arm, cell in row['arms'].items():
            lines.append(f"| {key} | {row['n']} | {row['answer_clusters']} | {arm} | {cell['content']} | {cell['form']} | {cell['degenerate']} | {cell['token_cap']} |")
    (output / 'RESULTS.md').write_text('\n'.join(lines)+'\n')
