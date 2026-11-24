#!/usr/bin/env python3
"""Language-neuron audit with frozen groups on failure items and untreated-content successes.

The neuron groups and doses are taken from a frozen language_neurons.py result; nothing is recalibrated.
Default arms: untreated, suppression-only where applicable, selected neurons, and five random draws at the
frozen dose. Optional dose curves use the same arms on both cohorts. Residual-norm (energy) matching is not
implemented here. Additional modes: a fixed small replay plan, a 32/64-token length check, an untreated-strata
audit, and checkpoint stability. --dry-run validates inputs without torch,
model loading or writes.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import sys

CODE = Path(__file__).resolve().parents[1]
ROOT = CODE.parent
RESULTS = ROOT / "02-data/results"
sys.path.insert(0, str(CODE / "src"))

from beacon.data.intervention_cohorts import build_arms, run_audit, save_json
from beacon.data.tokenizer_checks import tokenizer_identity, validate_pilot_files
from beacon.data.replay_pilot import validate_plan, run_replay
from beacon.data.replay_length_check import compare_length_runs, read_records
from beacon.data.untreated_strata import collect as collect_strata, extend_random_groups
from beacon.data.checkpoint_stability import make_plan as stability_plan, run as run_stability


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def restore_groups(loaded, neuron_module, frozen, pairs, seeds, torch, chat, expected_markers):
    """Reconstruct neuron selection and random-group draws (language_neurons.py algorithm and generator order) for all targets."""
    targets = list(frozen["eval"])
    prompt = frozen["protocol"]["fail_lang"]
    langs = sorted(set(targets + [prompt, frozen["protocol"]["pivot"]]))
    start, end = frozen["band"]
    band = list(range(start, end + 1))
    print(f"[discovery] activation means: {len(pairs)} pairs, {len(langs)} languages", flush=True)
    means, counts = neuron_module._lang_activation_means(loaded, pairs, band, langs, chat)
    selected, marker_counts = {}, {}
    for target in targets:
        print(f"[discovery] selecting {target}", flush=True)
        marker_ids = neuron_module._marker_token_ids(loaded, target)
        if marker_ids != expected_markers[target]:
            raise ValueError(f"Marker token IDs differ from pilot: {target}")
        marker_counts[target] = len(marker_ids)
        selected[target] = neuron_module._select_lang_neurons(
            loaded, means, band, target, langs, frozen["k"], marker_ids)
        if not any(selected[target].values()):
            raise ValueError(f"No selected neurons for {target}")
    if marker_counts != frozen["marker_counts"]:
        raise ValueError("Marker vocabulary counts differ from the frozen run; inspect tokenizer/version")
    random_groups = {}
    layers = neuron_module._decoder_layers(loaded.model)
    for seed in seeds:
        generator = torch.Generator().manual_seed(seed)
        by_target = {}
        for target in targets:
            by_target[target] = {}
            for layer, indices in selected[target].items():
                width = layers[int(layer)].mlp.down_proj.weight.shape[1]
                by_target[target][layer] = torch.randperm(width, generator=generator)[:len(indices)].tolist()
        random_groups[str(seed)] = by_target
    return dict(selected=selected, random=random_groups, suppression=selected[prompt],
                discovery_counts=counts, marker_counts=marker_counts, marker_token_ids=expected_markers)


def normalize_and_validate_groups(bundle, loaded, neuron_module, targets, seeds):
    layers = neuron_module._decoder_layers(loaded.model)

    def normalize(by_layer):
        result = {int(k): list(v) for k, v in by_layer.items()}
        for layer, ids in result.items():
            if not 0 <= layer < len(layers):
                raise ValueError("Invalid layer in group bundle")
            width = layers[layer].mlp.down_proj.weight.shape[1]
            if len(set(ids)) != len(ids) or any(not isinstance(i, int) or not 0 <= i < width for i in ids):
                raise ValueError("Invalid/duplicate neuron IDs in group bundle")
        return result

    bundle["selected"] = {t: normalize(bundle["selected"][t]) for t in targets}
    bundle["suppression"] = normalize(bundle["suppression"])
    bundle["random"] = {str(s): {t: normalize(bundle["random"][str(s)][t]) for t in targets}
                        for s in seeds}
    for seed in seeds:
        for target in targets:
            selected, random_group = bundle["selected"][target], bundle["random"][str(seed)][target]
            if {k: len(v) for k, v in selected.items()} != {k: len(v) for k, v in random_group.items()}:
                raise ValueError("Random groups must match selected layer/group sizes")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="New output directory; never overwrite")
    parser.add_argument("--pairs", type=Path, default=ROOT / "02-data/inputs/parallel_pairs.jsonl")
    parser.add_argument("--groups", type=Path, help="Reuse this runner's group bundle, bound to the frozen input")
    parser.add_argument("--tokenizer-pilot", type=Path, help="PASS report.json from tokenizer_preflight.py; required for GPU runs")
    parser.add_argument("--chat", action=argparse.BooleanOptionalAction, required=True,
                        help="Chat-template setting (must be given explicitly; the frozen result file does not record it)")
    parser.add_argument("--n-success", type=int, default=120, help="Success sample cap; 0 keeps all candidates")
    parser.add_argument("--cohort-seed", type=int, default=53)
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--dose-scales", type=float, nargs="+", default=[1.0])
    parser.add_argument("--suppression-modes", nargs="+", choices=["original", "off"], default=["original"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replay-plan", type=Path, help="Fixed paired sample plan; skips success screening")
    parser.add_argument("--length-check-reference", type=Path,
                        help="Completed preceding 32-token replay directory; reuse its groups and run 32 then 64 tokens")
    parser.add_argument("--strata-reference", type=Path, help="Completed 32-token replay; run untreated A/B/C audit with its groups/runtime")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--screening-input", type=Path, help="Reuse a completed, context-matched strata screen")
    parser.add_argument("--checkpoint-stability", action="store_true", help="Full frozen evaluation IDs, ko/zh, five random control seeds")
    parser.add_argument("--prompt-comparison", action="store_true", help="Frozen es versus target prompts; requires saved groups")
    args = parser.parse_args()
    global stability_plan, run_stability
    if args.prompt_comparison:
        if not args.groups:
            parser.error("Prompt comparison requires saved groups; rediscovery is prohibited")
        args.checkpoint_stability = True
        from beacon.data.prompt_comparison import make_plan as stability_plan, run as run_stability
    frozen = json.loads(args.frozen_run.read_text(encoding="utf-8"))
    if args.checkpoint_stability:
        if args.replay_plan or args.length_check_reference or args.strata_reference or args.screen_only or args.screening_input:
            parser.error('Checkpoint stability cannot be combined with other experiment modes')
        if not args.tokenizer_pilot:
            parser.error('Checkpoint stability requires tokenizer pilot, including dry runs')
        args.control_seeds = [7,17,27,37,47]
        args.dose_scales = [1.0]
        args.suppression_modes = ['original']
    reference_manifest = None
    if args.strata_reference:
        if args.replay_plan or args.length_check_reference or args.chat or frozen['model'] != 'gemma4_31b':
            parser.error('Strata audit requires Gemma base raw and no replay/length-check mode')
        if (frozen['protocol']['max_new'] != 32 or
                any(frozen['best'][t]['amp'] != 4 or frozen['best'][t]['supp'] != 1 for t in ('ko','zh'))):
            parser.error('Strata audit requires the frozen 32-token amp=4/supp=1 settings')
        reference_manifest = json.loads((args.strata_reference/'manifest.json').read_text())
        if (reference_manifest['status'] != 'complete' or reference_manifest.get('mode') != 'small_replay'
                or reference_manifest['max_new_tokens'] != 32):
            parser.error('Strata reference must be the completed 32-token replay')
        args.groups = args.strata_reference/'groups.json'
        if sha(args.groups) != reference_manifest['groups_sha256']:
            parser.error('Reference group hash differs')
        args.control_seeds = [7,17,27,37,47]
        args.dose_scales = [1.0]
        args.suppression_modes = ['original']
    elif args.screen_only or args.screening_input:
        parser.error('Screen options require --strata-reference')
    if args.length_check_reference:
        if not args.replay_plan or frozen['protocol']['max_new'] != 32:
            parser.error('Length check requires the frozen 32-token replay plan')
        reference_manifest = json.loads((args.length_check_reference/'manifest.json').read_text())
        if reference_manifest['status'] != 'complete' or reference_manifest.get('mode') != 'small_replay':
            parser.error('Reference must be a completed small replay')
        reference_groups = args.length_check_reference/'groups.json'
        if args.groups and sha(args.groups) != sha(reference_groups):
            parser.error('Length check must reuse exactly the reference groups')
        args.groups = reference_groups
    replay_plan = None
    if args.replay_plan:
        if not args.tokenizer_pilot:
            parser.error("Replay requires --tokenizer-pilot even for dry-run")
        replay_plan = json.loads(args.replay_plan.read_text(encoding="utf-8"))
        validate_plan(replay_plan, frozen, sha(args.frozen_run), sha(args.tokenizer_pilot), args.chat)
        if reference_manifest:
            if (reference_manifest['replay_plan'] != replay_plan
                    or reference_manifest['groups_sha256'] != sha(args.groups)
                    or reference_manifest['max_new_tokens'] != 32):
                parser.error('Reference plan/groups/token budget differs')
            read_records(args.length_check_reference/'generations.jsonl', replay_plan)
        args.control_seeds = [7]
        args.dose_scales = [1.0]
        args.suppression_modes = ["original"]
    if args.n_success < 0 or 7 not in args.control_seeds:
        parser.error("n-success must be >= 0 and control-seeds must include seed 7")
    targets = list(frozen["eval"])
    if targets != ["ko", "ja", "zh", "es"]:
        parser.error("Expected original target/RNG order ko ja zh es; do not rediscover a subset")
    if frozen["protocol"].get("fail_lang") != "es" or frozen["protocol"].get("rand_seed") != 7:
        parser.error("This runner expects the preserved Spanish-prompt seed-7 protocol")
    if set(frozen["calib_chains"]) & set(frozen["eval_chains"]):
        parser.error("Frozen calibration and evaluation overlap")
    for key in ("calib_chains", "eval_chains"):
        if len(frozen[key]) != len(set(frozen[key])):
            parser.error(f"Duplicate IDs in {key}")
    for target in targets:
        for key in ("steer_items", "rand_items"):
            if len(frozen["eval"][target][key]) != len(frozen["eval_chains"]):
                parser.error(f"Frozen generation alignment mismatch: {target}/{key}")
    arms = build_arms(frozen["best"], targets, args.control_seeds,
                      args.dose_scales, args.suppression_modes)
    # Load the released locked scorer as a module; its report-level checks are not executed.
    scoring = module_at("cohort_separation", ROOT/'01-code/analysis'/"paired_effects.py").V
    degeneracy = module_at("cohort_degeneracy", ROOT/'01-code/analysis'/"degenerate_outputs.py")
    if scoring.MATCHER_POLICY != "attribution_strict":
        parser.error("Unset AOL_MATCHER_POLICY or use attribution_strict for the primary audit")
    chains = scoring.CHAINS
    if set(frozen["calib_chains"] + frozen["eval_chains"]) - set(chains):
        parser.error("Frozen chains are missing from the released scoring inventory")
    input_paths = [args.frozen_run, args.pairs, ROOT / "02-data/inputs/chain_lang_prompts.json",
                   CODE / "configs/models.yaml", Path(__file__),
                   CODE / "scripts/language_neurons.py",
                   CODE / "src/beacon/data/intervention_cohorts.py",
                   ROOT/'01-code/analysis'/"verify_numbers.py", ROOT/'01-code/analysis'/"paired_effects.py", ROOT/'01-code/analysis'/"degenerate_outputs.py"]
    if args.groups:
        input_paths.append(args.groups)
    if args.replay_plan:
        input_paths.extend([args.replay_plan, CODE / "src/beacon/data/replay_pilot.py"])
    if args.length_check_reference:
        input_paths.extend([args.length_check_reference/name for name in ('manifest.json','generations.jsonl')])
        input_paths.append(CODE/'src/beacon/data/replay_length_check.py')
    if args.strata_reference:
        input_paths.extend([args.strata_reference/'manifest.json', CODE/'src/beacon/data/untreated_strata.py'])
    if args.checkpoint_stability:
        input_paths.extend([CODE/'src/beacon/data/checkpoint_stability.py',CODE/'src/beacon/data/untreated_strata.py'])
    pilot = None
    if args.tokenizer_pilot:
        pilot = validate_pilot_files(args.tokenizer_pilot)
        if pilot["model"] != frozen["model"] or pilot["chat"] != args.chat:
            parser.error("Tokenizer pilot model/chat differs from this run")
        expected = {"frozen_run": args.frozen_run, "pairs": args.pairs,
                    "chains": ROOT / "02-data/inputs/chain_lang_prompts.json",
                    "prompting": CODE / "src/beacon/data/prompting.py",
                    "tokenizer_checks": CODE / "src/beacon/data/tokenizer_checks.py",
                    "neuron_discovery": CODE / "scripts/language_neurons.py"}
        if any(pilot["input_sha256"].get(key) != sha(path) for key, path in expected.items()):
            parser.error("Tokenizer pilot inputs/code changed; prepare a new pilot")
        input_paths.append(args.tokenizer_pilot)
    manifest = dict(status="planned", model=frozen["model"], targets=targets,
                    chat=args.chat, historical_chat_setting="not recorded; supplied explicitly",
                    frozen_best=frozen["best"], control_seeds=args.control_seeds,
                    dose_scales=args.dose_scales, suppression_modes=args.suppression_modes,
                    n_historical_failures=len(frozen["eval_chains"]), n_success_requested=args.n_success,
                    success_count="determined from untreated-only screen", cohort_seed=args.cohort_seed,
                    max_new_tokens=frozen["protocol"]["max_new"], matcher="attribution_strict",
                    magnitude_matching=False, calibration_reselection=False,
                    tokenizer_pilot_sha256=sha(args.tokenizer_pilot) if args.tokenizer_pilot else None,
                    arms=arms, input_sha256={str(p.resolve()): sha(p) for p in input_paths})
    if replay_plan:
        manifest.update(mode="small_replay", replay_plan=replay_plan, arms=replay_plan['arms'],
                        success_count="not screened in replay pilot", n_success_requested=0)
    if args.length_check_reference:
        manifest.update(mode='fixed_groups_length_check', generation_budgets=[32,64],
                        reference_directory=str(args.length_check_reference),
                        group_discovery=False, expected_physical_generations=240)
    if args.strata_reference:
        manifest.update(mode='untreated_strata', evaluation_targets=['ko','zh'],
                        reference_directory=str(args.strata_reference), group_discovery=False,
                        screen_only=args.screen_only, screening_input=str(args.screening_input) if args.screening_input else None,
                        success_count='all nondegenerate correct after calibration-cluster exclusion',
                        degenerate_sample_cap=54, cohort_seed=53)
    full_plan = None
    if args.checkpoint_stability:
        full_plan = stability_plan(frozen, sha(args.tokenizer_pilot), sha(args.frozen_run), args.chat)
        manifest.update(mode='checkpoint_stability', stability_plan=full_plan,
                        arms=full_plan['arms'],evaluation_targets=full_plan['targets'],
                        group_discovery=args.groups is None, success_count='not screened',n_success_requested=0)
    if args.prompt_comparison:
        bundle = json.loads(args.groups.read_text())
        if (bundle['frozen_sha256'] != sha(args.frozen_run) or bundle['pairs_sha256'] != sha(args.pairs)
                or bundle['model'] != frozen['model'] or bundle['chat'] != args.chat
                or bundle['model_revision'] != pilot['resolved_revision']
                or bundle['tokenizer_identity_sha256'] != pilot['identity_sha256']
                or bundle['suppression'] != bundle['selected']['es']):
            parser.error('Frozen group provenance mismatch')
        if any(str(seed) not in bundle['random'] for seed in args.control_seeds):
            parser.error('Missing fixed control seed')
        prompt_rows = [json.loads(x) for x in (args.tokenizer_pilot.parent/'prompt_tokens.jsonl').read_text().splitlines()]
        lookup = {(r['key'],r['language']):r for r in prompt_rows if r['source']=='chains'}
        for c in frozen['eval_chains']:
            for lg in ('es','zh','ko'):
                if lookup[c,lg]['text'] != chains[c]['prompts'][lg] or not chains[c]['answer_label_by_lang'][lg]:
                    parser.error('Missing or changed multilingual prompt/answer')
            if not all(scoring.scoreable(c,t) for t in ('zh','ko')):
                parser.error('Unexpected ineligible evaluation item')
        manifest.update(mode='prompt_comparison', group_discovery=False,
                        groups_sha256=sha(args.groups), historical_group_identity='not certified')
        manifest['input_sha256'][str(CODE/'src/beacon/data/prompt_comparison.py')] = sha(CODE/'src/beacon/data/prompt_comparison.py')
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    if args.out.exists():
        parser.error("Output already exists; use a new directory (partial journals are preserved)")
    if pilot is None:
        parser.error("Run tokenizer pilot first and supply --tokenizer-pilot REPORT.json")
    # Heavy dependencies are confined to the actual GPU run.
    import torch
    from beacon.config import load_configs, load_env
    from beacon.data.prompting import encode
    from beacon.models import load_model
    from transformers import AutoTokenizer
    neuron_module = module_at("frozen_neuron_hooks", CODE / "scripts/language_neurons.py")
    args.out.mkdir(parents=True)
    manifest["status"] = "running"
    save_json(args.out / "manifest.json", manifest)
    try:
        load_env()
        configs = load_configs()
        source_config = configs.models["models"][frozen["model"]]
        if source_config.get("local_path") or source_config["hf_id"] != pilot["repo_id"]:
            raise ValueError("Model source must be the same pinned HF repo as the tokenizer pilot")
        for package in ("transformers", "tokenizers"):
            if importlib.metadata.version(package) != pilot["versions"][package]:
                raise ValueError(f"{package} version differs from tokenizer pilot")
        if reference_manifest:
            for package, version in reference_manifest['runtime']['versions'].items():
                if importlib.metadata.version(package) != version:
                    raise ValueError(f'{package} version differs from preceding replay')
        pilot_dir = args.tokenizer_pilot.parent
        tokenizer = AutoTokenizer.from_pretrained(pilot_dir / "snapshot", local_files_only=True, trust_remote_code=False)
        if tokenizer_identity(tokenizer)["sha256"] != pilot["identity_sha256"]:
            raise ValueError("Loaded tokenizer identity differs from pilot")
        print(f"[load] {frozen['model']} revision={pilot['resolved_revision']}", flush=True)
        loaded = load_model(frozen["model"], configs, framework="hf_hooks", attn_implementation="eager",
                            revision=pilot["resolved_revision"], tokenizer=tokenizer)
        loaded.model.eval()
        print("[load] model ready; validating generation configuration and prompt tensors", flush=True)
        actual_commit = getattr(loaded.model.config, "_commit_hash", None)
        if actual_commit is not None and actual_commit != pilot["resolved_revision"]:
            raise ValueError("Loaded model commit differs from tokenizer pilot")
        if max(tokenizer.get_vocab().values()) >= loaded.model.get_input_embeddings().weight.shape[0]:
            raise ValueError("Tokenizer IDs exceed actual model input embedding size")
        expected_generation = json.loads((pilot_dir / "effective_generation_config.json").read_text())
        actual_generation = loaded.model.generation_config.to_dict()
        if reference_manifest and actual_generation != reference_manifest['runtime']['generation_config']:
            raise ValueError('Full generation configuration differs from preceding replay')
        for key in ("eos_token_id", "bos_token_id", "repetition_penalty", "no_repeat_ngram_size",
                    "min_length", "min_new_tokens", "forced_bos_token_id", "forced_eos_token_id",
                    "suppress_tokens", "begin_suppress_tokens", "bad_words_ids", "stop_strings"):
            if expected_generation.get(key) != actual_generation.get(key):
                raise ValueError(f"Generation configuration differs from pilot: {key}")
        pilot_prompts = [json.loads(line) for line in (pilot_dir / "prompt_tokens.jsonl").read_text().splitlines()]
        expected_prompt_ids = {}
        for case in pilot_prompts:
            if encode(loaded, case["text"], args.chat)[0].tolist() != case["input_ids"]:
                raise ValueError(f"Tensor prompt encoding differs from pilot: {case['source']}/{case['key']}/{case['language']}")
            if case["source"] == "chains":
                expected_prompt_ids[case["key"], case["language"]] = case["input_ids"]
        manifest["tokenizer_preflight"] = dict(
            resolved_revision=pilot["resolved_revision"], identity_sha256=pilot["identity_sha256"],
            tensor_prompt_checks=len(pilot_prompts), historical_identity=pilot["historical_tokenizer_identity"],
            warnings=pilot["warnings"])
        print(f"[preflight] {len(pilot_prompts)} prompt tensors matched", flush=True)
        manifest["runtime"] = dict(
            versions={name: importlib.metadata.version(name) for name in ("torch", "transformers", "accelerate")},
            model_commit=getattr(loaded.model.config, "_commit_hash", None),
            model_class=type(loaded.model).__name__, tokenizer_class=type(loaded.tokenizer).__name__,
            dtype=str(loaded.model.dtype), generation_config=loaded.model.generation_config.to_dict())
        if args.prompt_comparison:
            manifest['runtime'].update(
                cuda_version=torch.version.cuda,
                gpu_devices=[dict(name=torch.cuda.get_device_name(i),
                                  memory_bytes=torch.cuda.get_device_properties(i).total_memory)
                             for i in range(torch.cuda.device_count())],
                device_map={str(k):str(v) for k,v in getattr(loaded.model,'hf_device_map',{}).items()},
                deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                effective_decode_overrides=dict(max_new_tokens=manifest['max_new_tokens'],
                                                do_sample=False,pad_token_id=loaded.tokenizer.eos_token_id))
        pairs = [json.loads(line) for line in args.pairs.read_text(encoding="utf-8").splitlines() if line.strip()]
        expected_markers = json.loads((pilot_dir / "marker_token_ids.json").read_text())
        if args.groups:
            bundle = json.loads(args.groups.read_text(encoding="utf-8"))
            if (bundle["model"] != frozen["model"] or bundle["frozen_sha256"] != sha(args.frozen_run)
                    or bundle["pairs_sha256"] != sha(args.pairs) or bundle["chat"] != args.chat
                    or bundle["targets"] != targets or bundle.get("tokenizer_identity_sha256") != pilot["identity_sha256"]
                    or bundle.get("model_revision") != pilot["resolved_revision"]
                    or bundle.get("marker_token_ids") != expected_markers):
                raise ValueError("Saved groups do not match this frozen input/chat/corpus")
        else:
            with torch.inference_mode():
                bundle = restore_groups(loaded, neuron_module, frozen, pairs, args.control_seeds, torch, args.chat, expected_markers)
            bundle.update(model=frozen["model"], frozen_sha256=sha(args.frozen_run),
                          pairs_sha256=sha(args.pairs), chat=args.chat, targets=targets,
                          tokenizer_identity_sha256=pilot["identity_sha256"], model_revision=pilot["resolved_revision"],
                          provenance="reconstructed from original algorithm; exact historical IDs unavailable")
        if args.strata_reference or (args.checkpoint_stability and args.groups and not args.prompt_comparison):
            layers = neuron_module._decoder_layers(loaded.model)
            widths = {str(i): layer.mlp.down_proj.weight.shape[1] for i, layer in enumerate(layers)}
            def draw(seed, selected, order, widths):
                generator = torch.Generator().manual_seed(seed)
                return {target: {layer: torch.randperm(widths[str(layer)], generator=generator)[:len(ids)].tolist()
                                 for layer, ids in selected[target].items()} for target in order}
            bundle = extend_random_groups(bundle, widths, draw)
            manifest['reference_selected_and_seed7_preserved'] = True
        bundle = normalize_and_validate_groups(bundle, loaded, neuron_module, targets, args.control_seeds)
        if bundle["suppression"] != bundle["selected"]["es"]:
            raise ValueError("Suppression must use the original selected Spanish group")
        save_json(args.out / "groups.json", bundle)
        if args.strata_reference or args.checkpoint_stability:
            save_json(args.out/'group_overlap.json', {
                str(seed): {target: {str(layer): len(set(ids) & set(bundle['random'][str(seed)][target][layer]))
                                    for layer, ids in bundle['selected'][target].items()} for target in targets}
                for seed in args.control_seeds})
        manifest["groups_sha256"] = sha(args.out / "groups.json")
        save_json(args.out / "manifest.json", manifest)
        layers = neuron_module._decoder_layers(loaded.model)

        @torch.inference_mode()
        def decode(chain, target, arm):
            groups = []
            if arm["kind"] in ("selected", "random"):
                by_layer = (bundle["selected"][target] if arm["kind"] == "selected"
                            else bundle["random"][str(arm["seed"])][target])
                groups.append((by_layer, 1. + arm["amp"]))
            if arm["suppression"] != 1:
                groups.append((bundle["suppression"], arm["suppression"]))
            prompt_language = arm.get("_prompt_language", "es")
            prompt = chains[chain]["prompts"][prompt_language]
            ids = encode(loaded, prompt, args.chat)
            if ids[0].tolist() != expected_prompt_ids[chain, prompt_language]:
                raise ValueError("Prompt token IDs changed after pilot validation")
            handles = []
            try:
                for by_layer, factor in groups:
                    for layer, neurons in by_layer.items():
                        handles.append(layers[layer].mlp.down_proj.register_forward_pre_hook(
                            neuron_module._mk_prehook_all(neurons, factor)))
                output = loaded.model.generate(input_ids=ids, max_new_tokens=manifest["max_new_tokens"],
                                               do_sample=False, pad_token_id=loaded.tokenizer.eos_token_id)
            finally:
                for handle in handles:
                    handle.remove()
            tokens = output[0, ids.shape[1]:].tolist()
            eos = loaded.model.generation_config.eos_token_id
            eos = eos if isinstance(eos, (list, tuple)) else [eos]
            stop = ("eos" if tokens and tokens[-1] in eos else
                    "max_new_tokens" if len(tokens) >= manifest["max_new_tokens"] else "other")
            return dict(text=loaded.tokenizer.decode(tokens, skip_special_tokens=True), token_ids=tokens,
                        max_new_tokens=manifest['max_new_tokens'],
                        text_with_specials=loaded.tokenizer.decode(tokens, skip_special_tokens=False),
                        prompt_token_ids=ids[0].tolist(),
                        n_generated_tokens=len(tokens), stop_reason=stop,
                        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())

        def score(text, chain, language):
            eligible = bool(scoring.scoreable(chain, language))
            return dict(content=bool(scoring.content(text, chain)), eligible=eligible,
                        form=bool(scoring.ilr(text, chain, language)) if eligible else None,
                        degenerate=bool(degeneracy.is_degenerate(degeneracy.first_line(text))))

        if args.checkpoint_stability:
            result = run_stability(chains=chains,frozen=frozen,plan=full_plan,decode=decode,score=score,
                                   output=args.out,progress=lambda message: print(message,flush=True))
        elif args.strata_reference:
            context = dict(model=frozen['model'], revision=pilot['resolved_revision'],
                           tokenizer_pilot_sha256=sha(args.tokenizer_pilot), chat=args.chat,
                           chains_sha256=sha(ROOT/'02-data/inputs/chain_lang_prompts.json'),
                           frozen_sha256=sha(args.frozen_run), max_new_tokens=manifest['max_new_tokens'],
                           runtime=manifest['runtime'], scorer_sha256=sha(ROOT/'01-code/analysis'/'verify_numbers.py'),
                           degeneration_sha256=sha(ROOT/'01-code/analysis'/'degenerate_outputs.py'),
                           prompting_sha256=sha(CODE/'src/beacon/data/prompting.py'))
            result = collect_strata(chains=chains, frozen=frozen, arms=arms, decode=decode, score=score,
                                    output=args.out, context=context, screen_input=args.screening_input,
                                    screen_only=args.screen_only, progress=lambda message: print(message, flush=True))
        elif args.length_check_reference:
            results = []
            for budget in (32,64):
                manifest['max_new_tokens'] = budget
                manifest['current_budget'] = budget
                save_json(args.out/'manifest.json', manifest)
                directory = args.out/f'length_{budget}'
                directory.mkdir()
                budget_plan = dict(replay_plan, max_new_tokens=budget,
                                   scope='fixed-group replication and length diagnostic; original sample and arm settings unchanged')
                print(f'[length-check] starting max_new_tokens={budget}; fixed reference neuron IDs',flush=True)
                results.append(run_replay(chains=chains,frozen=frozen,plan=budget_plan,decode=decode,
                                         score=score,output=directory,progress=lambda message: print(message,flush=True)))
            comparison = compare_length_runs(args.length_check_reference,args.out,replay_plan)
            result = dict(n_records=sum(r['n_records'] for r in results), replay=comparison,
                          cohorts=dict(selected_successes=0,success_shortfall=0))
            manifest['max_new_tokens'] = None  # Multiple budgets; see generation_budgets.
        elif replay_plan:
            result = run_replay(chains=chains, frozen=frozen, plan=replay_plan, decode=decode,
                                score=score, output=args.out, progress=lambda message: print(message, flush=True))
        else:
            result = run_audit(chains=chains, frozen=frozen, arms=arms, decode=decode, score=score,
                               output=args.out, n_success=args.n_success, sample_seed=args.cohort_seed,
                               progress=lambda message: print(message, flush=True))
        manifest.update(status="complete", n_records=result["n_records"],
                        selected_successes=result["cohorts"]["selected_successes"],
                        success_shortfall=result["cohorts"]["success_shortfall"], replay=result["replay"])
    except BaseException as exc:
        manifest.update(status="incomplete", error_type=type(exc).__name__)
        save_json(args.out / "manifest.json", manifest)
        raise
    save_json(args.out / "manifest.json", manifest)
    print(f"Saved {'small replay' if replay_plan else 'both cohorts'} to {args.out}; no pooled effect computed.")


if __name__ == "__main__":
    main()
