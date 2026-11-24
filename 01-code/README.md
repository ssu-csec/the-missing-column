# 01-code

The `beacon` package (`src/beacon/`: model loading, prompting, interventions, scoring), its configuration (`configs/`), the GPU experiment scripts (`scripts/`), and the CPU analyses of stored outputs (`analysis/`). Section, table, and figure numbers below refer to the paper.

## Setup

```bash
cd 01-code && pip install -e .     # Python 3.12; installs beacon and its dependencies
export PYTHONPATH=01-code/src
```

GPU experiments need a Hugging Face token (`HF_TOKEN`) for the gated checkpoints and an H100-class GPU; the 70B checkpoints shard across two 80 GB GPUs.
The scripts that take data locations as flags resolve relative paths against the repository root, so they can be run from any directory; the control reruns (energy- and output-matched controls, SCH strength control) locate `02-data/` themselves. Defaults point to the released files in `02-data/` (the chain set `02-data/inputs/chain_lang_prompts.json`, the solvability matrices `02-data/results/solvability/solvability_<model>.json`) and write to the matching folder of `02-data/results/`. Inputs that are not distributed are expected under `02-data/inputs/`: the full evaluation set (`dataset.jsonl`), the parallel prompt pairs (`generic_pairs4.jsonl`), the SCH campaign prompts, failure subsets, and expanded aliases (`dataset_campaign.jsonl`, `sch_subsets/`, `aliases_rich.json`).

## Experiment scripts (`scripts/`)

| Script | Paper | Produces |
|---|---|---|
| `evaluate_base_models.py` | §4 | base-model evaluation and the screened failure population |
| `solvability_matrix.py`, `solvability_rescore.py` | §4, App. A.7 | English-prompt solvability of the screened failures |
| `vt_override_layer.py`, `vt_surfacing_vectors.py` | §3.2, App. A.4, A.5 | surfacing vectors ($v_T$) and their random-direction control |
| `language_neurons.py` | §3.2, App. A.4, A.5 | language-neuron amplification and its random-group control |
| `language_neuron_groups.py` | App. A.4 | frozen language-neuron groups used by every neuron run (`frozen_inputs/`) |
| `mrre_anchor_vectors.py` | §3.2, App. A.4, A.5 | MRRE anchor vectors and their randomized-anchor control |
| `energy_matched_control_gemma.py`, `energy_matched_control.py` | §5.3, App. C.2 | energy-matched random groups |
| `output_matched_control.py` | §5.3, App. C.2 | output-matched (next-token KL) random groups |
| `sch_discover_pools.py`, `sch_validate_specific_pools.py`, `sch_amplification.py` | App. D | SCH head pools, the mean-ablation gate, and SCH amplification |
| `sch_strength_control.py` | App. D.3 | SCH control-strength sensitivity |
| `translation_vector.py` | App. E (Table 25) | translation-direction vector (TDV) and its norm-matched random control |
| `tdv_japanese_check.py` | App. E | Japanese single-hop TDV implementation check on Llama-3.1-70B-Instruct |
| `knowledge_neurons.py` | App. F.1 (Table 26) | knowledge-neuron activation on the screened failures |
| `multilingual_lens.py` | App. F.1 (Table 27) | multilingual logit lens on a fixed prompt panel (`02-data/inputs/es_flip_targets.jsonl`) |
| `script_decode.py` | App. F.2 (Table 28) | script-constrained decode |
| `english_axis_ablation.py` | App. F.2 (Table 29) | English-axis ablation |
| `form_profile.py` | App. F.2 (Table 30) | designated-form readout advantage |
| `token_fertility.py` | App. F.2 (Table 31) | answer-label token fertility (CPU, tokenizer only) |
| `rescreen.py` | App. F.3 (Table 32) | 32-token re-screening of the original pools |
| `tokenizer_preflight.py` | all GPU runs | CPU tokenizer/config preflight pinned before each GPU run |

Each GPU run writes `plan.json` (fixed before generation), `manifest.json` (runtime and versions), `calibration.json`/`validation.json` where applicable, `generations.jsonl`, and `analysis.json` into the matching folder of `02-data/results/`; the released `02-data/` keeps only the files the analyses read (see `02-data/README.md`).
`sch_amplification.py` needs an SCH head-pool file (`--heads`), an external input not distributed here.
The App. E and F scripts default to the released inputs: `02-data/inputs/`, the solvability matrices in `02-data/results/solvability/` (`--solvmatrix`), and, for `translation_vector.py`, the second-hop prompts in `02-data/results/tdv_japanese_check/data/inputs/` (`--dataset`). `rescreen.py` needs the original pool chain-ID lists (`--pool`), which are not distributed. Their outputs go to `02-data/results/` (`tdv/`, `rivals/`, `access/`, `rescreen/`). The Japanese TDV check regenerates from its frozen plan (`02-data/results/tdv_japanese_check/data/raw/plan.json`).

## Analyses of stored outputs (`analysis/`)

These need no model or GPU. Each reads or writes the listed folder in `02-data/results/`; `verify_numbers.py` recomputes the paper's numbers from them (`python 01-code/analysis/verify_numbers.py`).

| Paper | Script | Result folder |
|---|---|---|
| Table 1, Figure 2, §5.1--5.4; App. A.4 (Table 4), App. A.8 (Tables 7, 8, including the permissive and cluster-disjoint columns), Figure 4 | `paired_effects.py` | preserved generations in `interventions/` |
| App. A.9 (Table 9) | `verify_numbers.py` (`family_cell`) | preserved generations in `interventions/` |
| §5.2, Figure 3, App. A.9 (Table 10) | `state_transitions.py` | `state_transitions/` |
| §5.2 | `realized_form_composition.py` | `realized_form_composition/` |
| §4, §5.3, App. A.7 (Table 6), App. C.1 (Tables 12, 13) | `hop_solvability.py`, `degenerate_outputs.py`, `attempt_decomposition.py` | `solvability/`, `interventions/` |
| App. A.7 (solvability on the retained items) | `english_solvable_sensitivity.py` | `english_solvable/` |
| Table 8 (drop-one), App. C.2 (Table 16) | `checkpoint_exclusion_sensitivity.py` | `checkpoint_exclusion/` |
| §5.3, Table 8 (joint pass) | `leave_one_out_selection.py` | `checkpoint_exclusion/leave_one_out_selection.json` |
| §5.3 (without Gemma-4-31B, both arms attempting) | `neuron_robustness.py` | `checkpoint_exclusion/neuron_robustness.json` |
| §5.3, App. C.1 | `both_arms_attempt.py` | `both_arms_attempt/` |
| §5.3, App. C.2 (Tables 14, 15) | `energy_matched_summary.py`, `energy_matched_seed_effects.py`, `output_matched_summary.py` | `energy_matched_control/`, `output_matched_control/` |
| §5.2, App. B.2 | `human_validation_summary.py`, `human_validation_sensitivity.py` | `human_form_audit/grid_validation/` |
| App. D (Table 23) | `sch_form_composition.py` | printed table rows, from `interventions/` |
| App. E (Japanese TDV check) | `tdv_japanese_check_recompute.py` | `tdv_japanese_check/results/recomputed/` |

The analyses locate `02-data/results/` through `result_paths.py` (override with `RESULTS_DIR`).
