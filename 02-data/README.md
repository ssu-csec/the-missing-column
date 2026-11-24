# 02-data

The evaluation set and the result files behind *The Missing Column*: the files that the analyses in `01-code/analysis/` read to recompute the numbers in the paper, the returned human judgments of Appendix B, and the recorded values of the auxiliary diagnostics. Section, table, and figure numbers refer to the paper.

## inputs/

The evaluation set, built from Wikidata (CC0) two-hop chains whose answers are currencies, capitals, heads of state, and central banks:

| File | Contents |
|---|---|
| `chain_lang_prompts.json` | 466 chains keyed by `P<r1>_P<r2>_Q<entity>`: two-hop `prompts`, `answer_label_by_lang`, and `answer_aliases_by_lang` for en/es/ko/zh/ja |
| `es_flip_targets.jsonl` | the fixed prompt panel of the multilingual logit lens (App. F.1) |
| `single_hop_prompts.jsonl` | second-hop (single-hop) prompts for the 466 chains in en/es/ko/zh/ja, used by the TDV runs (App. E) |

## results/

| Path | Contents |
|---|---|
| `interventions/` | per-item intervention and control generations of the main audit: surfacing vectors (`surfacing_vector_*`), language neurons (`language_neurons_*`), MRRE (`mrre_*`); SCH pools, gates and generations (`sch_pool_*`, `sch_pool_*_specific_gate`, `sch_amplification_*`; App. D) |
| `solvability/solvability_*.json` | English-prompt solvability on all frozen chains (§4, Table 6) |
| `english_solvable/` | English-prompt solvability on the retained items of the main audit (App. A.7) |
| `attribution_strict_summary.json` | attribution-strict Spanish summary, recomputed by the checker |
| `state_transitions/` | item-level state transitions (§5.2, Figure 3, Table 10) |
| `realized_form_composition/` | realized forms of intervention-only content gains (§5.2) |
| `checkpoint_exclusion/` | estimates without one checkpoint (Table 8, App. C.2 Table 16), the selection-aware status of every cell under each leave-one-checkpoint-out roster (`leave_one_out_selection.json`; §5.3, Table 8), and the language-neuron robustness checks (`neuron_robustness.json`; §5.3) |
| `both_arms_attempt/` | the restriction to pairs in which both arms attempt an answer (§5.3, App. C.1) |
| `energy_matched_control/` | language-neuron reruns against energy-matched random groups, per checkpoint, with `summary/` (§5.3, App. C.2) |
| `output_matched_control/` | language-neuron reruns against output-matched random groups (App. C.2) |
| `human_form_audit/annotations/` | returned judgments and their summary for the gain-subset audit (App. B.1, Table 11) |
| `human_form_audit/grid_validation/` | stage-1 grid-wide validation of the locked matcher: the returned judgments, their item key, the summary, and the Spanish-cell sensitivity (§5.2, App. B.2) |
| `sch_strength_control/` | SCH runs against strength-calibrated random pools (App. D.3) |
| `tdv/` | TDV runs on Llama-3.1-70B-Instruct and Llama-3.2-3B (App. E, Table 25) |
| `tdv_japanese_check/` | Japanese single-hop TDV implementation check on Llama-3.1-70B-Instruct: the frozen plan, all generations, vectors, the run's own analysis, and the standard-library recomputation (App. E; see its README) |
| `rivals/` | per-item outputs of the script-constrained decode and the English-axis ablation on Llama-3.1-70B-Instruct (App. F.2, Tables 28, 29) |
| `access/` | per-run outputs of the knowledge-neuron (`suninstrument_*`), logit-lens (`mlens_*`), readout-advantage (`formprofile_*`), and token-fertility (`token_fertility_*`) diagnostics (App. F.1--F.2, Tables 26, 27, 30, 31) |
| `rescreen/` | item-level results of the 32-token pool re-screening (App. F.3, Table 32) |
| `restored_diagnostics/summary.json` | the values the paper reports from the auxiliary diagnostics (App. E Table 25, App. F Tables 26--32, Figure 6), drawn from the per-run outputs in `access/`, `rescreen/`, `rivals/`, and `tdv/`; these values are not recomputed by the checker |

Per-checkpoint folders keep each run's `analysis.json`; generations, run plans, and runtime manifests of the auxiliary GPU runs are not included. The preserved per-item generations of the main audit are in `interventions/`, released as generated.
`01-code/README.md` lists the script that writes each folder. The evaluation set contains no personal or sensitive data.
