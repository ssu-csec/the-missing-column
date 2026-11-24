# Japanese single-hop TDV implementation check (Llama-3.1-70B-Instruct)

Appendix E reports this check. It was added after submission because the original TDV implementation
check covered Korean only. It is a single-hop implementation check, not a rerun of the Japanese two-hop
transfer in `../tdv/`.

**Setup.** Llama-3.1-70B-Instruct, translation-direction vector at layer 60 (zero-based), scale 1.0,
added at the last position on every forward pass. Greedy decoding with 24 new tokens, repetition
penalty 1.3, no-repeat 3-gram blocking, and a two-token minimum. All 200 screened single-hop chains
are generated in eight arms: untreated Japanese (`U_ja`), untreated English (`U_en`), TDV, and five
norm-matched random directions (`R0`, `R17`, `R27`, `R37`, `R47`; `R0` is the primary control).
The 60 construction chains were rebuilt by excluding the preserved two-hop evaluation IDs, because the
original construction manifest and vector were not kept; the rebuilt vector's norm (12.54) is consistent
with the logged 12.5 but this does not certify that the two vectors are identical.

**Primary set.** Items that fail untreated in Japanese (the run's in-language verdict) and are
answered untreated in English: 102 items in 39 answer clusters.

| Set | Items / clusters | TDV | R0 | TDV − R0 (pp), 95% CI |
|---|---:|---:|---:|---:|
| Primary set | 102 / 39 | 16 (15.7%) | 6 (5.9%) | +9.8 [−5.6, +26.5] |
| Primary set, construction-cluster-disjoint | 50 / 21 | 7 (14.0%) | 6 (12.0%) | +2.0 [−20.7, +25.4] |
| All screened items | 200 / 91 | 105 (52.5%) | 99 (49.5%) | +3.0 [−7.0, +12.8] |
| Untreated Japanese successes (retention) | 98 / 53 | 89 (90.8%) | 93 (94.9%) | −4.1 [−15.8, +3.9] |

Against the mean of the five random directions, the primary-set contrast is +10.2 [−2.2, +25.0].
Intervals are paired answer-cluster bootstrap intervals (10,000 draws, fixed seed), conditional on the
fixed vector and random directions and not adjusted for multiplicity. No interval excludes zero, so the
check does not pass the operational rule and is reported as inconclusive.

## Files

| Path | Contents |
|---|---|
| `data/raw/generations.jsonl` | 1,600 outputs (200 chains × 8 arms): text, token IDs, stop reason, stored and locked scores |
| `data/raw/screening.jsonl` | untreated Japanese and English verdicts and set membership for the 200 chains |
| `data/raw/plan.json` | frozen construction and screening IDs, seeds, decoding settings, and the SHA-256 of the inputs and code the run used |
| `data/raw/vectors.pt`, `vector_metadata.json` | the TDV, the five random directions, and their metadata |
| `data/raw/prompt_tokens.jsonl`, `tokenizer_preflight.json` | tokenized inputs and the tokenizer preflight result |
| `results/server/` | the run's own `analysis.json` and report |
| `results/recomputed/` | the CPU recomputation: `analysis.json`, `rates.csv`, `contrasts.csv`, `generations_scored.csv`, `validation.json` |
| `MANIFEST.json` | SHA-256 of every file in this folder except this README and the manifest itself |

## Recompute without a GPU

From the repository root:

```bash
python3 -I 01-code/analysis/tdv_japanese_check_recompute.py --out /path/to/new-directory
```

The script checks every hash in `MANIFEST.json` and the run inputs recorded in `data/raw/plan.json`
(`02-data/inputs/chain_lang_prompts.json`, `02-data/inputs/single_hop_prompts.jsonl`, and the two-hop
TDV result in `../tdv/`), rescores all 1,600 outputs with the run-time scorer and with the locked
matcher, recomputes the bootstrap intervals, and compares them with `results/server/analysis.json`.
It needs only the Python standard library; the scorer functions run from their unchanged source in
`01-code/`, without importing torch.

The stored in-language verdict and the locked matcher are different rules and are reported separately;
locked scores are blank where an item is not attributable.

## Regenerate on a GPU host

```bash
python 01-code/scripts/tdv_japanese_check.py --tokenizer-pilot <report.json> \
    --plan 02-data/results/tdv_japanese_check/data/raw/plan.json --out <new directory>
```

The run used two 80 GB GPUs and gated model access. The tokenizer snapshot used for the run and the
job's runtime records are not redistributed, so the script takes a `tokenizer_preflight.py` report for
Llama-3.1-70B-Instruct (chat), and the model weights must be downloaded separately. The plan records
the SHA-256 of the code the run used; the scorer functions in the released code are identical to it.
