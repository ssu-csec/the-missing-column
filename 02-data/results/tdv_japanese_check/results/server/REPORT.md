# Japanese 70B TDV single-hop implementation check

Current pinned-runtime reconstruction; not exact replay of the historical vector.

| Subset | N | TDV recovery % | R0 recovery % | TDV−R0 pp [95% CI] |
|---|---:|---:|---:|---|
| all_screen | 200 | 52.5 | 49.5 | +3.0 [-7.0, +12.8] |
| implementation_failures | 102 | 15.7 | 5.9 | +9.8 [-5.6, +26.5] |
| implementation_failures_cluster_disjoint | 50 | 14.0 | 12.0 | +2.0 [-20.7, +25.4] |
| untreated_ja_success | 98 | 90.8 | 94.9 | -4.1 [-15.8, +3.9] |

No minimum absolute recovery threshold is asserted post hoc. Report rates and uncertainty together.
Cluster-disjoint sensitivity is mandatory because historical construction was item-disjoint only.
New single-hop evidence does not rescore or replace the historical two-hop +19.6 pp.
