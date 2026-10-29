# Reference algorithms and decisions

Inspected 2026-10-30. Only design ideas were used; no third-party implementation was copied into the new feature engine.

| Primary source | Pinned revision | What informed this implementation |
|---|---|---|
| [SPM Map Analyser](https://github.com/Ist1na07/spm_rating_map_analyser/tree/v1.2.0) | `1e99e4dc20f0c338de414e76afeaf73d8219d65c` | Baseline bundle, 7K taxonomy, API, segmentation, model and aggregation. Preserved verbatim in the legacy module. |
| [LeoBlack pattern primitives](https://github.com/LeoBlackMT/osumania_map_analyser/blob/bc5e8c6f7801f58efaec22cb492a7aa9dd7e5945/ManiaMapAnalyser%20by%20Leo_Black/js/patterns/primitives.js) and [categorisation](https://github.com/LeoBlackMT/osumania_map_analyser/blob/bc5e8c6f7801f58efaec22cb492a7aa9dd7e5945/ManiaMapAnalyser%20by%20Leo_Black/js/patterns/categorise.js) | `bc5e8c6f7801f58efaec22cb492a7aa9dd7e5945` | Separating heads, bodies and tails is useful. Coarse class names differ from 7K; do not copy class thresholds across key counts. |
| [Interlude primitives](https://github.com/YAVSRG/YAVSRG/blob/394be9281ce10c8ec992dd654157a80451ae6a50/prelude/src/Calculator/Patterns/Primitives.fs) and [categorisation](https://github.com/YAVSRG/YAVSRG/blob/394be9281ce10c8ec992dd654157a80451ae6a50/prelude/src/Calculator/Patterns/Categorise.fs) | `394be9281ce10c8ec992dd654157a80451ae6a50` | Consecutive head adjacency, movement direction, density and variety are distinct signals. Pattern composition should remain visible instead of collapsing all information to one label. Its hybrid definition differs from the supplied 7K notes. |
| [LeoBlack benchmark](https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark/tree/ac5366b1be84207406d9cea0d76427b9b7c5ffa0) | `ac5366b1be84207406d9cea0d76427b9b7c5ffa0` | Local converted 4K dataset uses top-level jack/ln/speed/stamina/tech/course plus finer sub-patterns. Course is a chart container; it is excluded from five-way local-pattern training and remains in corpus smoke tests. |

The historical `LeoBlackMT/osumania_estimator` URL returned 404. The maintained analyser's pattern modules above were inspected instead. Local dataset manifests and processing notes remain authoritative for the exact samples actually evaluated; the current benchmark revision is context, not a replacement for those samples.

The supplied `docs/7K键型分类.txt` and `docs/formerprompt.txt` define the intended player-facing distinctions. In particular, jack overlap is not synonymous with large chords; release is tail independence rather than mere LN duration variance; inverse requires both occupancy and short re-press gaps; vibro needs repetitive low-control structure, not just high speed.
