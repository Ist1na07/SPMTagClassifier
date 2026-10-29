# v1.2.0 research and release plan

## Acceptance

1. Publish v1.1.0 as the unmodified 7K extraction, with parity evidence.
2. Audit parsing, tail/head semantics, LN occupancy, temporal context, rhythm phase and aggregation normalization.
3. Implement interpretable, key-normalized features and per-segment distributions/evidence; compare confidence/effort aggregation and additional pattern descriptors.
4. Tune and evaluate 7K map tags plus six manually segmented maps, 4K coarse pattern labels, and 6K transfer. Difficulty labels are not pattern labels. A course is a container, not a locally observable key pattern.
5. Retune each ablation rather than removing features from a once-tuned model. Use fixed grouped train/validation/test splits, including rate variants, duplicates and same-song variants in the same split. Never tune on test results. Record every search candidate, chosen parameters, per-chart predictions, class metrics, calibration and failures.
6. No 6K pattern ground truth is present: use dimensional/mathematical transfer, synthetic invariants and a full-corpus finite-output/coverage audit; state the resulting evidence level.
7. Commit reproducible scripts, model artifacts, inventories, research report and tests. Push and release v1.2.0 only after required results are present.

## Local inputs

- 7K SPM 1.1.1: 393 records / 391 beatmaps, impression-level multi-label tags, human noise.
- ApoLar: 292 7K jack maps; coarse positive-only external diagnostic, not a negative-class validation set.
- 4K LeoBlack 0.2.0: 1,349 maps, jack/ln/speed/stamina/tech plus course.
- 6K 0.1.0: 2,344 maps, difficulty but no pattern labels.
- TagClassifierDataset: six maps, natural-language interval annotations, alternatives and unlabelled regions.

Raw charts, local caches, upstream checkout and prompt remain ignored. Research artifacts must not silently substitute weak map tags for precise segment labels.
