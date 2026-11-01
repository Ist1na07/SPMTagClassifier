# Source provenance

Extracted from [Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser), **tag v1.2.0**, peeled commit `1e99e4dc20f0c338de414e76afeaf73d8219d65c`.

`src/legacy/tag-engine.cjs` preserves the plugin's complete `tag_engine.js` bundle and appends a CommonJS export. The original browser global is `TagClassifier`. No classification logic or learned coefficients were changed for v1.1.0. Original MIT license retained. The source-map-style section comments identify the original modules; original TypeScript source was not present in that plugin repository.

The difficulty engine, overlay, tosu networking, and dan conversion are deliberately outside this standalone library. Existing `analyzeMap`, `buildCachedMap`, and `extractIntermediate` entry points and speedRate option are retained.

## Optional research input: SPMRating v1.2.0

`research/frontier` additionally consumes [Ist1na07/SPMRating](https://github.com/Ist1na07/SPMRating/tree/v1.2.0), tag **v1.2.0**, commit `a914a405d8e9a77f94c0518567c37511b0b7ac05`, MIT, copyright2026 Ist1na. It reuses existing public curve results and observes private components in a pinned, read-only research checkout; no upstream files are modified or bundled into the Node runtime. See [interface, clock and evidence limits](research/frontier/SPM_REUSE.md). The pretrained difficulty model is an auxiliary prior, not a classifier retrained independently within each tag-validation fold.
