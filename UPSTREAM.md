# Source provenance

Extracted from [Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser), **tag v1.2.0**, peeled commit `1e99e4dc20f0c338de414e76afeaf73d8219d65c`.

`src/legacy/tag-engine.cjs` preserves the plugin's complete `tag_engine.js` bundle and appends a CommonJS export. The original browser global is `TagClassifier`. No classification logic or learned coefficients were changed for v1.1.0. Original MIT license retained. The source-map-style section comments identify the original modules; original TypeScript source was not present in that plugin repository.

The difficulty engine, tosu networking, and dan conversion remain outside this standalone library. Existing `analyzeMap`, `buildCachedMap`, and `extractIntermediate` entry points and speedRate option are retained. The optional v1.2-series updater patches the already-installed v1.2.0 overlay caller and display; it does not bundle or replace the scoring implementation.

## Optional research input: SPMRating v1.2.0

`research/frontier` and `research/perception` consume [Ist1na07/SPMRating](https://github.com/Ist1na07/SPMRating/tree/v1.2.0), tag **v1.2.0**, commit `a914a405d8e9a77f94c0518567c37511b0b7ac05`, MIT, copyright2026 Ist1na. Research reuses existing public curves and observes private components in a pinned, read-only checkout. Production accepts the already-computed Detailed or Compact raw curve; it never calls or bundles a scorer. The updater includes the released numeric parameter dictionaries only to reject mismatched installed models before adding the raw-curve handoff. See [runtime contract](docs/perception.md) and [research interface](research/frontier/SPM_REUSE.md). The pretrained difficulty model is an auxiliary prior, not a classifier retrained independently within each tag-validation fold.

## Iteration identity

The original v1.1.0 is the research baseline. All subsequent research and implementation in this assignment belongs to the v1.2 series. The published v1.3.0 was a numbering error, not a new research iteration; its immutable snapshot is retained for traceability and superseded by v1.2.3. References to v1.2.2 in frozen aggregation protocols name an intermediate expert/control, not the original research baseline.
