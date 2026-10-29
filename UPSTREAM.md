# Source provenance

Extracted from [Ist1na07/spm_rating_map_analyser](https://github.com/Ist1na07/spm_rating_map_analyser), **tag v1.2.0**, peeled commit `1e99e4dc20f0c338de414e76afeaf73d8219d65c`.

`src/legacy/tag-engine.cjs` preserves the plugin's complete `tag_engine.js` bundle and appends a CommonJS export. The original browser global is `TagClassifier`. No classification logic or learned coefficients were changed for v1.1.0. Original MIT license retained. The source-map-style section comments identify the original modules; original TypeScript source was not present in that plugin repository.

The difficulty engine, overlay, tosu networking, and dan conversion are deliberately outside this standalone library. Existing `analyzeMap`, `buildCachedMap`, and `extractIntermediate` entry points and speedRate option are retained.
