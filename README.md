# SPMTagClassifier

Standalone osu!mania pattern classifier. v1.1.0 extracts the **7K** engine from SPM Map Analyser **v1.2.0**, without changing its predictions.

```js
const { analyzeMap } = require('spm-tag-classifier');
const fs = require('node:fs');
const result = analyzeMap(fs.readFileSync('map.osu', 'utf8'), { speedRate: 1 });
console.log(result.overall.tags, result.segments);
```

```sh
node bin/classify.cjs map.osu 1.5
npm test
npm run test:parity
```

Parity requires the ignored `.upstream/` checkout of the plugin v1.2.0 and local `dataset/spm_dataset-1.1.1`. The library itself has no runtime dependencies and does not need tosu. The original bundle also works as a browser script (`TagClassifier`).

## Research upgrade

v1.2.0 will audit feature semantics, segment evidence and aggregation, tune 4K on the LeoBlack benchmark, and transfer normalized 4K/7K geometry to 6K. Every candidate family will be retuned under the same grouped evaluation protocol. Raw beatmaps are not redistributed. See [research/PLAN.md](research/PLAN.md), [UPSTREAM.md](UPSTREAM.md), and the original research notes in `docs/`.

The baseline's `scores` are heuristic/model scores, **not calibrated probabilities**. The extraction preserves that contract and its limitations so future comparisons remain reproducible.
