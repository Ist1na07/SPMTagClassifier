'use strict';
// Read-only product diagnostics. No model fitting, threshold selection, or export.
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const ROOT = path.resolve(__dirname, '../..');
const output = path.resolve(__dirname, process.argv[2] ?? 'baseline.json');
if (fs.existsSync(output)) throw new Error(`Preserve existing evidence: ${output}. Supply a new output filename.`);
const api = require(path.join(ROOT, 'src/index.cjs'));
function browserEngine(file) {
  const context = {};
  vm.runInNewContext(fs.readFileSync(file, 'utf8'), context, { filename: file });
  assert.equal(typeof context.TagClassifier?.analyzeMap, 'function');
  return context.TagClassifier;
}
const engines = {
  libraryDefault: api,
  browserBundle: browserEngine(path.join(ROOT, 'dist/tag-engine.js')),
  upstreamOverlaySnapshot: browserEngine(path.join(ROOT, '.upstream/tag_engine.js')),
};
function chart(spacing, phase, masks = [[0, 2, 4]]) {
  const objects = Array.from({ length: 200 }, (_, i) => masks[i % masks.length].map(c =>
    `${(c + .5) * 512 / 7},192,${Math.round(1000 + phase + i * spacing)},1,0,0:0:0:0:`).join('\n'));
  return '[General]\nMode:3\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n' +
    '[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n' + objects.join('\n');
}
function read(engine, text, speedRate, target) {
  const result = engine.analyzeMap(text, { speedRate });
  const segments = Array.from(result.segments).filter(s => !target || (s.start < target[1] && s.end > target[0])).map(s => ({
    start: s.start, end: s.end, tag: s.tag, secondary: s.secondary ?? null,
    equivBpm: s.equivBpm ?? null, confidence: s.confidence,
  }));
  return { segments, wholeTags: Array.from(result.overall.tags),
    positiveWindowBpms: [...new Set(result.debug.windows.map(w => w.equivBpm).filter(x => x > 0))] };
}
function displayedDecision(r) { return r.segments.map(s => [s.tag, s.secondary, s.equivBpm]); }
const synthetic = [];
for (const spacing of [50, 62.5, 75, 100, 125, 150, 200, 250]) {
  for (const phase of [0, 13, 27]) for (const rate of [.75, 1, 1.5]) {
    const text = chart(spacing, phase), expectedBpm = Math.round(15000 * rate / spacing);
    const outputs = Object.fromEntries(Object.entries(engines).map(([name, engine]) => [name, read(engine, text, rate)]));
    assert.deepEqual(displayedDecision(outputs.libraryDefault), displayedDecision(outputs.browserBundle));
    synthetic.push({ id: `uniform-${spacing}-phase-${phase}-rate-${rate}`, spacing, phase, rate,
      expectedBpm, truthScope: 'Uniform adjacent-row cadence only; no synthetic primary class label.', outputs });
  }
}
const local = [];
for (const [pack, names, target] of [
  ['r2', ['C16', 'C24', 'C32', 'C48'], [1000, 9000]],
  ['r7', ['H01', 'H02', 'H03', 'H04'], [9000, 17000]],
  ['r5', ['F01', 'F02'], [9000, 16500]],
]) {
  const annotations = JSON.parse(fs.readFileSync(path.join(ROOT, `research/frontier/annotations-${pack}.json`), 'utf8'));
  for (const name of names) {
    const file = `artifacts/frontier-annotation-${pack}/${name}.osu`;
    const text = fs.readFileSync(path.join(ROOT, file), 'utf8');
    const outputs = Object.fromEntries(Object.entries(engines).map(([key, engine]) => [key, read(engine, text, 1, target)]));
    assert.deepEqual(displayedDecision(outputs.libraryDefault), displayedDecision(outputs.browserBundle));
    local.push({ id: name, input: file, targetMs: target, annotation: annotations.cases[name],
      annotationScope: pack === 'r7' ? 'Within-hand switch constraint; NOT an exhaustive primary class label.' :
        pack === 'r2' ? 'Original speed ladder answer; C48 Anchor overlap clarified by annotations-r10.json; parenthesized Vibro not mandatory.' :
          'Player local primary and non-exhaustive components.',
      outputs });
  }
}
const numericErrors = synthetic.flatMap(c => c.outputs.libraryDefault.segments
  .filter(s => s.equivBpm !== null && s.equivBpm !== c.expectedBpm)
  .map(s => ({ id: c.id, expected: c.expectedBpm, actual: s.equivBpm, tag: s.tag })));
const report = {
  created: new Date().toISOString(), scope: 'Constructed cadence checks and actively selected player examples, not population accuracy.',
  noTrainingOrSelection: true,
  defaultAndBrowserAndUpstreamDecisionParity: [...synthetic, ...local].every(c =>
    JSON.stringify(displayedDecision(c.outputs.libraryDefault)) === JSON.stringify(displayedDecision(c.outputs.upstreamOverlaySnapshot))),
  classificationAndWholeTagsMatchUpstream: [...synthetic, ...local].every(c => {
    const a = c.outputs.libraryDefault, b = c.outputs.upstreamOverlaySnapshot;
    return JSON.stringify(a.segments.map(s => [s.tag, s.secondary, s.confidence])) ===
      JSON.stringify(b.segments.map(s => [s.tag, s.secondary, s.confidence])) && JSON.stringify(a.wholeTags) === JSON.stringify(b.wholeTags);
  }),
  syntheticCases: synthetic.length, numericErrorSegments: numericErrors.length, numericErrors,
  synthetic, local,
  integration: {
    snapshot: '.upstream/index.html', script: 'tag_engine.js',
    invocation: 'TagClassifier.analyzeMap(content, { speedRate })',
    display: 'seg.tag, seg.secondary, seg.equivBpm and seg.confidence',
    dependency: 'The plugin snapshot loads its own embedded engine; changing the standalone library alone does not update this overlay.',
    liveTosuSessionVerified: false,
  },
};
fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n');
console.log(JSON.stringify({ syntheticCases: synthetic.length, numericErrorSegments: numericErrors.length,
  localCases: local.length, parity: report.defaultAndBrowserAndUpstreamDecisionParity,
  example: numericErrors.find(x => x.id === 'uniform-50-phase-0-rate-1') }, null, 2));
