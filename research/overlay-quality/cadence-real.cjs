'use strict';
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const api = require('../..'), { records7 } = require('../rework/data.cjs');
const { parseOsu, prepare } = require('../../src/parser.cjs');
const { correctCadenceDisplay } = require('../../src/cadence.cjs');
const output = path.resolve(__dirname, process.argv[2] ?? 'cadence-real-r1.json');
if (fs.existsSync(output)) throw new Error('Preserve completed cadence evidence; supply a new output filename');
const baseline = new Map(JSON.parse(fs.readFileSync('.cache/rework-legacy-full.json', 'utf8')).map(r => [r.uid, r]));
const changes = [], measurements = []; let charts = 0, segments = 0;
for (const record of records7()) {
  // Isolate the display treatment even after the stable API gains a separate
  // rest-gate fix. Do not redefine this historical classification-preservation check.
  const text = fs.readFileSync(record.path, 'utf8'), actual = api.legacy.analyzeMap(text), expected = baseline.get(record.uid);
  actual.modelInfo = {};
  actual.segments.forEach(s => { s.evidence = {}; });
  if (actual.segments.length) { actual.segments[0].start = actual.map.firstNote; actual.segments.at(-1).end = actual.map.lastNote; }
  correctCadenceDisplay(actual, prepare(parseOsu(text)));
  assert.ok(expected, record.uid);
  for (const key of ['tags', 'weights', 'sort']) assert.deepEqual(actual.overall[key], expected.overall[key], `${record.uid} overall.${key}`);
  assert.equal(actual.segments.length, expected.segments.length, record.uid);
  for (let i = 0; i < actual.segments.length; i++) {
    const a = actual.segments[i], b = expected.segments[i];
    for (const key of ['tag', 'family', 'confidence', 'secondary', 'scores']) assert.deepEqual(a[key], b[key], `${record.uid} ${i} ${key}`);
    assert.equal(a.start, i === 0 ? actual.map.firstNote : b.start);
    assert.equal(a.end, i === actual.segments.length - 1 ? actual.map.lastNote : b.end);
    if (a.evidence.cadence) measurements.push({ uid: record.uid, segment: i, start: a.start, end: a.end, tag: a.tag, ...a.evidence.cadence });
    if (a.equivBpm !== b.equivBpm) {
      assert.ok(a.evidence.cadence, 'Every changed number must have a direct cadence measurement');
      assert.equal(a.evidence.cadence.legacyEquivBpm, b.equivBpm);
      changes.push({ uid: record.uid, segment: i, start: a.start, end: a.end, tag: a.tag,
        before: b.equivBpm ?? null, after: a.equivBpm, measurement: a.evidence.cadence });
    }
    segments++;
  }
  charts++;
  if (charts % 50 === 0) console.log('cadence real charts', charts, 'changed numbers', changes.length);
}
const result = { created: new Date().toISOString(), scope: 'Complete existing 7K corpus vs archived cached original predictions; direct cadence display correction only.',
  charts, segments, classificationAndWholeDecisions: 'exact unchanged', measuredSegments: measurements.length,
  changedBpmSegments: changes.length, changes, measurements,
  accuracyClaim: 'No change to class P/R/F1. No new human BPM ground truth; changed values are direct near-uniform adjacent-row measurements.' };
fs.writeFileSync(output, JSON.stringify(result, null, 2) + '\n');
console.log(JSON.stringify({ charts, segments, changedBpmSegments: changes.length, measuredSegments: measurements.length }, null, 2));
