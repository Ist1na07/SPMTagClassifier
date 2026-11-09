'use strict';
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { parseOsu, prepare } = require('../../src/parser.cjs'), { KEYS, handCuts } = require('./cut-features.cjs');
const root = path.resolve(__dirname, '../..'), output = path.join(root, '.cache/overlay-cut-inputs-r1.jsonl');
if (fs.existsSync(output)) throw new Error('Preserve existing overlay cut inputs');
const rows = fs.readFileSync(path.join(root, '.cache/frontier-rc-type-inputs-r1.jsonl'), 'utf8').trim().split('\n').map(JSON.parse);
const contexts = new Map(), result = [];
for (const row of rows) {
  if (!contexts.has(row.source)) contexts.set(row.source, prepare(parseOsu(fs.readFileSync(path.join(root, row.source), 'utf8'))));
  const ctx = contexts.get(row.source);
  result.push({ uid: row.uid, source: row.source, intervals: row.intervals,
    windows: row.intervals.map(([a, b]) => handCuts(ctx, a, b)) });
}
for (const [name, expected] of [['H01', [0, 1]], ['H02', [1, 0]], ['H03', [0, 1]], ['H04', [1, 0]]]) {
  assert.deepEqual(result.find(r => r.uid === `concept/${name}`).windows[0].values, expected);
}
fs.writeFileSync(output, result.map(r => JSON.stringify(r)).join('\n') + '\n');
fs.writeFileSync(output + '.schema.json', JSON.stringify({ keys: KEYS, records: result.length,
  physicalSources: contexts.size, windows: result.reduce((n, r) => n + r.windows.length, 0), predictorCalls: 0,
  sourceSnapshots: Object.fromEntries(['cut-features.cjs', 'extract-cut.cjs'].map(f => [f, fs.readFileSync(path.join(__dirname, f), 'utf8')])) }, null, 2) + '\n');
console.log('Overlay cut inputs', result.length, 'sources', contexts.size, 'Predictor calls 0');
