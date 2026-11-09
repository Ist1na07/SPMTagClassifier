'use strict';
const fs = require('node:fs'), assert = require('node:assert/strict');
const { loadLab, context, inspectPrepared } = require('../rework/legacy-lab.cjs');
const { records7, scoreIntervals } = require('../rework/data.cjs');
const { patch, ID } = require('./break-regularity.cjs');
const root = 'research/overlay-quality/break-regularity-r1';
if (fs.existsSync(root + '/complete.json')) throw Error('Preserve completed evidence');
fs.mkdirSync(root, { recursive: true });
const before = loadLab(), after = loadLab({ patches: [patch] });
const previous = new Map(fs.readFileSync('.cache/frontier-rc-type-inputs-r1.jsonl', 'utf8').trim().split('\n').map(s => { const r = JSON.parse(s); return [r.uid, r]; }));
const frames = v => v.classified.map(s => ({ ...s, secondary: s.secondary ?? null, scores: s.scores ?? { [s.tag]: 1 } }));
const rows = [], changes = [], local = [];
for (const record of records7()) {
  const text = fs.readFileSync(record.path, 'utf8');
  // Only classify-time logic changes: sharing extracted context is exact.
  const prepared = context(before.lab, text);
  const a = inspectPrepared(before, prepared), b = inspectPrepared(after, prepared);
  const old = frames(a), candidate = frames(b), cached = previous.get(record.uid);
  assert.deepEqual(old.map(s => [s.start, s.end, s.tag, s.family, s.confidence, s.scores]),
    cached.frames.map(s => [s.start, s.end, s.tag, s.family, s.confidence, s.scores]), record.uid);
  assert.deepEqual(candidate.map(s => [s.start, s.end]), old.map(s => [s.start, s.end]));
  rows.push({ uid: record.uid, original: old, candidate });
  for (let i = 0; i < old.length; i++) if (old[i].tag !== candidate[i].tag) {
    assert.equal(old[i].tag, 'break');
    const { f, extra } = a.segments[i];
    changes.push({ uid: record.uid, start: old[i].start, end: old[i].end, before: old[i].tag,
      after: candidate[i].tag, nps: f.nps, chordMean: f.chordMean,
      vibroFraction: extra.vibroFraction, rhythmChangeRate: extra.segRhythmChangeRate });
  }
  if (record.dataset === 'manual') local.push({ uid: record.uid,
    original: scoreIntervals(record.annotations, old), candidate: scoreIntervals(record.annotations, candidate) });
}
const player = [];
for (const name of ['F01', 'F02']) {
  const text = fs.readFileSync(`artifacts/frontier-annotation-r5/${name}.osu`, 'utf8');
  const ctx = context(before.lab, text);
  player.push({ uid: name, targetMs: [9000, 16500], original: frames(inspectPrepared(before, ctx)),
    candidate: frames(inspectPrepared(after, ctx)), truth: name === 'F01' ? ['speed'] : ['speed', 'tech'],
    truthScope: 'Actively queried local primary/components; not a random held-out sample.' });
}
for (const [file, value] of Object.entries({ frames: rows, changes, local, player }))
  fs.writeFileSync(`${root}/${file}.json`, JSON.stringify(value, null, 2) + '\n');
fs.writeFileSync(root + '/complete.json', JSON.stringify({ id: ID, records: rows.length,
  changedSegments: changes.length, noNewParameters: true, baselineParity: rows.length,
  sourceSnapshots: Object.fromEntries(['break-regularity.cjs', 'extract-break-regularity.cjs'].map(f =>
    [f, fs.readFileSync(__dirname + '/' + f, 'utf8')])) }, null, 2) + '\n');
console.log(JSON.stringify({ records: rows.length, changedSegments: changes.length,
  local: local.map(r => ({ uid: r.uid, original: r.original.accuracy, candidate: r.candidate.accuracy })) }, null, 2));
