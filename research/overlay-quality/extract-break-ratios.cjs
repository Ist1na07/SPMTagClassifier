'use strict';
const fs = require('node:fs'), assert = require('node:assert/strict');
const { loadLab, context, inspectPrepared } = require('../rework/legacy-lab.cjs');
const { records7, scoreIntervals } = require('../rework/data.cjs');
const { ratioPatch } = require('./break-regularity.cjs');
const root = 'research/overlay-quality/break-regularity-r2';
const ratios = [0, .25, .4, .55, .7, .85, .95, 1];
fs.mkdirSync(root, { recursive: true });
const protocol = { ratios, change: 'Only the regular-small-chord rest disjunct gains a relative NPS upper bound; original outer f.nps<mapNps and all other branches remain.',
  endpoints: '0=delete disjunct; 1=original. Both verified against completed r1 frames.',
  sourceSnapshots: Object.fromEntries(['extract-break-ratios.cjs', 'break-regularity.cjs'].map(f => [f, fs.readFileSync(__dirname + '/' + f, 'utf8')])) };
if (fs.existsSync(root + '/extraction-protocol.json')) assert.deepEqual(JSON.parse(fs.readFileSync(root + '/extraction-protocol.json', 'utf8')), protocol);
else fs.writeFileSync(root + '/extraction-protocol.json', JSON.stringify(protocol, null, 2) + '\n');
if (fs.existsSync(root + '/complete.json')) throw Error('Preserve completed evidence');
const original = new Map(JSON.parse(fs.readFileSync('research/overlay-quality/break-regularity-r1/frames.json', 'utf8')).map(r => [r.uid, r]));
const apis = ratios.map(r => loadLab({ patches: [ratioPatch(r)] }));
const rows = [], local = [], player = [];
const frames = v => v.classified.map(s => ({ ...s, secondary: s.secondary ?? null, scores: s.scores ?? { [s.tag]: 1 } }));
for (const record of records7()) {
  const text = fs.readFileSync(record.path, 'utf8'), ctx = context(apis[0].lab, text);
  const views = apis.map(a => inspectPrepared(a, ctx)), values = views.map(frames), cached = original.get(record.uid);
  // r1 is JSON evidence: undefined optional fields are absent after transport.
  // Compare the complete serialized frames, including every numerical score.
  assert.deepEqual(JSON.parse(JSON.stringify(values[0])), cached.candidate);
  assert.deepEqual(JSON.parse(JSON.stringify(values.at(-1))), cached.original);
  rows.push({ uid: record.uid, frames: values });
  if (record.dataset === 'manual') local.push({ uid: record.uid,
    scores: values.map(f => scoreIntervals(record.annotations, f)) });
}
for (const name of ['F01', 'F02']) {
  const ctx = context(apis[0].lab, fs.readFileSync(`artifacts/frontier-annotation-r5/${name}.osu`, 'utf8'));
  player.push({ uid: name, targetMs: [9000, 16500], truth: name === 'F01' ? ['speed'] : ['speed', 'tech'],
    frames: apis.map(a => frames(inspectPrepared(a, ctx))) });
}
for (const [file, value] of Object.entries({ frames: rows, local, player }))
  fs.writeFileSync(`${root}/${file}.json`, JSON.stringify(value) + '\n');
fs.writeFileSync(root + '/complete.json', JSON.stringify({ records: rows.length, ratios, endpointParity: rows.length * 2, source: 'Existing notes and feature windows; no rating predictor, no model fit.' }, null, 2) + '\n');
console.log('Break ratio banks complete', rows.length, ratios.length);
