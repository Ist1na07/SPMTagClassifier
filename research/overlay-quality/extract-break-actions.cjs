'use strict';
const fs = require('node:fs'), assert = require('node:assert/strict');
const { loadLab, context, inspectPrepared } = require('../rework/legacy-lab.cjs');
const { records7, scoreIntervals } = require('../rework/data.cjs');
const { actionRatePatch } = require('./break-regularity.cjs');
const root = 'research/overlay-quality/break-regularity-r3';
const ratios = [0, .25, .4, .55, .7, .85, .95], rowCaps = [0, 4, 8, 12, 16, 24, 32, 48];
// ratio1 is the original irrespective of row cap; retain it ONCE, not eight
// duplicate flat parameters. cap0 is the complete previous ratio ablation.
const configs = [{ ratio: 1, rowRateMax: 0 }, ...ratios.flatMap(ratio => rowCaps.map(rowRateMax => ({ ratio, rowRateMax })))];
fs.mkdirSync(root, { recursive: true });
const protocol = { configs, hypothesis: 'NPS confounds more chord notes with faster successive actions. Preserve a regular low-action-rate rest even when its NPS is near the map median, while avoiding high-action-rate Speed->Break.',
  measurementsBeforeStudy: 'DS rest:25NPS,14.116rows/s; F01 Speed:22NPS,16.801rows/s; F02 Speed:31NPS,21.212rows/s. These actively studied examples motivate representation, not held-out performance.',
  stop: 'One complete two-coordinate grid, matched cap0 and original endpoints. No finer threshold expansion or new gate model in this product iteration if not worthwhile.',
  sourceSnapshots: Object.fromEntries(['extract-break-actions.cjs', 'break-regularity.cjs'].map(f => [f, fs.readFileSync(__dirname + '/' + f, 'utf8')])) };
if (fs.existsSync(root + '/extraction-protocol.json')) assert.deepEqual(JSON.parse(fs.readFileSync(root + '/extraction-protocol.json', 'utf8')), protocol);
else fs.writeFileSync(root + '/extraction-protocol.json', JSON.stringify(protocol, null, 2) + '\n');
if (fs.existsSync(root + '/complete.json')) throw Error('Preserve completed evidence');
const prior = new Map(JSON.parse(fs.readFileSync('research/overlay-quality/break-regularity-r2/frames.json', 'utf8')).map(r => [r.uid, r]));
const oldRatios = JSON.parse(fs.readFileSync('research/overlay-quality/break-regularity-r2/complete.json', 'utf8')).ratios;
const apis = configs.map(c => loadLab({ patches: [actionRatePatch(c)] }));
const rows = [], local = [], player = [];
const frames = value => value.map(s => ({ ...s, secondary: s.secondary ?? null, scores: s.scores ?? { [s.tag]: 1 } }));
let replayed = 0;
for (const record of records7()) {
  const text = fs.readFileSync(record.path, 'utf8'), ctx = context(apis[0].lab, text);
  const original = inspectPrepared(apis[0], ctx);
  const values = apis.map((a, i) => frames(i === 0 ? original.classified : a.lab.classifySegments(original.cache, { scorer: 'model' })));
  for (let i = 0; i < configs.length; i++) if (configs[i].rowRateMax === 0) {
    assert.deepEqual(JSON.parse(JSON.stringify(values[i])), prior.get(record.uid).frames[oldRatios.indexOf(configs[i].ratio)]);
    replayed++;
  }
  rows.push({ uid: record.uid, frames: values });
  if (record.dataset === 'manual') local.push({ uid: record.uid, scores: values.map(f => scoreIntervals(record.annotations, f)) });
}
for (const name of ['F01', 'F02']) {
  const ctx = context(apis[0].lab, fs.readFileSync(`artifacts/frontier-annotation-r5/${name}.osu`, 'utf8'));
  const original = inspectPrepared(apis[0], ctx);
  player.push({ uid: name, targetMs: [9000, 16500], truth: name === 'F01' ? ['speed'] : ['speed', 'tech'],
    frames: apis.map(a => frames(a.lab.classifySegments(original.cache, { scorer: 'model' }))) });
}
for (const [file, value] of Object.entries({ frames: rows, local, player })) fs.writeFileSync(`${root}/${file}.json`, JSON.stringify(value) + '\n');
fs.writeFileSync(root + '/complete.json', JSON.stringify({ records: rows.length, configs, replayedOldCases: replayed,
  ratingPredictorCalls: 0, newModelFits: 0 }, null, 2) + '\n');
console.log('Break action-rate banks complete', rows.length, configs.length, 'old cases replayed', replayed);
