const test = require('node:test');
const assert = require('node:assert/strict');
const api = require('../src/legacy/tag-engine.cjs');
const text = '[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n' + Array.from({length:160}, (_, i) => `${36 + (i%7)*73},192,${1000+i*125},1,0,0:0:0:0:`).join('\n');
test('exports the three original plugin entry points', () => {
  assert.deepEqual(Object.keys(api).sort(), ['analyzeMap','buildCachedMap','extractIntermediate']);
});
test('classifies a 7K map without the overlay or tosu', () => {
  const r = api.analyzeMap(text); assert.equal(r.map.keys, 7);
  assert.ok(r.segments.length); assert.ok(Array.isArray(r.overall.tags));
});
test('rate mod rescales time', () => {
  assert.equal(api.analyzeMap(text,{speedRate:2}).map.firstNote,500);
});
