const test = require('node:test'), assert = require('node:assert/strict');
const { parseOsu, prepare } = require('../src/parser.cjs');
const { regularRowCadence, correctCadenceDisplay } = require('../src/cadence.cjs');
const api = require('..');
function chart(spacing, phase = 0, gaps = null) {
  return '[General]\nMode:3\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n' +
    '[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n' +
    Array.from({ length: gaps?.length ?? 200 }, (_, i) => [0, 2, 4].map(c =>
      `${(c + .5) * 512 / 7},192,${gaps?.[i] ?? Math.round(1000 + phase + i * spacing)},1,0,0:0:0:0:`).join('\n')).join('\n');
}
test('uniform cadence survives integer-ms rounding, phase shifts and rate changes', () => {
  for (const spacing of [50, 62.5, 75, 100, 125, 150, 200, 250]) {
    for (const phase of [0, 13, 27]) for (const speedRate of [.75, 1, 1.5]) {
      const map = parseOsu(chart(spacing, phase), { speedRate });
      const measurement = regularRowCadence(prepare(map), map.firstNote, map.lastNote + 1, speedRate);
      assert.ok(measurement, `${spacing}/${phase}/${speedRate}`);
      assert.equal(measurement.equivBpm, Math.round(15000 * speedRate / spacing));
    }
  }
});
test('irregular rhythm, gaps and inadequate support do not receive a uniform measurement', () => {
  for (const times of [[1000, 1100, 1275, 1350, 1550], [1000, 1100, 1200, 1400, 1500], [1000, 1100, 1200]]) {
    const map = parseOsu(chart(null, 0, times));
    assert.equal(regularRowCadence(prepare(map), map.firstNote, map.lastNote + 1), null);
  }
});
test('measurement is bounded to its interval and does not absorb a different surrounding tempo', () => {
  const map = parseOsu(chart(null, 0, [100, 500, 1000, 1050, 1100, 1150, 2000, 2400]));
  const ctx = prepare(map);
  assert.equal(regularRowCadence(ctx, 1000, 1200).equivBpm, 300);
  assert.equal(regularRowCadence(ctx, 100, 2401), null);
});
test('Jack cadence keeps its repetition/chord clock when single rows interleave', () => {
  const rows = Array.from({ length: 12 }, (_, i) => ({ t: 1000 + i * 50, mask: i % 2 ? 3 : 1 }));
  for (const tag of ['chordjack', 'minijack']) {
    const result = { segments: [{ start: 1000, end: 1600, tag, equivBpm: 150, evidence: {} }], modelInfo: {} };
    correctCadenceDisplay(result, { rows });
    assert.equal(result.segments[0].equivBpm, 150);
    assert.equal(result.segments[0].evidence.cadence, undefined);
  }
  const result = { segments: [{ start: 1000, end: 1600, tag: 'chordjack', equivBpm: 299, evidence: {} }], modelInfo: {} };
  correctCadenceDisplay(result, { rows: rows.map(r => ({ ...r, mask: 3 })) });
  assert.equal(result.segments[0].equivBpm, 300);
});
test('7K display fixes the 300-to-200 alias without changing classification or aggregation', () => {
  const text = chart(50), old = api.legacy.analyzeMap(text);
  assert.equal(old.segments[0].equivBpm, 200);
  for (const algorithm of ['stable', 'modern']) {
    const r = api.analyzeMap(text, { algorithm });
    assert.equal(r.segments[0].equivBpm, 300);
    assert.equal(r.segments[0].evidence.cadence.legacyEquivBpm, 200);
  }
  const r = api.analyzeMap(text, {aggregation:'classic'});
  assert.deepEqual(r.overall, { ...old.overall, composition: r.overall.composition, scoreKind: r.overall.scoreKind });
  assert.deepEqual(r.segments.map(s => [s.tag, s.secondary, s.scores, s.confidence]),
    old.segments.map(s => [s.tag, s.secondary, s.scores, s.confidence]));
  assert.deepEqual(r.debug.windows, old.debug.windows);
});
