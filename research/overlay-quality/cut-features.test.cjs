const test = require('node:test'), assert = require('node:assert/strict');
const { parseOsu, prepare } = require('../../src/parser.cjs');
const { handCuts } = require('./cut-features.cjs');
function input(masks, rate = 1, mirror = false) {
  const text = '[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[HitObjects]\n' + masks.flatMap((m, i) =>
    m.map(c => `${((mirror ? 6 - c : c) + .5) * 512 / 7},192,${1000 + i * 100},1,0,0:0:0:0:`)).join('\n');
  return prepare(parseOsu(text, { speedRate: rate }));
}
test('alternating fixed grips are not substitutions; actual within-hand cuts are', () => {
  const fixed = [[0, 1, 2], [4, 5], [0, 1, 2], [4, 5]];
  const cut = [[0, 2, 4], [1, 5], [0, 2, 4], [1, 5]];
  assert.deepEqual(handCuts(input(fixed), 0, 2000).values, [0, 1]);
  assert.deepEqual(handCuts(input(cut), 0, 2000).values, [1, 0]);
});
test('the other hand does not reset finger history; mirror and rate preserve fractions', () => {
  const masks = [[0], [4], [1], [5], [2], [6]];
  for (const rate of [.75, 1, 1.5]) for (const mirror of [false, true])
    assert.deepEqual(handCuts(input(masks, rate, mirror), 0, 3000).values, [1, 0]);
});
test('unobserved hand transitions remain missing and the center is not assigned to a fixed hand', () => {
  assert.deepEqual(handCuts(input([[3], [3], [3], [3]]), 0, 2000).values, [null, null]);
  assert.deepEqual(handCuts(input([[0], [4], [1], [5]]), 1000, 1200).values, [null, null]);
});
