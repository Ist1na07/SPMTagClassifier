'use strict';
const test = require('node:test'), assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const native = require('./bilateral-runtime.cjs'), { compileBilateralEngine } = require('./compile-bilateral-engine.cjs');
const legacy = require('../../src/legacy/tag-engine.cjs');
const serial = x => JSON.parse(JSON.stringify(x));
const normal = frames => serial(frames.map(f => ({ ...f, secondary: f.secondary ?? null, scores: f.scores ?? { [f.tag]: 1 } })));
const policy = { dominantShare: .52, mixShare: .07, secondRel: .75 };
const fixturePath = process.env.SPM_BILATERAL_FIXTURE ?? '.cache/overlay-bilateral-native-fixture-r1.json';
function browser(variant) {
  const result = compileBilateralEngine(variant), c = {};
  vm.runInNewContext('var Bounded=' + result.code, c);return c.Bounded;
}
test('static native consumer replays the original public pipeline at partition4, including rate conversion', () => {
  const staticApi = browser('original');
  const a = native.createConsumer({ variant: 'original', partition: 4, policy });
  const b = staticApi.createConsumer({ variant: 'original', partition: 4, policy });
  for (const name of ['F01', 'F02']) {
    const text = fs.readFileSync(`artifacts/frontier-annotation-r5/${name}.osu`, 'utf8');
    for (const speedRate of [.75, 1, 1.5]) {
      const expected = legacy.analyzeMap(text, { speedRate });
      for (const fn of [a, b]) {
        const { rawFrames, ...result } = fn(text, { speedRate });assert.deepEqual(serial(result), serial(expected));
      }
    }
  }
});
test('both segmentation signal variants reproduce the cached native DS partition0 frames', () => {
  for (const variant of ['original', 'physical']) {
    const row = fs.readFileSync(`.cache/overlay-bilateral-partitions-r1/${variant}-0.jsonl`, 'utf8').trim().split('\n').map(JSON.parse).find(r => r.uid === 'manual/DS');
    const text = fs.readFileSync(row.source, 'utf8'), staticApi = browser(variant);
    for (const api of [native, staticApi]) {
      const result = api.createConsumer({ variant, partition: 0, policy })(text);
      assert.deepEqual(normal(result.rawFrames), normal(row.frames));
    }
  }
});
test('accepted Python correction states match JS scores and labels, preserving every other score', () => {
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const staticApi = browser('original');
  for (const c of fixture.cases) for (const api of [native, staticApi]) {
    const result = c.original.map((frame, j) => api.correctFrame(frame, c.features[j], c.model));
    for (let j = 0; j < result.length; j++) {
      assert.equal(result[j].tag, c.expected[j].tag, c.uid + '/' + c.kind);
      assert.equal(result[j].secondary ?? null, c.expected[j].secondary ?? null);
      assert.ok(Math.abs(result[j].confidence - c.expected[j].confidence) < 1e-11);
      for (const [key, value] of Object.entries(c.expected[j].scores)) {
        assert.ok(Math.abs(result[j].scores[key] - value) < 1e-11, key);
        if (!['tech', 'dense chordstream', 'fast chordstream'].includes(key)) assert.equal(result[j].scores[key], c.original[j].scores[key]);
      }
    }
  }
});
test('feature missingness is not silently coerced to zero and zero correction is exact', () => {
  const scaler = { rawImputation: [2], mean: [1, .5], scale: [2, .5], keep: [true, true] };
  assert.deepEqual(native.design([null], scaler), [1, .5, 1]);
  const frame = { tag: 'tech', family: 'rc', scores: { tech: .1, 'dense chordstream': .8, speed: .1 }, confidence: .1 };
  assert.deepEqual(native.correctFrame(frame, [], { theta: [0] }), frame);
});
test('zero nonconstant heads skip feature extraction and replay the native public result', () => {
  const text = fs.readFileSync('artifacts/frontier-annotation-r5/F02.osu', 'utf8');
  const expected = serial(legacy.analyzeMap(text));
  for (const api of [native, browser('original')]) {
    const { rawFrames, ...result } = api.createConsumer({ variant: 'original', partition: 4, policy,
      model: { kind: 'hand', theta: [0] } })(text);
    assert.deepEqual(serial(result), expected);
  }
});
test('full raw-text consumers reproduce Python corrected real-chart frames, not just supplied feature vectors', () => {
  const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));
  const staticApi = browser('original'), texts = new Map();
  for (const c of fixture.cases.filter(r => r.headId === 55 && !r.uid.startsWith('concept/'))) {
    if (!texts.has(c.source)) texts.set(c.source, fs.readFileSync(c.source, 'utf8'));
    for (const api of [native, staticApi]) {
      const result = api.createConsumer({ variant: 'original', partition: 0, policy, model: c.model })(texts.get(c.source));
      assert.equal(result.rawFrames.length, c.expected.length);
      for (let j = 0; j < c.expected.length; j++) {
        const a = result.rawFrames[j], b = c.expected[j];
        assert.deepEqual([a.start, a.end, a.tag, a.secondary ?? null], [b.start, b.end, b.tag, b.secondary ?? null]);
        for (const [key, value] of Object.entries(b.scores)) assert.ok(Math.abs((a.scores ?? { [a.tag]: 1 })[key] - value) < 1e-11, c.uid + '/' + c.kind + '/' + key);
      }
    }
  }
});
