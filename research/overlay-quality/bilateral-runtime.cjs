'use strict';
// Exact native consumer for the bounded study. The archive access dependency
// is replaced at BUILD TIME before any browser/product use (same as LN core).
const { loadLab } = require('../rework/legacy-lab.cjs');
const { parseOsu, prepare, lowerBound, popcount } = require('../../src/parser.cjs');
const { handCuts } = require('./cut-features.cjs');
const TYPES = ['dense chordstream', 'fast chordstream', 'tech'];
const physicalPatch = { id: 'bilateral-physical-partition', apply(source, replace) {
  source = replace(source, '  var SEG_FEATURES = [', '  var SEG_FEATURES = [\n    "physicalRowRate",\n    "bilateralRowShare",');
  return replace(source, '      nps: notes / durS,', `      nps: notes / durS,
      physicalRowRate: rowCount / durS,
      bilateralRowShare: (()=>{let both=0;for(let j=loIdx;j<hiIdx;j++){const row=rows[j];if(row.t>=t0&&row.t<t1&&row.hits>0&&(row.mask&7)&&(row.mask&112))both++;}return rowCount?both/rowCount:0;})(),`);
} };
const displayPatch = { id: 'bilateral-fixed-display-smoothing', apply(source, replace) {
  return replace(source, '    const minDisplay = CONFIG.minSegLenMs * 0.6;', '    const minDisplay = 1800;');
} };
function bilateral(ctx, start, end) {
  let rows = 0, notes = 0, bothRows = 0, bothNotes = 0;
  const lo = lowerBound(ctx.rows, start, r => r.t), hi = lowerBound(ctx.rows, end, r => r.t);
  for (let i = lo; i < hi; i++) {
    const mask = ctx.rows[i].mask, n = popcount(mask);rows++;notes += n;
    if ((mask & 7) && (mask & 112)) { bothRows++;bothNotes += n; }
  }
  return [rows ? bothRows / rows : 0, notes ? bothNotes / notes : 0];
}
function inputs(ctx, start, end, kind) {
  if (!['bias', 'hand', 'bilateral', 'both'].includes(kind)) throw Error('Invalid bounded representation');
  return [...(['hand', 'both'].includes(kind) ? handCuts(ctx, start, end).values : []),
    ...(['bilateral', 'both'].includes(kind) ? bilateral(ctx, start, end) : [])];
}
function design(raw, scaler) {
  const filled = raw.map((x, i) => Number.isFinite(x) ? x : scaler.rawImputation[i]);
  // JS null is missing, not a finite zero. Number.isFinite preserves this.
  const expanded = [...filled, ...raw.map(x => Number(!Number.isFinite(x)))];
  return [1, ...expanded.flatMap((x, i) => scaler.keep[i] ? [(x - scaler.mean[i]) / scaler.scale[i]] : [])];
}
function correctFrame(frame, raw, model) {
  if (!model || model.theta.every(x => x === 0) || frame.family !== 'rc') return { ...frame };
  const original = frame.scores ?? { [frame.tag]: 1 }, total = Object.values(original).reduce((a, b) => a + b, 0);
  const values = TYPES.map(k => original[k] ?? 0), mass = values.reduce((a, b) => a + b, 0);
  if (!(mass > 0)) return { ...frame };
  const x = design(raw, model.scaler);
  if (x.length !== model.theta.length) throw Error('Bounded model/scaler dimensions differ');
  const delta = x.reduce((s, v, i) => s + v * model.theta[i], 0);
  // Log-domain normalization preserves zero support and finite extremes.
  const logits = values.map((v, i) => v > 0 ? Math.log(v / mass) + (i === 2 ? delta : 0) : -Infinity);
  const peak = Math.max(...logits), exponent = logits.map(z => Math.exp(z - peak)), normalizer = exponent.reduce((a, b) => a + b, 0);
  const scores = { ...original };
  TYPES.forEach((k, i) => { scores[k] = exponent[i] / normalizer * (mass / total) * total; });
  const ranked = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  let [tag, confidence] = ranked[0];
  if (tag === 'wildcard' || tag === 'ln wildcard') {
    for (const [other, value] of ranked.slice(1)) if (other !== 'wildcard' && other !== 'ln wildcard' && value > .3) {
      tag = other;confidence = value;break;
    }
  }
  const secondary = ranked.length > 1 && ranked[1][1] >= ranked[0][1] * .75 && ranked[1][1] > .3 ? ranked[1][0] : null;
  return { ...frame, scores, tag, confidence: Math.min(1, confidence), secondary };
}
function createConsumer({ variant, partition, model = null, policy }) {
  if (!['original', 'physical'].includes(variant) || !Number.isInteger(partition) || partition < 0 || partition > 8)
    throw Error('Invalid bounded partition');
  const config = { minSegLenMs: [1500, 3000, 6000][Math.floor(partition / 3)], segProminenceAbs: [.025, .05, .1][partition % 3] };
  const api = loadLab({ patches: [displayPatch, ...(variant === 'physical' ? [physicalPatch] : [])], config });
  return function analyze(text, options = {}, preparedContext) {
    // Use the established production cache path, not the research inspector
    // that also computes four unused rule/model diagnostic heads per segment.
    let map = api.lab.parseOsu(text);
    if ((options.speedRate ?? 1) !== 1) map = api.lab.rescale(map, options.speedRate);
    const cm = api.buildCachedMap(text, options), classified = api.lab.classifySegments(cm, { scorer: 'model' });
    // The library can share its existing prepared rows; a constant/zero head
    // does not need an extra modern parse or hand geometry extraction at all.
    const activeModel = model && model.theta.some(v => v !== 0);
    const needsRows = activeModel && model.kind !== 'bias';
    const ctx = needsRows ? (preparedContext ?? prepare(parseOsu(text, options))) : null;
    const frames = classified.map(frame => correctFrame(frame,
      activeModel ? inputs(ctx, frame.start, frame.end, model.kind) : [], model));
    const segments = api.lab.smoothSegments(frames.map(s => ({ ...s })));
    return { map, rawFrames: frames, segments, overall: api.lab.aggregateOverall(segments, policy),
      debug: { windows: cm.windows, norm: undefined, ctx: undefined } };
  };
}
module.exports = { createConsumer, correctFrame, design, inputs, bilateral, physicalPatch, displayPatch };
