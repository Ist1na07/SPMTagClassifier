'use strict';
const { lowerBound } = require('./parser.cjs');

// A display measurement, not a replacement for the archived model's features.
// Round-to-ms timestamps on a uniform sequence can deviate from the line
// through its two endpoints by at most one source millisecond. This bound
// scales with the clock rate; it is not a fitted rhythm/confidence threshold.
function regularRowCadence(ctx, start, end, speedRate = 1) {
  const { rows } = ctx;
  const lo = lowerBound(rows, start, r => r.t), hi = lowerBound(rows, end, r => r.t);
  const count = hi - lo;
  // Match the existing grid fitter's minimum four-row support.
  if (count < 4) return null;
  const first = rows[lo].t, spacingMs = (rows[hi - 1].t - first) / (count - 1);
  const resolutionMs = 1 / speedRate;
  let maxResidualMs = 0;
  for (let i = 0; i < count; i++) {
    maxResidualMs = Math.max(maxResidualMs, Math.abs(rows[lo + i].t - (first + i * spacingMs)));
  }
  const epsilon = 16 * Number.EPSILON * Math.max(1, Math.abs(first), Math.abs(rows[hi - 1].t));
  if (!(spacingMs > 0) || maxResidualMs > resolutionMs + epsilon) return null;
  const bpm = 15000 / spacingMs;
  return { source: 'uniform-rows-within-source-ms', equivBpm: Math.round(bpm + 16 * Number.EPSILON * Math.max(1, bpm)),
    rowCount: count, spacingMs, maxResidualMs, resolutionMs };
}

const DISPLAY_TAGS = new Set(['chordjack', 'minijack', 'dense chordstream', 'fast chordstream',
  'speed', 'tech', 'wildcard', 'coordination', 'density']);
function jackClockMatchesRows(ctx, start, end) {
  const lo = lowerBound(ctx.rows, start, r => r.t), hi = lowerBound(ctx.rows, end, r => r.t);
  if (lo === hi) return false;
  const firstMask = ctx.rows[lo].mask;
  let allChords = true, sameGrip = true;
  for (let i = lo; i < hi; i++) {
    const mask = ctx.rows[i].mask;
    allChords &&= (mask & (mask - 1)) !== 0;
    sameGrip &&= mask === firstMask;
  }
  // The archive measures chord-row cadence for Jack, falling back to
  // same-column repetition. Do not replace either by an interleaved row clock.
  return allChords || sameGrip;
}
function correctCadenceDisplay(result, ctx, speedRate = 1) {
  for (const segment of result.segments) {
    if (!DISPLAY_TAGS.has(segment.tag)) continue;
    const start = segment.evidence?.sourceStart ?? segment.start;
    const end = segment.evidence?.sourceEnd ?? segment.end;
    if (['chordjack', 'minijack'].includes(segment.tag) && !jackClockMatchesRows(ctx, start, end)) continue;
    const measured = regularRowCadence(ctx, start, end, speedRate);
    if (!measured) continue;
    const { equivBpm, ...evidence } = measured;
    segment.evidence.cadence = { ...evidence, measuredStart:start,measuredEnd:end,legacyEquivBpm: segment.evidence.archivedEquivBpm??segment.equivBpm };
    segment.equivBpm = equivBpm;
  }
  result.modelInfo.cadencePolicy = 'uniform rows measured directly; other archived cadence unchanged';
  return result;
}
module.exports = { regularRowCadence, correctCadenceDisplay };
