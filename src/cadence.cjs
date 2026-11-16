'use strict';
const { lowerBound } = require('./parser.cjs');
const {jackCadence,switchCadence,intervalMode}=require('./action-clock.cjs');

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
function measureCadence(ctx,start,end,tag,speedRate=1){
  if(!DISPLAY_TAGS.has(tag))return null;
  if(['chordjack','minijack'].includes(tag))return jackCadence(ctx,start,end,tag,speedRate);
  if(['dense chordstream','fast chordstream'].includes(tag))return switchCadence(ctx,start,end,speedRate);
  const uniform=regularRowCadence(ctx,start,end,speedRate);if(uniform)return uniform;
  const rows=ctx.rows.slice(lowerBound(ctx.rows,start,r=>r.t),lowerBound(ctx.rows,end,r=>r.t));
  const pairs=rows.slice(1).map((r,i)=>({dt:r.t-rows[i].t,width:1})),m=intervalMode(pairs,1/speedRate);
  return m?{...m,source:'head-articulation-interval-mode',resolutionMs:1/speedRate,rowCount:rows.length,equivBpm:Math.round(15000/m.spacingMs)}:null;
}
function legacyRowClockAllowed(ctx,start,end,tag){
  if(!['chordjack','minijack'].includes(tag))return true;
  const rows=ctx.rows.slice(lowerBound(ctx.rows,start,r=>r.t),lowerBound(ctx.rows,end,r=>r.t));
  return rows.length>0&&(rows.every(r=>(r.mask&(r.mask-1))!==0)||rows.every(r=>r.mask===rows[0].mask));
}
function correctCadenceDisplay(result, ctx, speedRate = 1, {actionClocks=true}={}) {
  for (const segment of result.segments) {
    if (!DISPLAY_TAGS.has(segment.tag)) continue;
    const start = segment.evidence?.sourceStart ?? segment.start;
    const end = segment.evidence?.sourceEnd ?? segment.end;
    delete segment.evidence.secondaryCadence;
    const measured=actionClocks?measureCadence(ctx,start,end,segment.tag,speedRate):
      legacyRowClockAllowed(ctx,start,end,segment.tag)?regularRowCadence(ctx,start,end,speedRate):null;
    if(!actionClocks&&!measured)continue;
    if(actionClocks&&segment.secondary){
      const secondary=measureCadence(ctx,start,end,segment.secondary,speedRate);
      if(secondary)segment.evidence.secondaryCadence={tag:segment.secondary,...secondary,measuredStart:start,measuredEnd:end};
    }
    if(!measured){delete segment.equivBpm;segment.evidence.cadenceUnresolved='no dominant, type-appropriate action period';delete segment.evidence.cadence;continue;}
    delete segment.evidence.cadenceUnresolved;
    const { equivBpm, ...evidence } = measured;
    segment.evidence.cadence = { ...evidence, measuredStart:start,measuredEnd:end,legacyEquivBpm: segment.evidence.archivedEquivBpm??segment.equivBpm };
    segment.equivBpm = equivBpm;
  }
  result.modelInfo.cadencePolicy = !actionClocks?'uniform rows measured directly; other archived cadence unchanged':'type-conditional observed action periods: per-hand strikes, same-finger rearticulation, within-hand substitution, or head articulation; unsupported scalar tempo omitted';
  return result;
}
module.exports = { regularRowCadence, measureCadence, correctCadenceDisplay };
