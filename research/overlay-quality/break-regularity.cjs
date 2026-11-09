'use strict';
// A parameter-free deletion: rhythmic regularity and small chord size alone
// do not imply rest, even when the segment is below the map median NPS.
const ID = 'remove-unbounded-regularity-break';
const CLAUSE = ' || f.chordMean < 1.9 && ex.vibroFraction > 0.3 && ex.segRhythmChangeRate < 0.1';
const patch = { id: ID, apply(source, replace) { return replace(source, CLAUSE, ''); } };
function ratioPatch(ratio) {
  if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1) throw Error('Invalid relative rest density');
  return { id: 'regularity-break-ratio-' + ratio, apply(source, replace) {
    return replace(source, CLAUSE, ` || f.nps < mapNps * ${ratio} && f.chordMean < 1.9 && ex.vibroFraction > 0.3 && ex.segRhythmChangeRate < 0.1`);
  } };
}
function actionRatePatch({ ratio, rowRateMax }) {
  if (!Number.isFinite(ratio) || ratio < 0 || ratio > 1 || !Number.isFinite(rowRateMax) || rowRateMax < 0)
    throw Error('Invalid rest activity parameters');
  return { id: 'regularity-break-action-rate', apply(source, replace) {
    return replace(source, CLAUSE, ` || (f.nps < mapNps * ${ratio} || ex.rowCount * 1000 / (s.end - s.start) < ${rowRateMax}) && f.chordMean < 1.9 && ex.vibroFraction > 0.3 && ex.segRhythmChangeRate < 0.1`);
  } };
}
module.exports = { ID, CLAUSE, patch, ratioPatch, actionRatePatch };
