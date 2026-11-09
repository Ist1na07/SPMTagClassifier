'use strict';
// Called only by the regular-small-chord branch. Sparse, explicit-gap and
// other archived Break tests retain their original conditions.
function regularityRestSupported(nps, referenceNps, rowCount, durationMs, relativeNps, rowRateMax) {
  return nps < referenceNps * relativeNps || rowCount * 1000 / durationMs < rowRateMax;
}
module.exports = { regularityRestSupported };
