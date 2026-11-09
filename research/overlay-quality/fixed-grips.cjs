'use strict';
const { lowerBound } = require('../../src/parser.cjs');
function alternatingFixedGrips(ctx, start, end) {
  if (ctx.map.keys !== 7) return null;
  // Any active hold or the flexible center prevents this strict RC proof.
  for (const intervals of ctx.holds) {
    const i = lowerBound(intervals, end, h => h[0]) - 1;
    if (i >= 0 && intervals[i][1] > start) return null;
  }
  const lo = lowerBound(ctx.rows, start, r => r.t), hi = lowerBound(ctx.rows, end, r => r.t);
  if (hi - lo < 4) return null;
  const masks = [0, 0]; let previous = -1;
  for (let i = lo; i < hi; i++) {
    const mask = ctx.rows[i].mask;
    if (mask & 0b0001000) return null;
    const left = mask & 0b0000111, right = mask & 0b1110000;
    if ((!left && !right) || (left && right)) return null;
    const hand = left ? 0 : 1;
    if (hand === previous || (masks[hand] && masks[hand] !== mask)) return null;
    masks[hand] = mask; previous = hand;
  }
  return { rowCount: hi - lo, leftMask: masks[0], rightMask: masks[1] };
}
module.exports = { alternatingFixedGrips };
