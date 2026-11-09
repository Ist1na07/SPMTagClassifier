'use strict';
const { lowerBound, popcount } = require('../../src/parser.cjs');
const KEYS = ['handSubstitutionFraction', 'handFixedGripTransitionShare'];
function handCuts(ctx, start, end) {
  if (ctx.map.keys !== 7) throw new Error('This bounded RC correction is 7K-only');
  const rows = ctx.rows, lo = lowerBound(rows, start, r => r.t), hi = lowerBound(rows, end, r => r.t);
  // Compare consecutive ACTIVE strikes of each fixed hand. An intervening
  // strike by the other hand must not erase the previous finger assignment.
  // The flexible center key does not establish a fixed left/right substitution.
  const masks = [0b0000111, 0b1110000], previous = [0, 0];
  let substitutions = 0, support = 0, transitions = 0, fixed = 0;
  for (let i = lo; i < hi; i++) for (let hand = 0; hand < 2; hand++) {
    const current = rows[i].mask & masks[hand];
    if (!current) continue;
    const old = previous[hand];
    if (old) {
      substitutions += Math.min(popcount(old & ~current), popcount(current & ~old));
      support += Math.min(popcount(old), popcount(current));
      transitions++; fixed += Number(old === current);
    }
    previous[hand] = current;
  }
  return { values: [support ? substitutions / support : null, transitions ? fixed / transitions : null],
    support, transitions, substitutions, fixedGripTransitions: fixed };
}
module.exports = { KEYS, handCuts };
