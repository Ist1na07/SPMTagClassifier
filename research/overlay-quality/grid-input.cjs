'use strict';
// Grid phase error must be relative to the candidate period to distinguish
// harmonics. The physical period floor follows the row-merging resolution.
const GRID = [false, true].flatMap(physicalFloor => [null, .1, .15, .2, .25].map(relativeTolerance => ({ physicalFloor, relativeTolerance })));
function gridPatch({ physicalFloor, relativeTolerance }) {
  if (typeof physicalFloor !== 'boolean' || !(relativeTolerance === null || [ .1, .15, .2, .25 ].includes(relativeTolerance)))
    throw Error('Unknown declared cadence-input configuration');
  return { id: 'physical-grid-input', apply(source, replace) {
    if (relativeTolerance !== null) source = replace(source, '      const cov = gridCoverage(arr, s, tol);',
      `      const cov = gridCoverage(arr, s, Math.min(tol, s * ${relativeTolerance}));`);
    if (physicalFloor) {
      source = replace(source, '      if (!(s >= CONFIG.gridMinSpacing && s <= CONFIG.gridMaxSpacing)) return;',
        '      if (!(s >= CONFIG.rowMergeMs && s <= CONFIG.gridMaxSpacing)) return;');
      source = replace(source, '      for (let s = CONFIG.gridMinSpacing; s <= 800; s = s * 1.12) evaluate(s);',
        '      for (let s = CONFIG.rowMergeMs; s <= 800; s = s * 1.12) evaluate(s);');
    }
    return source;
  } };
}
module.exports = { GRID, gridPatch };
