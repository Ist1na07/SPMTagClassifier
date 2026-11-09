'use strict';
// Reuse the established static-LN compilation route. All archive patching and
// filesystem work happen here, never inside the generated browser consumer.
const fs = require('node:fs'), path = require('node:path');
const lab = require('../rework/legacy-lab.cjs');
const { bundle } = require('../../scripts/bundle.cjs');
const { physicalPatch, displayPatch } = require('./bilateral-runtime.cjs');
function compileBilateralEngine(variant) {
  if (!['original', 'physical'].includes(variant)) throw Error('Invalid compiled variant');
  let archive = fs.readFileSync(path.resolve(__dirname, '../../src/legacy/tag-engine.cjs'), 'utf8').replace(/\r\n/g, '\n');
  archive = displayPatch.apply(archive, lab.replaceOnce);
  if (variant === 'physical') archive = physicalPatch.apply(archive, lab.replaceOnce);
  archive = lab.replaceOnce(archive, '  return __toCommonJS(index_exports);',
    `  Object.assign(index_exports,{lab:{${lab.EXPOSE.join(',')}}});\n  return __toCommonJS(index_exports);`);
  const ids = ['bilateral-fixed-display-smoothing', ...(variant === 'physical' ? ['bilateral-physical-partition'] : [])];
  const factory = `function loadLab({patches=[],config={}}={}){
    const expected=${JSON.stringify(ids)};
    if(patches.length!==expected.length||patches.some((p,i)=>p.id!==expected[i]))throw Error('Compiled partition signal mismatch');
    const module={exports:{}};(function(module,exports){\n${archive}\n})(module,module.exports);
    const api=module.exports;Object.assign(api.lab.CONFIG,config);return api;
  }
  module.exports={loadLab};`;
  return bundle(path.join(__dirname, 'bilateral-runtime.cjs'), {
    overrides: new Map([[path.resolve(__dirname, '../rework/legacy-lab.cjs'), factory]]) });
}
module.exports = { compileBilateralEngine };
