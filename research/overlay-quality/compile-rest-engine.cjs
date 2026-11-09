'use strict';
// Build-time only. The generated engine is static archive code, not an eval
// adapter. Keep the archived public/cache entry points untouched on disk.
const fs = require('node:fs'), path = require('node:path');
const { replaceOnce } = require('../rework/legacy-lab.cjs');
const { actionRatePatch, CLAUSE } = require('./break-regularity.cjs');
const { regularityRestSupported } = require('../../src/rest-policy.cjs');
function compileRestEngine(config, post) {
  for (const k of ['dominantShare', 'mixShare', 'secondRel'])
    if (!Number.isFinite(post[k]) || post[k] < 0 || post[k] > 1) throw Error('Invalid post parameter: ' + k);
  let source = fs.readFileSync(path.resolve(__dirname, '../../src/legacy/tag-engine.cjs'), 'utf8').replace(/\r\n/g, '\n');
  // Validate through the same declared patch contract, then compile the
  // shared, unit-tested predicate rather than maintaining another formula.
  actionRatePatch(config);
  source = replaceOnce(source, '  function classifySegments(cm, opts = {}) {',
    '  '+regularityRestSupported.toString()+'\n  function classifySegments(cm, opts = {}) {');
  source = replaceOnce(source, CLAUSE,
    ` || regularityRestSupported(f.nps, mapNps, ex.rowCount, s.end - s.start, ${config.ratio}, ${config.rowRateMax}) && f.chordMean < 1.9 && ex.vibroFraction > 0.3 && ex.segRhythmChangeRate < 0.1`);
  source = replaceOnce(source, '    const overall = aggregateOverall(segments);',
    '    const overall = aggregateOverall(segments, ' + JSON.stringify(post) + ');');
  return source;
}
module.exports = { compileRestEngine };
