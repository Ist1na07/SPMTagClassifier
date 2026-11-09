'use strict';
// Export the selected finite gate, not coefficients from an OUTER winner.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const root=process.argv.includes('--root')?process.argv[process.argv.indexOf('--root')+1]:'research/overlay-quality/overlay-impression-fixed-post-r1';
const source=JSON.parse(fs.readFileSync(path.join(root,'final-inner-selection.json'),'utf8'));
assert.equal(source.usesOuterTruth,false);
const selected=source.selections.experience_fixed_post;
assert.ok(selected,'Expected the original-aggregation product control');
assert.deepEqual(selected.policy,{dominantShare:.52,mixShare:.07,secondRel:.75});
const {ratio,rowRateMax}=selected.config;
assert.ok(Number.isFinite(ratio)&&ratio>=0&&ratio<=1&&Number.isFinite(rowRateMax)&&rowRateMax>=0);
const value={schemaVersion:1,gate:{ratio,rowRateMax},aggregation:selected.policy,
  selectionSource:path.posix.join(root.replaceAll('\\','/'),'final-inner-selection.json'),
  selectionMeaning:'Mean original6x3 INNER tables, source-TRAIN Speed queries only; never an OUTER model winner.',
  coefficients:'Original archived expert coefficients unchanged; only the regular-small-chord rest disjunct changes.',
  exposure:'Historical expert training exposure unknown; repeatedly studied data, not new blind validation.'};
fs.writeFileSync('src/models/overlay7k.json',JSON.stringify(value,null,2)+'\n');
console.log('Exported original-aggregation 7K overlay gate',value.gate);
