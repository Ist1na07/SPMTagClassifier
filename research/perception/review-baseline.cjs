'use strict';
// Read-only deterministic replay of the ORIGINAL v1.1.0 research baseline.
const fs=require('node:fs'),api=require('../../src/index.cjs');
const meta=JSON.parse(fs.readFileSync('.cache/perception-r1/manifest.json','utf8'));
const ids=new Set(meta.mainUids),rows=[];
for(const r of meta.records)if(ids.has(r.uid)){
 const result=api.legacy.analyzeMap(fs.readFileSync(r.source,'utf8'));
 rows.push({uid:r.uid,predicted:result.overall.tags});
}
if(rows.length!==349)throw Error('Expected original whole-chart inventory');
fs.writeFileSync('.cache/perception-original-v110.json',JSON.stringify(rows)+'\n');
console.log('Replayed immutable original v1.1.0:',rows.length);
