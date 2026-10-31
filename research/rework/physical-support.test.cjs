'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {physicalSupport}=require('./physical-support.cjs'),{audit}=require('./coarse_sanity.cjs');
test('template support distinguishes rotation, timing variation, and actual LN-only activity',()=>{
 const row={notes:48,contextStart:0,contextEnd:2000,metrics:{rowRate:8,occupancy:0,'release:tailRatePerKey':0,lnHeadFraction:0,chordChange:0,overlap:1,ioiCv:0,rowBpm:120}};
 assert.equal(physicalSupport(row,'both').tag,'jack');assert.equal(physicalSupport(row,'ln'),null);
 for(const change of[{chordChange:.5},{ioiCv:.05},{lnHeadFraction:.1},{occupancy:.1}])assert.equal(physicalSupport({...row,metrics:{...row.metrics,...change}},'both'),null);
 assert.equal(physicalSupport({...row,notes:0,metrics:{...row.metrics,rowRate:0,occupancy:1/6}},'both').tag,'ln');
 assert.equal(physicalSupport({...row,notes:1,metrics:{...row.metrics,rowRate:.5,lnHeadFraction:1,occupancy:1/6}},'both').tag,'ln');
 assert.equal(physicalSupport({...row,notes:0,metrics:{...row.metrics,rowRate:0}},'both'),null);
 assert.throws(()=>physicalSupport({...row,contextStart:undefined},'both'),/full finite/);
 assert.equal(physicalSupport({...row,contextStart:undefined},'none'),null,'existing raw usage has no new precondition');
});
test('actual native-key synthetic mechanisms are corrected without changing the default fixture model',()=>{
 const p=JSON.parse(fs.readFileSync('research/rework/results/coarse-4k-baseline-r1/weak4_only/0/outer-model.json','utf8'));
 const result=audit(p,{physicalSupport:'both'});assert.equal(result.tests,99);assert.equal(result.counterexamples.length,0);
 // The original default path deliberately remains unchanged until retuning.
 assert.equal(audit(p).rows.find(r=>r.id==='K4-repeat-0123-120').predicted,'speed');
});
