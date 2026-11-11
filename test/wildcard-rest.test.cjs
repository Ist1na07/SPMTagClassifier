'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const {compileOverlayEngine}=require('../scripts/build-overlay.cjs');
const {replaceOnce}=require('../research/rework/legacy-lab.cjs');
function lab(repaired){
 const source=replaceOnce(compileOverlayEngine({segmentReadout:repaired}),'  return __toCommonJS(index_exports);',
  '  Object.assign(index_exports,{testOnly:{classifySegments,MODEL,MEDIAN_KEYS,VEC_KEYS}});\n  return __toCommonJS(index_exports);');
 const c={module:{exports:{}}};vm.runInNewContext(source,c);return c.module.exports.testOnly;
}
function fixture(l,{ln=true,nps=6,physicalBreak=false,empty=false}={}){
 for(const family of ['rc','ln']){
  const m=l.MODEL[family],K=m.classes.length,D=m.mean.length;m.mean.fill(0);m.std.fill(1);m.W.fill(0);m.b=m.classes.map(c=>c===('ln'===family?'ln wildcard':'wildcard')?10:0);assert.equal(m.W.length,K*D);
 }
 const zero=Object.fromEntries([...l.MEDIAN_KEYS,...l.VEC_KEYS].map(k=>[k,0]));
 const windows=[nps,12,12,12].map(n=>({...zero,nps:n,chordMean:1,lnWeightRatio:ln?1:0,heldCoverage:ln?.3:0,lockRatio:ln?.6:0}));
 const rawSegs=windows.map((_,i)=>({start:1000+i*5000,end:6000+i*5000,windowRange:[i,i+1],family0:i===0&&physicalBreak?'break':'rc'}));
 const extras=windows.map((_,i)=>({...zero,rowCount:i===0&&empty?0:30,vibroWidth:7,segIvLogStd:0,segRhythmChangeRate:0,tailSingleRatio:1,lnHeadChordMean:1}));
 return {windows,rawSegs,extras,cutDemandRef:1};
}
test('low-fit LN wildcard remains active rather than being relabelled as Break',()=>{
 const old=lab(false),now=lab(true);assert.equal(old.classifySegments(fixture(old),{scorer:'model'})[0].tag,'break');
 const first=now.classifySegments(fixture(now),{scorer:'model'})[0];assert.equal(first.tag,'ln wildcard');assert.equal(first.family,'ln');assert.equal(first.confidence,.156);
});
test('physical sparse/empty/explicit rests and RC ambiguity retain the original handling',()=>{
 for(const options of [{nps:1},{physicalBreak:true},{empty:true},{ln:false}]){
  const l=lab(true);assert.equal(l.classifySegments(fixture(l,options),{scorer:'model'})[0].tag,'break');
 }
});
