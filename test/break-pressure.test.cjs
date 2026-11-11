'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const {createPressureGuard}=require('../src/break-pressure.cjs');
const {compileOverlayEngine}=require('../scripts/build-overlay.cjs');
const {replaceOnce}=require('../research/rework/legacy-lab.cjs');

test('pressure is a sustained time fraction, not note sampling or a single peak',()=>{
 const rating={query:[0,100,200,900,1000,2000],demand:[20,20,1,1,1,999]};
 const g=createPressureGuard(rating,{quantile:.5,fraction:.5});
 assert.equal(g.reference,1);assert.equal(g.measure(0,1000).highFraction,1);
 const peak=createPressureGuard(rating,{quantile:.95,fraction:.5});
 assert.equal(peak.reference,20);assert.equal(peak.measure(0,1000).protected,false);assert.equal(peak.measure(0,1000).highMs,200);
});
test('stale difficulty across long silence or after the last head is not pressure evidence',()=>{
 const g=createPressureGuard({query:[0,100,10000,10100],demand:[10,100,10,100]},{quantile:.5,fraction:.5});
 assert.equal(g.measure(100,10000).observedMs,0);assert.equal(g.measure(10100,20000).observedMs,0);assert.equal(g.measure(100,10000).protected,false);
 const zero=createPressureGuard({query:[0,100,200],demand:[0,0,0]},{quantile:.5,fraction:.5});assert.equal(zero.measure(0,200).protected,false);
});
test('pressure integration is partition invariant and uses the converted native clock',()=>{
 const q=[0,100,200,300,400],d=[2,10,10,2,2],g=createPressureGuard({query:q,demand:d},{quantile:.75,fraction:.5});
 assert.equal(g.measure(50,350).highMs,g.measure(50,170).highMs+g.measure(170,350).highMs);
 for(const rate of [.75,1.5]){
  const scaled=createPressureGuard({query:q.map(t=>t/rate),demand:d},{quantile:.75,fraction:.5});
  assert(Math.abs(scaled.measure(50/rate,350/rate).highFraction-g.measure(50,350).highFraction)<1e-12);
 }
});
function lab(){
 const s=replaceOnce(compileOverlayEngine({segmentReadout:true}),'  return __toCommonJS(index_exports);','  Object.assign(index_exports,{lab:{classifySegments,MEDIAN_KEYS,VEC_KEYS}});\n  return __toCommonJS(index_exports);');
 const c={module:{exports:{}}};vm.runInNewContext(s,c);return c.module.exports.lab;
}
function fixture(l){
 const zero=Object.fromEntries([...l.MEDIAN_KEYS,...l.VEC_KEYS].map(k=>[k,0]));
 const windows=[12,24,24,24].map(nps=>({...zero,nps,chordMean:2,lnWeightRatio:1,heldCoverage:.7,lockRatio:.7,lnAreaRatio:.7}));
 return{cutDemandRef:1,windows,rawSegs:windows.map((_,i)=>({start:1000+i*5000,end:6000+i*5000,windowRange:[i,i+1],family0:'rc'})),extras:windows.map(()=>({...zero,rowCount:50,segIvLogStd:.5,vibroWidth:7,tailSingleRatio:1,lnHeadChordMean:2}))};
}
test('actual soft Break gate can be vetoed while marked and empty intervals stay Break',()=>{
 const l=lab(),cm=fixture(l);assert.equal(l.classifySegments(cm)[0].tag,'break');
 const opts={_breakPolicy:{confidenceFloor:.5,lowNpsScale:1,protects:()=>true}};
 assert.notEqual(l.classifySegments(cm,opts)[0].tag,'break');
 cm.rawSegs[0].family0='break';assert.equal(l.classifySegments(cm,opts)[0].tag,'break');
 cm.rawSegs[0].family0='rc';cm.extras[0].rowCount=0;assert.equal(l.classifySegments(cm,opts)[0].tag,'break');
});
test('public rated API actually forwards sustained difficulty into the pre-classification Break gate',()=>{
 const api=require('..'),{parseOsu}=require('../src/parser.cjs'),{EXPECTED_RATING}=require('../src/perception.cjs');
 const notes=[];let t=1000;
 for(let i=0;i<160;i++){for(const c of [i%7,(i+2)%7])notes.push(`${(c+.5)*512/7},192,${t},1,0,0:0:0:0:`);t+=[80,120,300,80,120,500][i%6];}
 for(let i=0;i<400;i++){for(const c of [i%7,(i+2)%7,(i+4)%7])notes.push(`${(c+.5)*512/7},192,${t},1,0,0:0:0:0:`);t+=75;}
 const text='[General]\nMode:3\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n'+notes.join('\n'),map=parseOsu(text),query=[...new Set(map.notes.map(n=>n.t))];
 const rating={schemaVersion:1,keyCount:7,speedRate:1,clock:'post-rate-ms',modelIdentity:EXPECTED_RATING,query,demand:query.map(()=>10)},before=structuredClone(rating);
 const unrated=api.analyzeMap(text),rated=api.analyzeMap(text,{ratingEvidence:rating});
 assert.equal(unrated.segments[0].tag,'break');assert.notEqual(rated.segments[0].tag,'break');assert.equal(rated.modelInfo.breakPolicy.protectedParents,1);assert.deepEqual(rating,before);
});
