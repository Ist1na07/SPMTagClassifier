'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const fs=require('node:fs');
const {loadLab,context,inspect}=require('./legacy-lab.cjs');
const {records7,scoreIntervals}=require('./data.cjs');
const patches=require('./patches.cjs');
const legacy=require('../../src/legacy/tag-engine.cjs');
function chart(notes,timing='0,500,4,2,0,100,1,0'){
 return 'osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n'+timing+'\n[HitObjects]\n'+notes.map(n=>`${Math.floor((n[1]+.5)*512/7)},192,${n[0]},${n.length>2?128:1},0,${n.length>2?n[2]+':0:0:0:0':'0:0:0:0:'}`).join('\n');
}
const nearly=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('instrumentation and reconstruction preserve complete original output on all six annotated charts',()=>{
 const api=loadLab();
 for(const r of records7().filter(r=>r.dataset==='manual')){
  const text=fs.readFileSync(r.path,'utf8'),expected=legacy.analyzeMap(text),actual=api.analyzeMap(text),view=inspect(api,text);
  assert.deepEqual(actual,expected,r.uid);assert.deepEqual(view.smoothed,expected.segments,r.uid+' reconstructed pipeline');
  assert.deepEqual(inspect(api,text,{intervals:view.segments}).classified,view.classified,r.uid+' frozen-boundary control');
  assert.equal(view.featureKeys.length,56);
  assert.ok(view.segments.every(s=>s.vector.every(Number.isFinite)));
  assert.equal(scoreIntervals(r.annotations,view.smoothed).uncoveredMs,scoreIntervals(r.annotations,expected.segments).uncoveredMs);
 }
});
test('LN area counts one clipped interval once; overlaps on one lane form a union',()=>{
 const text=chart([[0,0,900],[100,0,700],[550,2],[1300,1]]);
 const original=loadLab(),fixed=loadLab({patches:[patches.area]});
 const a=original.lab.computeWindowFeatures(context(original.lab,text).ctx,500,1000);
 const b=fixed.lab.computeWindowFeatures(context(fixed.lab,text).ctx,500,1000);
 nearly(a.lnAreaRatio,1200/(500*7));nearly(b.lnAreaRatio,400/(500*7));
});
test('head adjacency is invariant to an intervening release-only row',()=>{
 const text=chart([[0,0,300],[500,0],[1000,1]]);
 const before=loadLab(),after=loadLab({patches:[patches.adjacency]});
 nearly(before.lab.computeWindowFeatures(context(before.lab,text).ctx,0,700).jackRowRatio,0);
 nearly(after.lab.computeWindowFeatures(context(after.lab,text).ctx,0,700).jackRowRatio,1);
});
test('a same-lane tail/head junction leaves the new hold active',()=>{
 const text=chart([[0,0,500],[500,0,900],[700,2],[1100,1]]);
 for(const [ps,expected] of [[[],0],[[patches.heldOrder],1]]){
  const api=loadLab({patches:ps}),c=context(api.lab,text);
  assert.equal(c.ctx.held.heldMask[c.rows.findIndex(r=>r.t===700)],expected);
 }
});
test('beat phase resets at an uninherited timing point',()=>{
 const text=chart([[1100,0],[1350,1],[1600,2],[1850,3],[2100,4]],'0,500,4,2,0,100,1,0\n1100,1000,4,2,0,100,1,0');
 for(const [ps,expected] of [[[],1],[[patches.phase],0]]){
  const api=loadLab({patches:ps});nearly(api.lab.computeWindowFeatures(context(api.lab,text).ctx,1100,2100).offBeatRatio,expected);
 }
});
test('release isolation measures nearest head, not the release itself or simultaneous event ordering',()=>{
 const api=loadLab({patches:[patches.releaseIsolation]});
 for(const [notes,expected] of [[[[0,0,500],[800,2]],1],[[[0,0,500],[500,2],[800,1]],0]]){
  nearly(api.lab.computeWindowFeatures(context(api.lab,chart(notes)).ctx,0,700).tailIndependence,expected);
 }
});
test('sustaining LN area and releases are not erased when there is no head in window',()=>{
 const api=loadLab({patches:[patches.area,patches.sustaining]});
 const f=api.lab.computeWindowFeatures(context(api.lab,chart([[0,0,900],[1500,2]])).ctx,500,1000);
 nearly(f.lnAreaRatio,400/(500*7));nearly(f.tailRate,2);assert.equal(f.notes,0);
 assert.ok(Object.values(f).every(Number.isFinite));
});
test('normalization quantile span is measured in the transformed signal domain',()=>{
 const api=loadLab({patches:[patches.transformedScale]});
 const f=api.lab.computeWindowFeatures(context(api.lab,chart([[0,0],[500,1],[1000,2],[1500,3]])).ctx,0,1000);
 const windows=Array.from({length:9},(_,i)=>({...f,nps:(i+1)*10}));
 const norm=api.lab.normalizeWindows(windows),j=norm.featureNames.indexOf('nps');
 nearly(norm.values[7][j]-norm.values[0][j],1);
});
test('uniform confidence rescaling must not turn a 60% dominant pattern into Mix',()=>{
 const original=loadLab(),fixed=loadLab({patches:[patches.weightedShares]});
 const segments=[['chordjack',6000],['dense chordstream',2000],['tech',2000]].map(([tag,d])=>({tag,start:0,end:d,family:'rc',confidence:1}));
 assert.deepEqual(original.lab.aggregateOverall(segments).tags,['chordjack']);
 const weaker=segments.map(s=>({...s,confidence:.4}));
 assert.deepEqual(original.lab.aggregateOverall(weaker).tags,['rc mix']);
 assert.deepEqual(fixed.lab.aggregateOverall(weaker).tags,['chordjack']);
});
module.exports={chart};
