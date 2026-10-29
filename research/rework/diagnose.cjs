'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadLab,inspect}=require('./legacy-lab.cjs');
const {ROOT,records7,scoreIntervals}=require('./data.cjs');
const patches=require('./patches.cjs'),legacy=require('../../src/legacy/tag-engine.cjs');
const out=path.join(ROOT,'research/rework/results');fs.mkdirSync(out,{recursive:true});
const save=(file,data)=>fs.writeFileSync(path.join(out,file),JSON.stringify(data,null,2)+'\n');
const records=records7(),manual=records.filter(r=>r.dataset==='manual');
const conditions=[['original',[]],...['area','adjacency','heldOrder','phase','sustaining','releaseIsolation'].map(k=>[patches[k].id,[patches[k]]]),
 ['logical',patches.logical],['logical_release',patches.logical.concat(patches.releaseIsolation)]];
const cached=new Map(),numeric=[],parity=[];
const api=loadLab();
// Check the actual archived entry point, not only a reimplementation of it.
for(const r of records){
 const text=fs.readFileSync(r.path,'utf8'),baseline=legacy.analyzeMap(text),exposed=api.analyzeMap(text);
 assert.deepEqual(exposed,baseline,r.uid);parity.push(r.uid);
 if(r.dataset==='manual')cached.set(r.uid,{text,baseline,view:inspect(api,text)});
}
save('instrumentation-parity.json',{records:parity.length,uids:parity,comparison:'complete original analyzeMap return, deepStrictEqual'});
function boundaryOracle(ann,segments){
 let good=0,total=ann.filter(a=>a.tags.length).reduce((s,a)=>s+a.end-a.start,0);
 for(const s of segments){
  const counts={};
  for(const a of ann){if(!a.tags.length)continue;const d=Math.max(0,Math.min(a.end,s.end)-Math.max(a.start,s.start));
   for(const t of a.tags)counts[t]=(counts[t]||0)+d;}
  good+=Math.max(0,...Object.values(counts));
 }
 return good/Math.max(total,1);
}
function compact(m){const {details,...rest}=m;return rest;}
for(const [id,ps] of conditions){
 const model=loadLab({patches:ps});
 for(const r of manual){
  const {text,view:base}=cached.get(r.uid),view=id==='original'?base:inspect(model,text);
  // Fixed original boundaries isolate representation/head changes from segmentation changes.
  const fixed=id==='original'?view:inspect(model,text,{intervals:base.segments});
  const oracle=inspect(model,text,{intervals:r.annotations});
  for(const [partition,v] of [['automatic',view],['fixed_original',fixed],['human_boundary',oracle]]){
   const m=scoreIntervals(r.annotations,v.smoothed),internal=scoreIntervals(r.annotations,v.smoothed,{trimMs:1000});
   numeric.push({condition:id,chart:r.id,partition,...compact(m),internal1s:compact(internal),segments:v.smoothed.length,
    boundaryOracle:boundaryOracle(r.annotations,v.smoothed)});
  }
  if(id==='original'){
   save(r.id+'-trace.json',{annotations:r.annotations,segments:view.segments.map((s,i)=>({...s,prediction:view.classified[i]})),
    humanBoundary:oracle.segments.map((s,i)=>({...s,prediction:oracle.classified[i]}))});
  }
 }
}
const prior=JSON.parse(fs.readFileSync(path.join(ROOT,'research/results/segment-test-predictions.json'),'utf8')).filter(r=>r.variant==='full');
for(const r of manual){
 const {view}=cached.get(r.uid),p=prior.filter(p=>p.uid===r.uid);
 const m=scoreIntervals(r.annotations,p.map(p=>({...p,tag:p.predicted})));
 numeric.push({condition:'previous_modern_oof',chart:r.id,partition:'automatic',...compact(m),
  internal1s:compact(scoreIntervals(r.annotations,p.map(p=>({...p,tag:p.predicted})),{trimMs:1000}))});
 const start=view.map.firstNote,end=Math.max(...r.annotations.map(a=>a.end)),grid=[];
 for(let t=start;t<end;t+=2000)grid.push({start:t,end:Math.min(t+2000,end)});
 const quantized=grid.map(s=>{
  const sums={};for(const p of view.smoothed){const d=Math.max(0,Math.min(s.end,p.end)-Math.max(s.start,p.start));sums[p.tag]=(sums[p.tag]||0)+d;}
  return {...s,tag:Object.entries(sums).sort((a,b)=>b[1]-a[1])[0]?.[0]};
 });
 numeric.push({condition:'original_quantized_2s',chart:r.id,partition:'automatic',...compact(scoreIntervals(r.annotations,quantized)),
  boundaryOracle:boundaryOracle(r.annotations,grid)});
}
const summaries=[];
for(const condition of [...new Set(numeric.map(r=>r.condition))])for(const partition of [...new Set(numeric.filter(r=>r.condition===condition).map(r=>r.partition))]){
 const rows=numeric.filter(r=>r.condition===condition&&r.partition===partition);
 const correct=rows.reduce((s,r)=>s+r.correctMs,0),total=rows.reduce((s,r)=>s+r.labelledMs,0);
 const cm={};for(const r of rows)for(const [k,v]of Object.entries(r.confusion))cm[k]=(cm[k]||0)+v;
 summaries.push({condition,partition,durationAccuracy:correct/total,chartMeanAccuracy:rows.reduce((s,r)=>s+r.accuracy,0)/rows.length,
  uncoveredMs:rows.reduce((s,r)=>s+r.uncoveredMs,0),largestErrors:Object.entries(cm).sort((a,b)=>b[1]-a[1]).slice(0,15)});
}
save('diagnosis.json',{scope:'fixed pretrained coefficients; no refit or parameter selection; human-boundary conditions are counterfactual diagnostics',rows:numeric,summaries});
for(const row of summaries)console.log(row.condition,row.partition,row.durationAccuracy.toFixed(5),row.chartMeanAccuracy.toFixed(5));
