'use strict';
const fs=require('node:fs');
const {records7}=require('./data.cjs'),{mirrorText}=require('./mirror.cjs');
const original=require('../../src/legacy/tag-engine.cjs');
const {loadLab}=require('./legacy-lab.cjs'),patches=require('./patches.cjs');
const useFix=process.argv.includes('--neutral-hands');
const model=useFix?loadLab({patches:[patches.neutralHands]}):original;
const cached=new Map(JSON.parse(fs.readFileSync('.cache/rework-legacy-full.json','utf8')).map(r=>[r.uid,r]));
const rows=[];
for(const r of records7()){
 const text=fs.readFileSync(r.path,'utf8'),base=useFix?model.analyzeMap(text):cached.get(r.uid),m=model.analyzeMap(mirrorText(text));
 const duration=base.segments.reduce((s,x)=>s+x.end-x.start,0);let disagreement=0;
 for(const a of base.segments)for(const b of m.segments){
  const d=Math.max(0,Math.min(a.end,b.end)-Math.max(a.start,b.start));if(a.tag!==b.tag)disagreement+=d;
 }
 const tagChanged=JSON.stringify([...base.overall.tags].sort())!==JSON.stringify([...m.overall.tags].sort());
 rows.push({uid:r.uid,overallChanged:tagChanged,original:base.overall.tags,mirrored:m.overall.tags,
  segmentDisagreementMs:disagreement,durationMs:duration,
  boundariesChanged:JSON.stringify(base.segments.map(s=>[s.start,s.end]))!==JSON.stringify(m.segments.map(s=>[s.start,s.end]))});
}
const output={scope:'physical-column horizontal mirror; fixed original v1.1.0 coefficients, full native segmentation/aggregation',
 records:rows.length,overallChanges:rows.filter(r=>r.overallChanged).length,
 segmentChangedRecords:rows.filter(r=>r.segmentDisagreementMs>0).length,
 durationDisagreement:rows.reduce((s,r)=>s+r.segmentDisagreementMs,0)/rows.reduce((s,r)=>s+r.durationMs,0),rows};
if(useFix)output.scope='Neutral/tied hand rows repaired; full native segmentation/aggregation; original coefficients otherwise unchanged';
fs.writeFileSync('research/rework/results/mirror-audit'+(useFix?'-neutral-hands':'')+'.json',JSON.stringify(output,null,2)+'\n');
console.log(Object.fromEntries(Object.entries(output).filter(([k])=>k!=='rows')));
