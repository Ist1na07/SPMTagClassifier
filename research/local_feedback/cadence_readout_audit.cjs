'use strict';
// Rendering-risk audit of a finer readout. Both arms see identical fixed tags
// and identical affinities; do not mistake a busier timeline for improvement.
const fs=require('node:fs');
const {createProbe}=require('./activity_overlay_probe.cjs');
const {createReadoutProbe}=require('./cadence_readout.cjs');
function split(_model,_map,raw,_candidates,classes){
  const frames=[];raw.forEach((s,parent)=>{for(let start=s.start;start<s.end;start+=500)frames.push({...s,start,end:Math.min(s.end,start+500),parent});});
  const support=frames.map(s=>{const scores=s.scores??{[s.tag]:1},div=Math.max(1,...Object.values(scores));return classes.map(c=>s.family==='break'?0:Math.min(.999,Math.max(0,scores[c]??0)/div));});
  return {frames,support};
}
const a=createProbe({},split),b=createReadoutProbe({},split),rows=fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n').map(JSON.parse).slice(0,349),out=[];
for(const row of rows){
  const text=fs.readFileSync(row.source,'utf8'),old=a(text),next=b(text),count=r=>({segments:r.segments.length,subsecond:r.segments.filter(s=>s.end-s.start<1000).length,oneBpmTransitions:r.segments.filter((s,j)=>j&&s.tag===r.segments[j-1].tag&&Math.abs(s.equivBpm-r.segments[j-1].equivBpm)===1).length});
  out.push({uid:row.uid,source:row.source,before:count(old),after:count(next),segments:next.segments.map(({start,end,tag,equivBpm})=>({start,end,tag,equivBpm}))});
  if(out.length%50===0)console.log('cadence readout audit',out.length);
}
const sums=arm=>Object.fromEntries(Object.keys(out[0][arm]).map(k=>[k,out.reduce((s,r)=>s+r[arm][k],0)]));
const receipt={scope:'Fixed old local primary tags and supports, first349 original labelled maps, Actions fallback; actual compiled fine-frame consumer only. Display structure is diagnostic, not type accuracy.',before:sums('before'),after:sums('after'),worst:[...out].sort((a,b)=>(b.after.segments-b.before.segments)-(a.after.segments-a.before.segments)).slice(0,10),cases:out};
fs.writeFileSync('research/local_feedback/results/cadence-readout-audit.json',JSON.stringify(receipt,null,2));console.log(JSON.stringify({before:receipt.before,after:receipt.after,worst:receipt.worst.slice(0,3).map(r=>({uid:r.uid,before:r.before,after:r.after}))}));
