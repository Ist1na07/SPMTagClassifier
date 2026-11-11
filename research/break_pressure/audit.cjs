'use strict';
const fs=require('node:fs'),zlib=require('node:zlib');
const {compileOverlayEngine}=require('../../scripts/build-overlay.cjs');
const {replaceOnce}=require('../rework/legacy-lab.cjs');
function classifier({low=true,late=true,marked=true}={}){
 let s=compileOverlayEngine({segmentReadout:true,breakPressure:false});
 if(!low)s=replaceOnce(s,'if (lowStakes) {','if (false && lowStakes) {');
 if(!late)s=replaceOnce(s,'if (easyAmbiguous || shortSparse) {','if (false && (easyAmbiguous || shortSparse)) {');
 if(!marked)s=replaceOnce(s,'if (s.family0 === "break" || cm.extras[i].rowCount === 0) {','if (cm.extras[i].rowCount === 0) {');
 s=replaceOnce(s,'  return __toCommonJS(index_exports);','  index_exports.classifySegments=classifySegments;\n  return __toCommonJS(index_exports);');
 const m={exports:{}};new Function('module','exports',s)(m,m.exports);return m.exports.classifySegments;
}
function dist(query,demand,start,end){
 const v=[];let area=0,duration=0;
 for(let i=0;i<query.length;i++){
  const a=Math.max(start,query[i]),b=Math.min(end,i+1<query.length?query[i+1]:end),dt=b-a;
  if(dt>0){v.push([demand[i],dt]);area+=dt*demand[i];duration+=dt;}
 }
 v.sort((a,b)=>a[0]-b[0]);const q=p=>{let s=0;for(const[x,w]of v){s+=w;if(s>=duration*p)return x;}return 0;};
 return{mean:area/Math.max(1,duration),p50:q(.5),p75:q(.75),p90:q(.9),max:v.at(-1)?.[0]??0};
}
if(require.main===module){
 const rows=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r2.jsonl.gz')).toString().trim().split('\n').map(JSON.parse);
 const resources=new Map(fs.readFileSync('.cache/perception-final-resource.jsonl','utf8').trim().split('\n').map(s=>{const r=JSON.parse(s);return[r.uid,r];}));
 const full=classifier(),noLow=classifier({low:false}),noLate=classifier({late:false}),all=classifier({low:false,late:false,marked:false}),out=[];const reasons={};
 for(const r of rows){
  const before=full(r.cm),after=all(r.cm),nl=noLow(r.cm),nt=noLate(r.cm),resource=resources.get(r.uid),q=resource.ratings.detail.query,d=resource.ratings.detail.demand;
  const map=dist(q,d,q[0],Math.max(...r.cm.rawSegs.map(s=>s.end)));const candidates=[];
  for(let j=0;j<before.length;j++){
   if(before[j].family!=='break')continue;const p=r.parents[j],s=before[j];
   const reason=!p.extra.rowCount?'empty':r.cm.rawSegs[j].family0==='break'?'marked':nl[j].family!=='break'?'lowStakes':nt[j].family!=='break'?'late':'multiple';reasons[reason]=(reasons[reason]??0)+1;
   const rating=dist(q,d,s.start,s.end);candidates.push({parent:j,start:s.start,end:s.end,reason,nps:p.f.nps,lnWeight:p.f.lnWeightRatio,alternative:after[j].tag,alternativeConfidence:after[j].confidence,rating,relativeMean:rating.mean/Math.max(1e-12,map.p50)});
  }
  out.push({uid:r.uid,source:r.source,map,candidates});
 }
 fs.mkdirSync('research/break_pressure/results',{recursive:true});fs.writeFileSync('research/break_pressure/results/audit.json',JSON.stringify({scope:'Current default; cached physical parents and existing Detailed demand. Time-weighted right-continuous raw D, not section stars. Observational flags, not new truth.',reasons,records:out},null,2)+'\n');
 console.log(reasons);console.log(out.flatMap(r=>r.candidates.filter(s=>s.reason!=='empty').map(s=>({uid:r.uid,...s}))).sort((a,b)=>b.rating.mean-a.rating.mean).slice(0,20));
}
module.exports={classifier,dist};
