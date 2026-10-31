'use strict';
// Key-count-aware morphology. These are descriptors, not local truth labels.
const {popcount,lowerBound}=require('../../src/parser.cjs');
const KEYS=['peakShare','peakExcess','repeatConcentration','runAsymmetry','runCoverage','regularity','regularAnchor'];
function describe(rows,K){
 const N=rows.length,counts=Array(K).fill(0),pairs=Array(K).fill(0),longest=Array(K).fill(0),run=Array(K).fill(0);
 const times=Array.from({length:K},()=>[]);
 for(let i=0;i<N;i++)for(let c=0;c<K;c++){
  if(rows[i].mask&(1<<c)){
   counts[c]++;times[c].push(rows[i].t);run[c]++;longest[c]=Math.max(longest[c],run[c]);
   if(i&&(rows[i-1].mask&(1<<c)))pairs[c]++;
  }else run[c]=0;
 }
 const total=counts.reduce((a,b)=>a+b,0),repeat=pairs.reduce((a,b)=>a+b,0);
 const sorted=[...longest].sort((a,b)=>a-b),peakCount=Math.max(...counts);
 const steady=times.map(a=>{
  const iv=a.slice(1).map((t,i)=>t-a[i]),mean=iv.reduce((a,b)=>a+b,0)/Math.max(1,iv.length);
  const cv=mean>0?Math.sqrt(iv.reduce((s,v)=>s+(v-mean)**2,0)/Math.max(1,iv.length))/mean:Infinity;
  return iv.length>=2?1/(1+cv):0;
 });
 const peak=total?Math.max(...counts)/total:0,excess=total?Math.max(0,(peak-1/K)/(1-1/K)):0;
 const concentration=repeat?Math.max(0,(Math.max(...pairs)/repeat-1/K)/(1-1/K)):0;
 const asymmetry=N?(sorted[K-1]-sorted[Math.floor((K-1)/2)])/N:0;
 const regularity=Math.max(...steady.filter((_,c)=>counts[c]===peakCount));
 return [peak,excess,concentration,asymmetry,N?sorted[K-1]/N:0,regularity,excess*regularity];
}
function windows(ctx,ms){
 const start=ctx.map.firstNote,end=ctx.map.lastNote+1,out=[];
 for(let t=start;t<end;t+=ms/2){
  const rows=ctx.rows.slice(lowerBound(ctx.rows,t,r=>r.t),lowerBound(ctx.rows,t+ms,r=>r.t));
  if(rows.length>=3)out.push(describe(rows,ctx.map.keys));
 }
 return out;
}
function pooled(ctx,ms){
 const ws=windows(ctx,ms),out=[];
 for(let j=0;j<KEYS.length;j++){
  const a=ws.map(w=>w[j]).sort((a,b)=>a-b),mean=a.reduce((s,v)=>s+v,0)/Math.max(1,a.length);
  out.push(mean,Math.sqrt(a.reduce((s,v)=>s+(v-mean)**2,0)/Math.max(1,a.length)),a[Math.floor(.9*(a.length-1))]||0);
 }
 return out;
}
module.exports={KEYS,describe,windows,pooled};
