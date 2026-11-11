'use strict';
// Right-continuous raw demand on the native head-row clock. Never extend a
// sample through a physical long gap or beyond the final observed head.
function distribution(query,demand,gapMs){
 const spans=[];
 for(let i=0;i+1<query.length;i++){
  const dt=query[i+1]-query[i];
  if(dt>0&&dt<gapMs)spans.push({start:query[i],end:query[i+1],value:demand[i]});
 }
 return spans;
}
function weightedQuantile(spans,q){
 const sorted=[...spans].sort((a,b)=>a.value-b.value),total=sorted.reduce((s,x)=>s+x.end-x.start,0);let seen=0;
 for(const s of sorted){seen+=s.end-s.start;if(seen>=q*total)return s.value;}
 return 0;
}
function createPressureGuard(rating,{quantile,fraction},gapMs=2500){
 if(!Number.isFinite(quantile)||quantile<0||quantile>1||!Number.isFinite(fraction)||fraction<=0||fraction>1||!Number.isFinite(gapMs)||gapMs<=0)throw Error('InvalidBreakPressurePolicy');
 const spans=distribution(rating.query,rating.demand,gapMs),reference=weightedQuantile(spans,quantile);
 const starts=spans.map(s=>s.start),prefix=[0],covered=[0];
 for(const s of spans){prefix.push(prefix.at(-1)+(s.value>=reference&&reference>0?s.end-s.start:0));covered.push(covered.at(-1)+s.end-s.start);}
 function integral(t,values,high){
  let lo=0,hi=starts.length;while(lo<hi){const m=(lo+hi)>>1;if(starts[m]<t)lo=m+1;else hi=m;}
  if(!lo)return 0;const i=lo-1,s=spans[i],dt=Math.min(s.end,t)-s.start;
  return values[i]+Math.max(0,dt)*(!high||s.value>=reference&&reference>0?1:0);
 }
 function measure(start,end){
  const highMs=integral(end,prefix,true)-integral(start,prefix,true),observedMs=integral(end,covered,false)-integral(start,covered,false);
  return{reference,highMs,observedMs,highFraction:highMs/Math.max(1,end-start),protected:reference>0&&highMs>0&&highMs>=(end-start)*fraction};
 }
 return{measure,reference};
}
module.exports={distribution,weightedQuantile,createPressureGuard};
