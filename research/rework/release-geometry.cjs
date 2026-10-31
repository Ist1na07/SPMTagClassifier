'use strict';
const {lowerBound:lb}=require('../../src/parser.cjs');
const KEYS=['tailRatePerKey','headIsolation','sameFingerRecovery','tailAsynchrony','tailTimingVariability',
 'heldOtherAtTail','sameHandOrder','releaseChordFraction','independentHeldRelease'];
const mean=a=>a.length?a.reduce((s,x)=>s+x,0)/a.length:0;
const sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(x=>(x-m)**2)));};
const ratio=(gap,scale)=>Number.isFinite(gap)?gap/(gap+scale):1;

function releaseContext(ctx){
 // Timing-normalized geometry uses the active quarter-beat. Without timing
 // points use a scale inferred from actual event gaps, not a new fixed-ms gate.
 const times=[...new Set([...ctx.map.notes.map(n=>n.t),...ctx.tails.map(n=>n.t)])].sort((a,b)=>a-b);
 const gaps=times.slice(1).map((t,i)=>t-times[i]).sort((a,b)=>a-b);
 return {...ctx,releaseFallbackScale:gaps[Math.floor((gaps.length-1)/2)]||1};
}
function scaleAt(ctx,t){
 const points=ctx.map.timingPoints;
 const index=Math.max(0,lb(points,t+1e-8,p=>p.t)-1);
 return points.length?points[index].beatLength/4:ctx.releaseFallbackScale;
}
function releaseGeometry(ctx,start,end){
 const K=ctx.map.keys,tails=ctx.tails,heads=ctx.map.notes;
 const lo=lb(tails,start,n=>n.t),hi=lb(tails,end,n=>n.t),n=hi-lo;
 if(!n)return {values:KEYS.map(()=>0),metrics:Object.fromEntries(KEYS.map(k=>[k,0])),tails:0};
 const isolation=[],recovery=[],asynchrony=[],held=[],interaction=[],rhythm=[],width=[];let order=0;
 const hand=c=>K===7&&c===3?-1:c<Math.floor(K/2)?0:1;
 for(let j=lo;j<hi;j++){
  const tail=tails[j],scale=scaleAt(ctx,tail.t),i=lb(heads,tail.t,n=>n.t);
  const before=i>0?tail.t-heads[i-1].t:Infinity,after=i<heads.length?heads[i].t-tail.t:Infinity;
  const independent=ratio(Math.min(before,after),scale);isolation.push(independent);
  const same=ctx.cols[tail.col],next=same[lb(same,tail.t,n=>n.t)];
  recovery.push(ratio(next?next.t-tail.t:Infinity,scale));
  const previous=j?tail.t-tails[j-1].t:Infinity,following=j+1<tails.length?tails[j+1].t-tail.t:Infinity;
  asynchrony.push(ratio(Math.min(previous,following),scale));
  let other=0;
  for(let col=0;col<K;col++){
   if(col===tail.col)continue;
   const holds=ctx.holds[col],last=lb(holds,tail.t,h=>h[0])-1;
   // Immediately BEFORE release: new simultaneous heads do not count, other
   // simultaneous tails are still held. No epsilon-time magic is needed.
   if(last>=0&&holds[last][1]>=tail.t)other++;
  }
  const load=other/Math.max(1,K-1);held.push(load);interaction.push(load*independent);
  const a=lb(tails,tail.t,n=>n.t),b=lb(tails,tail.t+1e-8,n=>n.t);width.push((b-a)/K);
  if(j>lo&&tail.t>tails[j-1].t)rhythm.push(Math.log1p((tail.t-tails[j-1].t)/scale));
  if(j>0&&j+1<tails.length){
   const a=tails[j-1],b=tail,c=tails[j+1],h=hand(b.col);
   // All three timestamp groups must be singletons. Strict a.t < b.t < c.t
   // alone is insufficient: a/c can be the last/first member of a chord,
   // whose arbitrary column ordering reverses under a physical mirror.
   const uniqueEndpoints=(j<2||tails[j-2].t<a.t)&&(j+2>=tails.length||c.t<tails[j+2].t);
   if(uniqueEndpoints&&a.t<b.t&&b.t<c.t&&h>=0&&hand(a.col)===h&&hand(c.col)===h&&new Set([a.col,b.col,c.col]).size===3){
    const distance=Math.abs(a.col-b.col)+Math.abs(b.col-c.col),span=Math.max(a.col,b.col,c.col)-Math.min(a.col,b.col,c.col);
    order+=(distance-span)/Math.max(1,Math.floor(K/2)-1)*Math.exp(-(c.t-a.t)/(3*scale));
   }
  }
 }
 const values=[n/((end-start)/1000)/K,mean(isolation),mean(recovery),mean(asynchrony),sd(rhythm),
  mean(held),order/n,mean(width),mean(interaction)];
 if(values.some(v=>!Number.isFinite(v)))throw Error('Nonfinite release geometry');
 return {values,metrics:Object.fromEntries(KEYS.map((k,i)=>[k,values[i]])),tails:n};
}
module.exports={KEYS,releaseContext,releaseGeometry};
