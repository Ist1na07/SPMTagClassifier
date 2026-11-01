'use strict';
// Event descriptors, NOT a pressure score or classifier. Consume the caller's
// existing native-key prepared context; do not parse or build occupancy again.
const {lowerBound:lb}=require('../../src/parser.cjs');
function releaseActions(ctx,start,end,scalesMs=[15,30,60,120]){
 if(!Number.isFinite(start)||!Number.isFinite(end)||!(end>start))throw Error('Finite half-open interval required');
 if(!scalesMs.length||scalesMs.some(t=>!Number.isFinite(t)||t<=0))throw Error('Positive finite release scales required');
 const K=ctx.map.keys,hand=c=>K===7&&c===3?-1:c<Math.floor(K/2)?0:1;
 const groups=[],weighted=scalesMs.map(scaleMs=>({scaleMs,retainedPairs:0,sameHandPairs:0}));let largeScalePairs=0,largeScaleSameHandPairs=0,j=lb(ctx.tails,start,n=>n.t);
 while(j<ctx.tails.length&&ctx.tails[j].t<end){
  const t=ctx.tails[j].t,released=new Set();
  while(j<ctx.tails.length&&ctx.tails[j].t===t)released.add(ctx.tails[j++].col);
  const before=[],remaining=[],spans=[];
  for(let col=0;col<K;col++){
   const holds=ctx.holds[col],i=lb(holds,t,h=>h[0])-1;
   if(i>=0&&holds[i][1]>=t)before.push(col);
   if(i>=0&&holds[i][1]>t&&!released.has(col)){
    remaining.push(col);spans.push({col,past:t-holds[i][0],future:holds[i][1]-t});
   }
  }
  let sameHandPairs=0;
  for(const a of released)for(const b of remaining)if(hand(a)>=0&&hand(a)===hand(b))sameHandPairs++;
  for(const span of spans){
   // tau^2 times the exponential product tends to past*future. Express
   // that limit in seconds squared; train-only scaling handles its units.
   const v=span.past*span.future/1e6;largeScalePairs+=released.size*v;
   for(const a of released)if(hand(a)>=0&&hand(a)===hand(span.col))largeScaleSameHandPairs+=v;
  }
  for(const w of weighted)for(const span of spans){
   // Continuous at both a simultaneous head and a simultaneous tail. An
   // infinitesimal split of a release chord must not create finite pressure.
   const v=(-Math.expm1(-span.past/w.scaleMs))*(-Math.expm1(-span.future/w.scaleMs));
   w.retainedPairs+=released.size*v;
   for(const a of released)if(hand(a)>=0&&hand(a)===hand(span.col))w.sameHandPairs+=v;
  }
  groups.push({t,released:[...released],heldBefore:before,retained:remaining,
   retainedPairs:released.size*remaining.length,sameHandPairs});
 }
 const seconds=(end-start)/1000,n=groups.length,sum=k=>groups.reduce((s,g)=>s+g[k],0);
 return {groups,largeScaleLimit:{retainedPairsPerKeySecond:largeScalePairs/K/seconds,
  sameHandPairsPerKeySecond:largeScaleSameHandPairs/K/seconds,
  meaning:'tau^2/1e6 rescaled tau->infinity shape, not bounded retained fraction or physical pressure'},continuous:weighted.map(w=>({scaleMs:w.scaleMs,
  retainedPairsPerKeySecond:w.retainedPairs/K/seconds,
  sameHandPairsPerKeySecond:w.sameHandPairs/K/seconds})),
  metricScope:'exact event groups are diagnostics; continuous retained-pair fields are candidate model inputs, not calibrated difficulty',
  metrics:{releaseGroupsPerSecond:n/seconds,
  releaseWidthMean:n?groups.reduce((s,g)=>s+g.released.length,0)/n:0,
  allHeldReleasedGroupFraction:n?groups.filter(g=>g.heldBefore.length>0&&g.retained.length===0).length/n:0,
  retainedFingerPairsPerKeySecond:sum('retainedPairs')/K/seconds,
  sameHandRetainedPairsPerKeySecond:sum('sameHandPairs')/K/seconds}};
}
module.exports={releaseActions};
