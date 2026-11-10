'use strict';
// Label-free ordering diagnostic. Symbols are exact simultaneous HAND masks,
// not sorted notes inside a chord. Timing/contact scales remain separate.
const {lowerBound}=require('../../src/parser.cjs');
const KEYS=['handMaskEntropy','handFirstOrderUncertainty','handSecondOrderUncertainty','handTransitionRatePerKey'];
const side=(c,K,p)=>K===7&&c===3?p:c<Math.floor(K/2)?0:1;
const add=(m,k)=>m.set(k,(m.get(k)||0)+1);
function entropy(counts){const total=[...counts.values()].reduce((a,b)=>a+b,0);return total?[...counts.values()].reduce((s,n)=>s-n/total*Math.log(n/total),0):0;}
function prepareHandSequence(ctx){
 const K=ctx.map.keys,profiles=K===7?[0,1]:[0];
 return{source:ctx,views:profiles.map(p=>[0,1].map(h=>{
  const rows=[];for(const n of ctx.map.notes)if(side(n.col,K,p)===h){const previous=rows.at(-1);if(previous&&previous.t===n.t)previous.mask|=1<<n.col;else rows.push({t:n.t,mask:1<<n.col});}
  const lanes=Array.from({length:K},(_,c)=>c).filter(c=>side(c,K,p)===h);return{rows,lanes};
 }))};
}
function uncertainty(rows,a,b,order){
 const marginal=new Map(),contexts=new Map(),joint=new Map();let n=0;
 for(let j=Math.max(order,a);j<b;j++){
  const key=rows.slice(j-order,j).map(r=>r.mask).join(',');add(marginal,rows[j].mask);add(contexts,key);add(joint,key+'|'+rows[j].mask);n++;
 }
 const H=entropy(marginal),conditional=Math.max(0,entropy(joint)-entropy(contexts));
 return{marginalEntropy:n?H:null,uncertainty:n?(H>0?Math.min(1,conditional/H):0):null,observations:n};
}
function handSequence(ctx,start,end,shared=prepareHandSequence(ctx)){
 if(!(Number.isFinite(start)&&Number.isFinite(end)&&end>start))throw Error('Finite nonempty interval required');
 if(shared.source!==ctx)throw Error('Sequence context belongs to another prepared map');
 const profileValues=[],observations=[];
 for(const view of shared.views){
  const stats=[];
  for(const {rows,lanes}of view){
   const a=lowerBound(rows,start,r=>r.t),b=lowerBound(rows,end,r=>r.t),first=uncertainty(rows,a,b,1),second=uncertainty(rows,a,b,2);
   stats.push({mass:first.observations,values:[first.observations?first.marginalEntropy/Math.log(Math.max(2,2**lanes.length-1)):null,first.uncertainty,second.uncertainty],secondObservations:second.observations});
  }
  const total=stats.reduce((s,r)=>s+r.mass,0),v=[null,null,null];
  for(let j=0;j<3;j++){const mass=r=>j===2?r.secondObservations:r.mass,n=stats.reduce((s,r)=>s+mass(r),0);if(n)v[j]=stats.reduce((s,r)=>s+(mass(r)?mass(r)*r.values[j]:0),0)/n;}
  profileValues.push([...v,total/(ctx.map.keys*(end-start)/1000)]);observations.push({firstOrder:total,secondOrder:stats.reduce((s,r)=>s+r.secondObservations,0)});
 }
 const values=KEYS.map((_,j)=>{const seen=profileValues.map(r=>r[j]).filter(v=>v!==null);return seen.length?seen.reduce((a,b)=>a+b,0)/seen.length:null;});
 return{values,observations,meaning:'empirical conditional uncertainty of exact hand-mask sequences; finite-sample plug-in diagnostic, not calibrated predictability or a type rule'};
}
module.exports={KEYS,prepareHandSequence,handSequence,uncertainty};
