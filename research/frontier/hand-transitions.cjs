'use strict';
// Exact simultaneous finger sets on each hand's OWN event sequence. Empty
// other-hand rows do not become finger changes. Heads only; no LN inference.
const {lowerBound:lb,popcount:pc}=require('../../src/parser.cjs');
const {prepareHandSequence}=require('./hand-sequence.cjs');
const KEYS=['handRepeatOverlap','handChangeJaccard','handReturn',
 'handSubstitutionRatePerKey','handFixedGripRatePerKey'];
function handTransitions(ctx,start,end,shared=prepareHandSequence(ctx)){
 if(!(Number.isFinite(start)&&Number.isFinite(end)&&end>start))throw Error('Finite nonempty interval required');
 if(shared.source!==ctx)throw Error('Hand context belongs to another prepared map');
 const seconds=(end-start)/1000,profiles=[];
 for(const view of shared.views){
  let overlap=0,minSize=0,jaccard=0,pairs=0,returns=0,returnSize=0,substitutions=0,fixed=0;
  for(const {rows}of view){
   const a=lb(rows,start,r=>r.t),b=lb(rows,end,r=>r.t);
   for(let i=Math.max(1,a);i<b;i++){
    const A=rows[i-1].mask,B=rows[i].mask,inter=pc(A&B);
    overlap+=inter;minSize+=Math.min(pc(A),pc(B));
    jaccard+=pc(A^B)/pc(A|B);pairs++;
    substitutions+=Math.min(pc(A&~B),pc(B&~A));fixed+=A===B?1:0;
    if(i>=2){const P=rows[i-2].mask;returns+=pc(P&B&~A);returnSize+=pc(P);}
   }
  }
  profiles.push([minSize?overlap/minSize:null,pairs?jaccard/pairs:null,
   returnSize?returns/returnSize:null,substitutions/(seconds*ctx.map.keys),fixed/(seconds*ctx.map.keys)]);
 }
 const values=KEYS.map((_,j)=>{const observed=profiles.map(p=>p[j]).filter(v=>v!==null);return observed.length?observed.reduce((s,v)=>s+v,0)/observed.length:null;});
 return{values,profiles,meaning:'Set-valued within-hand head transitions, allowing previous source events before the interval. Flexible 7K center assigned to each hand in separate profiles, then averaged. No class probability or cut-force calibration.'};
}
module.exports={KEYS,handTransitions};
