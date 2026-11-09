'use strict';
// Independent implementation inspired by ordered-row recognisers and hand
// sequencing. No external implementation or thresholds are ported.
const {lowerBound,popcount}=require('../../src/parser.cjs');
const KEYS=['adjacentChordShare','chordSingleSwitchShare','handLogIoiSd','handIoiChange','globalLogIoiSd','globalIoiChange'];
function timing(times){
  const iv=[];for(let i=1;i<times.length;i++)if(times[i]>times[i-1])iv.push(times[i]-times[i-1]);
  if(iv.length<3)return [null,null,0];
  const logs=iv.map(Math.log),mean=logs.reduce((a,b)=>a+b,0)/logs.length;
  let changes=0;for(let i=1;i<iv.length;i++)changes+=Number(Math.max(iv[i],iv[i-1])/Math.min(iv[i],iv[i-1])>1.35);
  return [Math.sqrt(logs.reduce((a,b)=>a+(b-mean)**2,0)/logs.length),changes/(iv.length-1),iv.length];
}
function sequenceEvidence(ctx,start,end){
  if(ctx.map.keys!==7)throw Error('Bounded 7K study only');
  const rows=ctx.rows.slice(lowerBound(ctx.rows,start,r=>r.t),lowerBound(ctx.rows,end,r=>r.t));
  let adjacent=0,switches=0;
  for(let i=1;i<rows.length;i++){
    const a=popcount(rows[i-1].mask),b=popcount(rows[i].mask);
    adjacent+=Number(a>1&&b>1&&(rows[i-1].mask&rows[i].mask)===0);
    switches+=Number((a>1)!==(b>1));
  }
  const hands=[7,112].map(mask=>timing(rows.filter(r=>r.mask&mask).map(r=>r.t)));
  const support=hands.reduce((a,r)=>a+r[2],0),global=timing(rows.map(r=>r.t));
  return [rows.length>1?adjacent/(rows.length-1):null,rows.length>1?switches/(rows.length-1):null,
    ...[0,1].map(j=>support?hands.reduce((a,r)=>a+(r[2]?r[j]*r[2]:0),0)/support:null),...global.slice(0,2)];
}
module.exports={KEYS,sequenceEvidence};
