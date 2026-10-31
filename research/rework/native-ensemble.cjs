'use strict';
const {CLASSES}=require('./native.cjs');
const LN=new Set(['coordination','density','inverse','release','technical','ln wildcard']);
const entropy=p=>-p.reduce((s,v)=>s+(v?v*Math.log(v):0),0);
function combineSegments(first,second){
 const points=[...new Set([...first,...second].flatMap(s=>[s.start,s.end]))].sort((a,b)=>a-b),out=[];
 let a=0,b=0;
 for(let i=0;i+1<points.length;i++){
  const start=points[i],end=points[i+1];if(!(end>start))continue;
  while(a<first.length&&first[a].end<=start)a++;
  while(b<second.length&&second[b].end<=start)b++;
  const available=[first[a],second[b]].filter(s=>s&&s.start<=start&&s.end>=end);
  if(!available.length)continue; // Both branches lack this interval; do not invent a label.
  const probability=CLASSES.map((_,k)=>available.reduce((s,r)=>s+r.probability[k],0)/available.length);
  const tag=CLASSES[probability.indexOf(Math.max(...probability))],family=tag==='break'?'break':LN.has(tag)?'ln':'rc';
  const disagreement=Math.max(0,(entropy(probability)-available.reduce((s,r)=>s+entropy(r.probability),0)/available.length)/Math.log(2));
  out.push({start,end,tag,family,probability,confidence:Math.max(...probability),
   burden:available.reduce((s,r)=>s+r.burden,0)/available.length,availableViews:available.length,viewDisagreement:disagreement});
 }
 return out;
}
module.exports={combineSegments};
