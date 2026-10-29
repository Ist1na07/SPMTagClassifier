'use strict';
/** Separate uncertain local labels from genuine between-segment diversity.
 * Inputs are probability vectors (not raw legacy scores), nonnegative evidence
 * weights and an optional Break index. Break mass reduces active evidence.
 * Entropies are divided by log(number of non-Break classes).
 */
function composition(probabilities,weights,{breakIndex=-1}={}){
 if(!probabilities.length||probabilities.length!==weights.length)throw Error('ShapeMismatch');
 const D=probabilities[0].length,indices=Array.from({length:D},(_,i)=>i).filter(i=>i!==breakIndex);
 if(indices.length<2)throw Error('NeedAtLeastTwoActiveClasses');
 const dist=Array(indices.length).fill(0),rows=[];let total=0;
 const H=p=>-p.reduce((s,x)=>s+(x>0?x*Math.log(x):0),0)/Math.log(indices.length);
 probabilities.forEach((p,i)=>{
  if(p.length!==D||p.some(x=>!Number.isFinite(x)||x<0)||Math.abs(p.reduce((a,b)=>a+b,0)-1)>1e-9)throw Error('InvalidProbability');
  if(!Number.isFinite(weights[i])||weights[i]<0)throw Error('InvalidWeight');
  const active=indices.reduce((s,j)=>s+p[j],0),weight=weights[i]*active;
  if(!weight)return;
  const q=indices.map(j=>p[j]/active);total+=weight;
  for(let j=0;j<q.length;j++)dist[j]+=weight*q[j];rows.push({q,weight});
 });
 if(!total)return {indices,distribution:dist,activeWeight:0,entropy:0,meanLocalEntropy:0,diversity:0};
 for(let j=0;j<dist.length;j++)dist[j]/=total;
 const entropy=H(dist),local=rows.reduce((s,r)=>s+r.weight/total*H(r.q),0);
 return {indices,distribution:dist,activeWeight:total,entropy,meanLocalEntropy:local,
  diversity:Math.max(0,entropy-local)};
}
module.exports={composition};
