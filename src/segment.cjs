'use strict';
const {named} = require('./model.cjs');
const LN = new Set(['coordination','density','inverse','release','technical','ln wildcard','ln']);
function family(tag){return tag==='break'?'break':LN.has(tag)?'ln':'rc';}
function decode(probabilities, penalty=1) {
  let cost=probabilities[0].map(p=>-Math.log(Math.max(p,1e-15)));const back=[];
  for(let i=1;i<probabilities.length;i++) {
    const next=[],prev=[];
    for(let k=0;k<cost.length;k++) {
      let best=Infinity,index=0;
      for(let j=0;j<cost.length;j++){const v=cost[j]+(j===k?0:penalty);if(v<best){best=v;index=j;}}
      next.push(best-Math.log(Math.max(probabilities[i][k],1e-15)));prev.push(index);
    }
    back.push(prev);cost=next;
  }
  const path=[cost.indexOf(Math.min(...cost))];
  for(let i=back.length-1;i>=0;i--)path.push(back[i][path.at(-1)]);
  return path.reverse();
}
function segmentsFrom(windows,probabilities,classes,penalty=1) {
  const path=decode(probabilities,penalty), groups=[];
  windows.forEach((w,i)=>{let g=groups.at(-1);if(!g||g.label!==path[i])groups.push(g={label:path[i],indices:[]});g.indices.push(i);});
  return groups.map(g=>{
    const first=windows[g.indices[0]],last=windows[g.indices.at(-1)],tag=classes[g.label];
    const p=Array(classes.length).fill(0);let total=0,effort=0,notes=0,occupancy=0,bpm=0,bpmW=0;
    for(const i of g.indices){
      const w=windows[i],weight=w.end-w.start;total+=weight;
      probabilities[i].forEach((v,j)=>p[j]+=weight*v);
      effort+=weight*w.metrics.effort;occupancy+=weight*w.metrics.occupancy;notes+=w.notes;
      if(w.equivBpm){bpm+=weight*w.equivBpm;bpmW+=weight;}
    }
    const probs=p.map(x=>x/Math.max(total,1e-12)), ranked=classes.map((c,i)=>[c,probs[i]]).sort((a,b)=>b[1]-a[1]);
    const entropy=-probs.reduce((s,x)=>x>0?s+x*Math.log(x):s,0)/Math.log(classes.length);
    return {start:first.start,end:last.end,tag,family:family(tag),confidence:probs[g.label],
      scores:named(classes,probs),secondary:ranked.find(([t])=>t!==tag)?.[0],
      equivBpm:tag==='break'||!bpmW?undefined:Math.round(bpm/bpmW),
      evidence:{normalizedEntropy:entropy,margin:ranked[0][1]-(ranked[1]?.[1]||0),
        effort:effort/Math.max(total,1e-12),occupancy:occupancy/Math.max(total,1e-12),
        featureWindows:g.indices.length,contextNoteObservations:notes}};
  });
}
module.exports={decode,segmentsFrom,family};
