'use strict';
// Independent fixed-product ablation: remove only CS support where hand motion
// certifies no cut, leaving every other original score and support unchanged.
const guard=require('./static_cut_guard.cjs');
const {prepare}=require('../../src/parser.cjs');
const CS=new Set(['dense chordstream','fast chordstream']);
function change(s){
  const scores={...s.scores,'dense chordstream':0,'fast chordstream':0};
  if(!CS.has(s.tag))return {...s,scores,secondary:CS.has(s.secondary)?undefined:s.secondary};
  const ranked=Object.entries(scores).sort((a,b)=>b[1]-a[1]);let [tag,value]=ranked[0];
  if(tag==='wildcard')for(const [c,v]of ranked.slice(1))if(c!=='wildcard'&&v>.3){tag=c;value=v;break;}
  const secondary=ranked[1]&&ranked[1][1]>=.75*ranked[0][1]&&ranked[1][1]>.3?ranked[1][0]:undefined;
  return {...s,scores,tag,confidence:Math.min(1,value),secondary};
}
function apply(_model,map,source,_candidates,classes,context){
  const ctx=context??prepare(map),shared=guard.prepare(ctx),frames=[],support=[],certificates=[];
  source.forEach((old,parent)=>{
    const scores=old.scores??{[old.tag]:1},div=Math.max(1,...Object.values(scores)),q=classes.map(c=>old.family==='break'?0:Math.min(.999,Math.max(0,scores[c]??0)/div));
    for(let start=old.start;start<old.end;start+=500){
      const end=Math.min(old.end,start+500),mid=(start+end)/2,a=Math.max(map.firstNote,mid-2000),b=Math.max(a+1,Math.min(map.lastNote,mid+2000));
      const certified=old.family==='rc'&&guard.certifiedStaticRice(shared,a,b),s={...old,start,end,parent};
      frames.push(certified?change(s):s);support.push(q.map((v,j)=>certified&&CS.has(classes[j])?0:v));certificates.push(certified);
    }
  });
  return {frames,support,certificates};
}
module.exports={apply,change};
