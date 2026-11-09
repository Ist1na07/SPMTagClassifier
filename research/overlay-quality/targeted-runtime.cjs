'use strict';
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {sequenceEvidence}=require('./sequence-evidence.cjs');
const TYPES=['dense chordstream','fast chordstream','tech'];
const FEATURE_INDEX={bias:[],sequence:[0,1],time:[2,3],global:[4,5],both:[0,1,2,3]};
function correction(values,model){
  const raw=FEATURE_INDEX[model.kind].map(i=>values[i]),s=model.scaler;
  const x=[...raw.map((v,j)=>Number.isFinite(v)?v:s.rawImputation[j]),...raw.map(v=>Number(!Number.isFinite(v)))];
  let delta=model.theta[0],k=1;
  for(let j=0;j<x.length;j++)if(s.keep[j])delta+=model.theta[k++]*(x[j]-s.mean[j])/s.scale[j];
  return delta;
}
function refine(parents,ctx,profile){
  const model=profile.model;
  if(!model||model.theta.every(v=>v===0))return parents;
  const out=[];
  for(const parent of parents){
    if(parent.family!=='rc'||!TYPES.includes(parent.tag)){out.push(parent);continue;}
    const n=profile.context==='local2s'?Math.max(1,Math.floor((parent.end-parent.start)/500)):1,local=[];
    for(let i=0;i<n;i++){
      const start=parent.start+(parent.end-parent.start)*i/n,end=parent.start+(parent.end-parent.start)*(i+1)/n,mid=(start+end)/2;
      const a=profile.context==='local2s'?Math.max(parent.start,mid-1000):parent.start;
      const b=profile.context==='local2s'?Math.min(parent.end,mid+1000):parent.end;
      const values=sequenceEvidence(ctx,a,b),delta=correction(values,model);
      const scores={...parent.scores},mass=TYPES.reduce((sum,c)=>sum+(scores[c]??0),0);
      const z=TYPES.map((c,j)=>Math.log((scores[c]??0)/mass)+(j===2?delta:0)),max=Math.max(...z),q=z.map(v=>Math.exp(v-max)),sum=q.reduce((a,b)=>a+b,0);
      TYPES.forEach((c,j)=>{scores[c]=mass*q[j]/sum;});
      const ranked=Object.entries(scores).sort((a,b)=>b[1]-a[1]);let[tag,confidence]=ranked[0];
      if(tag==='wildcard'||tag==='ln wildcard')for(const[c,v]of ranked.slice(1))if(!['wildcard','ln wildcard'].includes(c)&&v>.3){tag=c;confidence=v;break;}
      const secondary=ranked.length>1&&ranked[1][1]>=ranked[0][1]*.75&&ranked[1][1]>.3?ranked[1][0]:undefined;
      const current={...parent,start,end,tag,confidence:Math.min(1,confidence),secondary,scores},last=local.at(-1);
      if(last&&last.tag===tag&&last.secondary===secondary&&last.family===current.family){
        const x=last.end-last.start,y=end-start;last.end=end;
        for(const c of Object.keys(scores))last.scores[c]=(last.scores[c]*x+scores[c]*y)/(x+y);
        last.confidence=Math.min(1,last.scores[tag]??1);
      }else local.push(current);
    }
    out.push(...local);
  }
  return out;
}
function createConsumer(archive,profile){
  return {analyzeMap(text,opts={}){
    let map=archive.native.parseOsu(text);const speed=opts.speedRate??1;
    if(speed!==1)map=archive.native.rescale(map,speed);
    const cm=archive.buildCachedMap(text,opts),parents=archive.native.classifySegments(cm,{scorer:'model'});
    const refined=refine(parents,prepare(parseOsu(text,opts)),profile),segments=archive.native.smoothSegments(refined);
    const overall=archive.native.aggregateOverall(segments,profile.policy);
    return {map,segments,overall,debug:{windows:cm.windows,norm:undefined,ctx:undefined}};
  }};
}
module.exports={refine,correction,createConsumer};
