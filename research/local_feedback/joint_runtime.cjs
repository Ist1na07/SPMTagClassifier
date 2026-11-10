'use strict';
const {measure}=require('./activity_runtime.cjs');
const {allocate}=require('./runtime.cjs');
const softmax=z=>{const m=Math.max(...z),p=z.map(v=>Math.exp(v-m)),s=p.reduce((a,b)=>a+b,0);return p.map(v=>v/s);};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
function apply(model,map,source,candidates,classes,context){
  if(model.kind!=='joint-activity-budget'||model.allocation!=='fixed-affinity Bernoulli entropy projection')throw Error('Unsupported joint allocation');
  if(source.length!==candidates.length)throw Error('Joint expert inventory mismatch');
  const local=measure(map,source,model,context),frames=[],support=[],probabilities=[];
  for(let i=0;i<local.frames.length;i++){
    const old=local.frames[i],x=local.inputs[i],v=[1,...model.features.map(k=>x[k])],residual=model.blocks.map(b=>b.coefficients.map(row=>dot(row,v))),family=['break','rc','ln'].indexOf(old.family);residual[0][family]+=model.priorStrength;
    if(model.config.physicalEligibility){
      if(!(x.npsPerKey>0))residual[0][1]=-Infinity;
      if(!(x.lnHeadFraction>0||x.occupancy>0||x.tailRate>0))residual[0][2]=-Infinity;
    }
    const p=softmax(residual[0]),scores=[];probabilities.push(p);
    for(let f=0;f<2;f++){
      const raw=candidates[old.parent][f===0?'rc':'ln'],div=Math.max(1,...Object.values(raw)),block=model.blocks[f+1],q0=block.classes.map(c=>Math.min(.999,Math.max(0,raw[c]??0)/div)),possible=Number.isFinite(residual[0][f+1]),q=possible?allocate(q0,residual[f+1]):q0;scores.push(Object.fromEntries(block.classes.map((c,j)=>[c,q[j]])));
    }
    support.push(classes.map(c=>p[1]*(scores[0][c]??0)+p[2]*(scores[1][c]??0)));
    const route=p.indexOf(Math.max(...p));
    if(!route){frames.push({...old,family:'break',tag:'break',secondary:null,confidence:p[0],scores:{break:1}});continue;}
    // Learned type competition replaces the old unconditional wildcard
    // demotion. Secondary semantics are unchanged and scores stay bounded.
    const s=scores[route-1],ranked=Object.keys(s).sort((a,b)=>s[b]-s[a]),tag=ranked[0],secondary=ranked.length>1&&s[ranked[1]]>=.75*s[tag]&&s[ranked[1]]>.3?ranked[1]:null;
    frames.push({...old,family:route===1?'rc':'ln',tag,secondary,confidence:s[tag]*p[route],scores:s});
  }
  return {frames,support,probabilities,parents:local.parents,inputs:local.inputs};
}
module.exports={apply};
