'use strict';
// Independent JS consumer of exported LOCAL parameters. Research-only until
// source-held-out results and actual integration have passed acceptance.
const RATE=new Set(['npsPerKey','rowRate','rowBpm','tailRate','effort','handTransitionRatePerKey','handSubstitutionRatePerKey','handFixedGripRatePerKey']);
const sigmoid=x=>x>=0?1/(1+Math.exp(-x)):Math.exp(x)/(1+Math.exp(x));
const dot=(a,b)=>a.reduce((s,x,i)=>s+x*b[i],0);
const logit=x=>x>0?Math.log(x)-Math.log1p(-x):-Infinity;
function allocate(base,delta){
  const budget=base.reduce((s,x)=>s+x,0);if(!budget)return base.map(()=>0);
  let lo=Math.min(...delta),hi=Math.max(...delta);
  for(let pass=0;pass<70;pass++){
    const mid=(lo+hi)/2,q=base.map((b,j)=>b>0?sigmoid(logit(b)+delta[j]-mid):0),sum=q.reduce((s,x)=>s+x,0);
    if(Math.abs(sum-budget)<2e-13)return q;
    if(sum>budget)lo=mid;else hi=mid;
  }
  throw Error('Local support-budget projection did not converge');
}
function choose(frame,scores,family=frame.family,routeDegree=1){
  if(family==='break')return {...frame,family,tag:'break',secondary:null,confidence:routeDegree,scores:{break:1}};
  const ranked=Object.entries(scores).sort((a,b)=>b[1]-a[1]);let [tag,value]=ranked[0];
  if(['wildcard','ln wildcard'].includes(tag))for(const [c,v]of ranked.slice(1))if(!['wildcard','ln wildcard'].includes(c)&&v>.3){tag=c;value=v;break;}
  const secondary=ranked.length>1&&ranked[1][1]>=ranked[0][1]*.75&&ranked[1][1]>.3?ranked[1][0]:null;
  return {...frame,family,tag,secondary,confidence:Math.min(1,value)*routeDegree,scores};
}
function features(geometry,hand,names){
  return geometry.map((g,i)=>Object.fromEntries(names.map(k=>[k,RATE.has(k)?Math.log1p(g[k]??hand[i][k]):g[k]??hand[i][k]])));
}
function competition(frames,x,model,classes){
  const support=frames.map(f=>{const den=Math.max(1,...Object.values(f.scores));return classes.map(c=>f.family==='break'?0:Math.min(.999,Math.max(0,f.scores[c]??0)/den));});
  const output=frames.map((f,i)=>{
    if(f.family==='break')return {...f};
    const block=model.blocks.find(b=>b.family===f.family),den=Math.max(1,...Object.values(f.scores));
    const q0=block.classes.map(c=>Math.min(.999,Math.max(0,f.scores[c]??0)/den));
    const v=[1,...model.features.map(k=>x[i][k])],delta=block.coefficients.map(row=>dot(row,v)),q=allocate(q0,delta),scores={...f.scores};
    block.classes.forEach((c,j)=>{scores[c]=q[j]*den;const at=classes.indexOf(c);if(at>=0)support[i][at]=q[j];});
    return choose(f,scores);
  });
  return {frames:output,support};
}
function routing(frames,x,model,candidates,classes){
  const probabilities=[],support=[];
  const output=frames.map((f,i)=>{
    const v=[1,...model.features.map(k=>x[i][k])],z=model.coefficients.map(row=>dot(row,v));z[['break','rc','ln'].indexOf(f.family)]+=model.config.priorStrength;
    const max=Math.max(...z),p=z.map(v=>Math.exp(v-max)),sum=p.reduce((s,v)=>s+v,0);p.forEach((_,j)=>p[j]/=sum);probabilities.push(p);
    const experts=['rc','ln'].map(name=>{const scores=candidates[i][name],den=Math.max(1,...Object.values(scores));return classes.map(c=>Math.min(.999,Math.max(0,scores[c]??0)/den));});
    support.push(classes.map((_,j)=>p[1]*experts[0][j]+p[2]*experts[1][j]));
    const route=p.indexOf(Math.max(...p));return route?choose(f,{...candidates[i][route===1?'rc':'ln']},route===1?'rc':'ln',p[route]):choose(f,{},'break',p[0]);
  });
  return {frames:output,support,probabilities};
}
module.exports={allocate,choose,features,competition,routing};
