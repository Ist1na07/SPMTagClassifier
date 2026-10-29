'use strict';
const {FEATURE_KEYS} = require('./features.cjs');
const {scores,linearQuantile} = require('./model.cjs');
const FINE_TO_COARSE = {
  chordjack:'jack',minijack:'jack',vibro:'jack',anchor:'jack',
  'dense chordstream':'stamina','fast chordstream':'stamina',speed:'speed',tech:'tech',wildcard:'tech',
  coordination:'ln',density:'ln',inverse:'ln',release:'ln',technical:'ln','ln wildcard':'ln'
};
function aggregateEvidence(windows, probabilities, mode='full', duration=false) {
  if (['no_geometry','no_holds','no_burden'].includes(mode)) mode='confidence_evidence';
  if (!['hard_evidence','duration_evidence','effort_evidence','confidence_evidence'].includes(mode)) mode='full';
  const out=Array(probabilities[0].length).fill(0); let total=0;
  windows.forEach((w,i)=>{
    const p=probabilities[i];let weight=duration?Math.max(0,w.end-w.start):1;
    if (['full','effort_evidence'].includes(mode)) weight*=Math.max(w.values[FEATURE_KEYS.indexOf('effort')],1e-8);
    if (['full','confidence_evidence'].includes(mode)) weight*=Math.max(...p);
    const best=p.indexOf(Math.max(...p));
    for(let j=0;j<out.length;j++)out[j]+=weight*(mode==='hard_evidence'?Number(j===best):p[j]);
    total+=weight;
  });
  return out.map(v=>v/Math.max(total,1e-12));
}
function pooledVector(windows, profile, localProbabilities) {
  const idx=profile.indices, means=[],sds=[],q90=[];
  for(const j of idx) {
    const a=windows.map(w=>w.values[j]),m=a.reduce((s,x)=>s+x,0)/a.length;
    means.push(m);sds.push(Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/a.length));q90.push(linearQuantile(a,.9));
  }
  const v=[...means,...sds,...q90];
  if(profile.variant!=='no_evidence')v.push(...aggregateEvidence(windows,localProbabilities,profile.variant));
  return v;
}
function localProbabilities(windows, profile) {
  return profile.local ? windows.map(w=>scores(profile.local,profile.indices.map(i=>w.values[i]))) : null;
}
/** Coarse support union for independent whole-map labels, not a softmax. */
function coarseSupports(classes,p) {
  const out={jack:0,ln:0,speed:0,stamina:0,tech:0};
  for(let i=0;i<classes.length;i++) {
    const c=FINE_TO_COARSE[classes[i]];
    if(c)out[c]=1-(1-out[c])*(1-p[i]);
  }
  return out;
}
/** Preserve 7K conditional detail while interpolating the 4K/7K coarse masses. */
function transferDistribution(classes,p7,classes4,p4,lambda=2/3) {
  const coarse={},groups={};let active=0;
  classes.forEach((t,i)=>{const c=FINE_TO_COARSE[t];if(c){coarse[c]=(coarse[c]||0)+p7[i];(groups[c]??=[]).push(i);active+=p7[i];}});
  const out=[...p7];
  for(const [c,ids] of Object.entries(groups)) {
    const other=p4[classes4.indexOf(c)]||0,desired=lambda*(coarse[c]||0)+(1-lambda)*other*active;
    for(const i of ids)out[i]=coarse[c]>0?desired*p7[i]/coarse[c]:desired/ids.length;
  }
  const sum=out.reduce((a,b)=>a+b,0);return out.map(x=>x/Math.max(1e-12,sum));
}
module.exports={aggregateEvidence,pooledVector,localProbabilities,FINE_TO_COARSE,coarseSupports,transferDistribution};
