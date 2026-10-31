'use strict';
// Actual research caller for a selected small LN update. No archive edits.
const {loadLab,context,inspectPrepared}=require('./legacy-lab.cjs');
const parser=require('../../src/parser.cjs'),geometry=require('./release-geometry.cjs');
const {decisionSupport}=require('./native.cjs'),{evaluateCachedNative}=require('./native-policy.cjs');
const injection={id:'ln_adapter_override',apply:(s,r)=>r(s,'  function modelScores(fam, f, extras, cutDemandRef) {',
 '  function modelScores(fam, f, extras, cutDemandRef) {\n    if(fam==="ln"&&extras.__spmLnAdapter)return extras.__spmLnAdapter;')};

function analyzeLnAdapter(text,profile,options={}){
 const model=profile.model,policy=profile.policy,rate=options.speedRate??1;
 if(!Number.isFinite(rate)||rate<=0)throw Error('speedRate must be finite and positive');
 const api=loadLab({patches:[injection]}),prepared=context(api.lab,text,rate);
 if(prepared.map.keys!==7)throw Error('LN expert adapter requires7K');
 const view=inspectPrepared(api,prepared),expert=api.lab.MODEL.ln,D=expert.mean.length;
 if(model.delta.length!==model.keys.length+1)throw Error('LN adapter shape differs');
 if(model.delta.some(v=>v!==0)){
  const modern=geometry.releaseContext(parser.prepare(parser.parseOsu(text,{speedRate:rate}),10));
  for(let i=0;i<view.segments.length;i++){
   const s=view.segments[i],g=geometry.releaseGeometry(modern,s.start,s.end).metrics;
   const fields={a:g.headIsolation,b:g.heldOtherAtTail,ab:g.headIsolation*g.heldOtherAtTail,
    ...Object.fromEntries(geometry.KEYS.map(k=>['geometry:'+k,g[k]])),
    ...Object.fromEntries(['tailIndependence','lnAreaRatio','lnWeightRatio'].map(k=>['old:'+k,s.vector[view.featureKeys.indexOf(k)]]))};
   let delta=model.delta.at(-1);
   model.keys.forEach((key,j)=>{if(!Object.hasOwn(fields,key))throw Error('Unsupported LN adapter feature '+key);
    delta+=model.delta[j]*Math.max(-model.clip,Math.min(model.clip,(fields[key]-model.mean[j])/model.scale[j]));});
   const z=expert.classes.map((tag,k)=>{
    let value=expert.b[k];for(let j=0;j<D;j++)value+=expert.W[k*D+j]*(s.vector[j]-expert.mean[j])/expert.std[j];
    return value+(tag==='release'?delta:0);
   });
   const peak=Math.max(...z),p=z.map(v=>Math.exp(v-peak)),sum=p.reduce((a,b)=>a+b,0);
   view.cache.extras[i].__spmLnAdapter=Object.fromEntries(expert.classes.map((tag,k)=>[tag,p[k]/sum]));
  }
 }
 const classified=api.lab.classifySegments(view.cache,{lnDominantRatio:policy.lnRatio,scorer:policy.scorer});
 const raw=classified.map((s,i)=>({start:s.start,end:s.end,tag:s.tag,family:s.family,secondary:s.secondary,equivBpm:s.equivBpm,confidence:s.confidence,probability:decisionSupport(s),
  burden:view.segments[i].f.nps*(.5+view.segments[i].f.chordMean/3)*(.6+view.segments[i].f.jackSpeedBpm/150)}));
 const result=evaluateCachedNative({partitions:[{cases:[{raw}]}]},{...policy,extraGate:profile.extraGate??'all',partition:0,head:0,smoothing:'coherent'});
 return {map:prepared.map,windows:prepared.windows,...result,modelInfo:{researchOnly:true,variant:model.variant,
  coefficientPrior:'archived experts, exact historical exposure unknown',updatedClass:'conditional LN Release only',boundaries:'original raw automatic'}};
}
module.exports={analyzeLnAdapter};
