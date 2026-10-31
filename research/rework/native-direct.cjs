'use strict';
// Fresh text -> ONE selected native partition/head -> the actual policy.
// Research entry: fs-based instrumented experts are not a browser delivery.
const {metricApi}=require('./metric.cjs'),{context,inspectPrepared}=require('./legacy-lab.cjs');
const {BOUNDARIES,HEADS,decisionSupport,nativeApi}=require('./native.cjs');
const {evaluateCachedNative}=require('./native-policy.cjs');

function analyzeNative(text,profile,options={}){
 const {config:c,sourceParameters:p}=profile;
 const api=p.logical5NeutralCoverage?metricApi(p.releaseMs,p.metricStrength):nativeApi('original');
 Object.assign(api.lab.CONFIG,BOUNDARIES[c.partition]);
 const prepared=context(api.lab,text,options.speedRate??1);
 if(prepared.map.keys!==7)throw Error('Native fine classifier requires 7K');
 const view=inspectPrepared(api,prepared),classified=api.lab.classifySegments(view.cache,HEADS[c.head]);
 const burdens=view.segments.map(s=>s.f.nps*(.5+s.f.chordMean/3)*(.6+s.f.jackSpeedBpm/150));
 function compact(s){
  let mass=0,duration=0;
  view.segments.forEach((r,i)=>{const d=Math.max(0,Math.min(r.end,s.end)-Math.max(r.start,s.start));mass+=d*burdens[i];duration+=d;});
  return {start:s.start,end:s.end,tag:s.tag,family:s.family,confidence:s.confidence,
   probability:decisionSupport(s),burden:duration?mass/duration:0};
 }
 const raw=classified.map(compact),legacy=api.lab.smoothSegments(structuredClone(classified)).map(compact);
 // The singleton wrapper intentionally has indices 0/0; c's original indices
 // identify the real chosen feature extraction above, not wrapper positions.
 const result=evaluateCachedNative({partitions:[{cases:[{raw,legacy}]}]},{...c,partition:0,head:0});
 return {map:prepared.map,windows:prepared.windows,...result,
  modelInfo:{nativeBoundary:BOUNDARIES[c.partition],head:HEADS[c.head],sourceParameters:p,
   scoreKind:'normalized conditional historical expert affinity, not calibrated correctness',
   coefficientTraining:'frozen historical experts; exact historical exposure unknown',researchOnly:true}};
}
module.exports={analyzeNative};
