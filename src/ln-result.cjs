'use strict';
const {analyzeLnAdapter,nativeEvidence,classes}=require('./generated/ln-engine.cjs');
const {features}=require('./features.cjs'),{version}=require('../package.json');
const geometry=require('./generated/native-geometry.cjs');
const named=p=>Object.fromEntries(classes.map((c,i)=>[c,p[i]]));
const ignored=new Set(['break','wildcard','ln wildcard','rc mix','ln mix','hybrid']);
function lnResult(text,profile,options,e){
 const r=analyzeLnAdapter(text,profile,options),atoms=nativeEvidence(r.segments,profile.policy),ctx=geometry.prepare(e.ctx);
 const descriptors=[],segments=r.segments.map(s=>{
  const sourceStart=s.start,sourceEnd=s.end,measurement=features(e.ctx,s.start,Math.max(s.start+1,s.end));
  const members=atoms.filter(a=>a.start>=s.start&&a.end<=s.end);
  const aggregationWeight=members.reduce((sum,a)=>sum+((profile.policy.mode==='soft'||!ignored.has(a.tag))?a.evidenceWeight:0),0);
  const distribution=named(s.probability),metrics=s.metrics;
  const measuredGeometry=geometry.describeInterval(ctx,s.start,Math.max(s.start+1,s.end)),m=measurement.metrics;
  descriptors.push({start:s.start,end:s.end,anchor:m.anchorShare,trill:m.trillFraction,roll:m.rollFraction,
   repetition:1-m.maskEntropy,occupancy:m.occupancy,releaseIndependence:m.releaseIndependence,geometry:measuredGeometry});
  return {...s,confidence:distribution[s.tag],scores:distribution,distribution,aggregationWeight,
   evidence:{normalizedEntropy:metrics.entropy,margin:metrics.margin,concentration:metrics.concentration,
    effort:measurement.metrics.effort,occupancy:measurement.metrics.occupancy,geometry:measuredGeometry,
    featureWindows:members.length,contextNoteObservations:measurement.notes,sourceStart,sourceEnd,
    ...(s.confidence===undefined?{}:{legacyConfidence:s.confidence})}};
 });
 // Match the stable API's display-envelope repair without silently changing
 // the fitted aggregation. Original score intervals remain explicit evidence.
 if(segments.length){segments[0].start=e.map.firstNote;segments.at(-1).end=e.map.lastNote+1;}
 const overall={...r.overall,scores:r.overall.shares,composition:r.overall.shares,
  compositionKind:profile.policy.mode==='soft'?'same local normalized support, weighted by fitted evidence policy':'weighted primary-label composition, conditional on concrete non-Break labels',
  scoreKind:'retrospectively tuned decision support; not calibrated correctness; weights retain weighted seconds'};
 return {map:e.map,segments,overall,descriptors,debug:{windows:r.windows,norm:undefined,ctx:undefined},
  modelInfo:{version,keyCount:7,variant:'anchored-ln-release',segmentEvidence:'archived RC/LN prior plus conditional Release adaptation; historical prior exposure unknown',
   temporalResolutionMs:250,classes,defaultReplacementAccepted:false,extraGate:profile.extraGate,
   aggregationPolicy:profile.policy,coverage:'display envelope repaired; aggregation uses original source intervals',
   confidenceMeaning:'normalized within-family support of displayed class; legacy affinity retained in evidence',supervision:'weak LN impressions plus available manual LN intervals'}};
}
module.exports={lnResult};
