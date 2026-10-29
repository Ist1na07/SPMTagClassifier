'use strict';
const legacy=require('./legacy/tag-engine.cjs');
const {parseOsu,prepare}=require('./parser.cjs');
const {features,windows,FEATURE_KEYS}=require('./features.cjs');
const {scores,named,choose}=require('./model.cjs');
const {pooledVector,localProbabilities,aggregateEvidence,coarseSupports,transferDistribution}=require('./aggregate.cjs');
const {segmentsFrom}=require('./segment.cjs');
const profile4=require('./models/4k.json'),profile7=require('./models/7k.json');
const display7=require('./models/segment7k.json'),profile6=require('./models/6k.json');
function profile(key){return key===4?profile4:profile7;}
function evidenceMap(osuText,opts={}) {
  const map=parseOsu(osuText,opts),ctx=prepare(map,10);
  const contexts={10:ctx};
  const cache={};
  const getWindows=(ms,merge=10)=>{
    const key=ms+':'+merge;
    return cache[key]??(cache[key]=windows(contexts[merge]??(contexts[merge]=prepare(map,merge)),ms,2000));
  };
  return {map,ctx,getWindows};
}
function mapSupports(e,p) {
  const ws=e.getWindows(p.windowMs,p.rowMergeMs??10),local=localProbabilities(ws,p);
  const vector=pooledVector(ws,p,local);
  return {probabilities:scores(p.map,vector),local,windows:ws,vector};
}
function describe(w) {
  const m=w.metrics;
  return {anchor:m.anchorShare,trill:m.trillFraction,roll:m.rollFraction,
    repetition:1-m.maskEntropy,occupancy:m.occupancy,releaseIndependence:m.releaseIndependence};
}
function stable7(osuText,opts,e) {
  const r=legacy.analyzeMap(osuText,opts);
  // The inherited segmenter can omit a short leading pickup when snapping.
  // Extend only the envelope; keep the learned tags and overall decision.
  if(r.segments.length){r.segments[0].start=r.map.firstNote;r.segments.at(-1).end=r.map.lastNote;}
  for(const s of r.segments) {
    const w=features(e.ctx,s.start,Math.max(s.start+1,s.end));
    const raw=s.scores??{[s.tag]:1},sum=Object.values(raw).reduce((a,b)=>a+b,0);
    const distribution=Object.fromEntries(Object.entries(raw).map(([k,v])=>[k,sum>0?v/sum:Number(k===s.tag)]));
    const ranked=Object.values(distribution).sort((a,b)=>b-a);
    s.distribution=distribution;
    s.evidence={normalizedEntropy:ranked.length>1?-ranked.reduce((a,v)=>v>0?a+v*Math.log(v):a,0)/Math.log(ranked.length):0,
      margin:(ranked[0]||0)-(ranked[1]||0),effort:w.metrics.effort,occupancy:w.metrics.occupancy,
      featureWindows:1,contextNoteObservations:w.notes,legacyConfidence:s.confidence};
  }
  const total=Object.values(r.overall.weights).reduce((a,b)=>a+b,0);
  r.overall.composition=total>0?Object.fromEntries(Object.entries(r.overall.weights).map(([k,v])=>[k,v/total])):{break:1};
  r.overall.scoreKind='legacy decision scores; distribution is normalized affinity, not calibrated confidence';
  r.modelInfo={version:'1.2.0',keyCount:7,variant:'legacy-stable',researchVariant:profile7.variant,
    segmentEvidence:'original decisions; corrected interval-union diagnostics',temporalResolutionMs:250};
  r.descriptors=e.getWindows(4000).map(w=>({start:w.start,end:w.end,...describe(w)}));
  return r;
}
function analyzeMap(osuText,opts={}) {
  if(opts.algorithm==='legacy')return legacy.analyzeMap(osuText,opts);
  if(opts.algorithm && !['modern','stable'].includes(opts.algorithm))throw new Error('UnknownAlgorithm: expected modern, stable or legacy');
  const e=evidenceMap(osuText,opts),{map}=e,p=profile(map.keys);
  // The research head did not establish a 7K accuracy gain. Keep the mature
  // production decisions by default; opt in explicitly to the fully new head.
  if(map.keys===7&&opts.algorithm!=='modern')return stable7(osuText,opts,e);
  const ms=mapSupports(e,p);
  let display,probs,classes,penalty;
  if(map.keys===4) {
    display=ms.windows;classes=p.local?.classes??p.map.classes;penalty=0;
    probs=p.local?ms.local:display.map(()=>ms.probabilities);
  } else {
    display=e.getWindows(display7.windowMs);classes=display7.model.classes;penalty=display7.smoothingPenalty;
    probs=display.map(w=>scores(display7.model,display7.indices.map(i=>w.values[i])));
    if(map.keys===6&&profile4.local)probs=probs.map((p7,i)=>transferDistribution(classes,p7,profile4.local.classes,
      scores(profile4.local,profile4.indices.map(j=>display[i].values[j])),2/3));
  }
  // Physical silence is a structural Break, even for the five-way weak 4K head.
  if(!classes.includes('break')){classes=[...classes,'break'];probs=probs.map(p=>[...p,0]);}
  const breakIndex=classes.indexOf('break');
  probs=probs.map((p,i)=>display[i].notes===0&&display[i].metrics.occupancy===0?p.map((_,j)=>Number(j===breakIndex)):p);
  const segments=segmentsFrom(display,probs,classes,penalty);
  const composition=aggregateEvidence(display,probs,'full',true);
  let overallScores=ms.probabilities,overallClasses=p.map.classes,threshold=p.threshold;
  const transfer={};
  if(map.keys===6) {
    const other=mapSupports(e,profile4),fine=coarseSupports(profile7.map.classes,ms.probabilities);
    overallClasses=profile6.classes;threshold=profile6.threshold;
    overallScores=overallClasses.map(c=>profile6.lambda*fine[c]+(1-profile6.lambda)*other.probabilities[profile4.map.classes.indexOf(c)]);
    transfer.lambda=profile6.lambda;transfer.segmentLambda=2/3;
    transfer.evidenceLevel='4K/7K projected-chart surrogate validation; no human 6K pattern labels';
  }
  let sort;
  if(map.keys===4)sort=overallScores[overallClasses.indexOf('ln')]>=threshold?'ln':'rc';
  else {
    const ln=classes.reduce((s,t,i)=>s+(['coordination','density','inverse','release','technical','ln wildcard'].includes(t)?composition[i]:0),0);
    const s=profile6.sort??{rcMax:.18,lnMin:.68};sort=ln<s.rcMax?'rc':ln>s.lnMin?'ln':'hb';
  }
  const overall={tags:choose(overallClasses,overallScores,threshold),sort,
    scores:named(overallClasses,overallScores),weights:named(classes,composition),
    scoreKind:'independent multi-label model support',compositionKind:'normalized duration × effort × maximum local support',
    ...(map.keys===6?{transfer}:{} )};
  return {map,segments,overall,debug:{windows:display,norm:undefined,ctx:undefined},
    modelInfo:{version:'1.2.0',keyCount:map.keys,variant:p.variant,
      segmentEvidence:map.keys===4?'weak whole-map labels':map.keys===6?'cross-key extrapolation':'manual interval labels plus weak map labels',
      temporalResolutionMs:2000},descriptors:display.map(w=>({start:w.start,end:w.end,...describe(w)}))};
}
function extractIntermediate(text,opts={}) {
  if(opts.algorithm==='legacy')return legacy.extractIntermediate(text,opts);
  const r=analyzeMap(text,opts);return {map:r.map,windows:r.debug.windows,segments:r.segments,overall:r.overall};
}
function buildEvidenceMap(text,opts={}) {
  const e=evidenceMap(text,opts),ws=e.getWindows(profile(e.map.keys).windowMs);
  return {map:e.map,windows:ws,global:features(e.ctx,e.map.firstNote,Math.max(e.map.firstNote+1,e.map.lastNote+1)),featureKeys:FEATURE_KEYS};
}
// Preserve the original cache schema and entry point for existing callers.
module.exports={analyzeMap,extractIntermediate,buildCachedMap:legacy.buildCachedMap,buildEvidenceMap,legacy};
