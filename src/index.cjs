'use strict';
const legacy=require('./legacy/tag-engine.cjs');
const {parseOsu,prepare}=require('./parser.cjs');
const {features,windows,FEATURE_KEYS}=require('./features.cjs');
const {coarseResult}=require('./coarse-result.cjs'),{lnResult}=require('./ln-result.cjs');
const geometry=require('./generated/native-geometry.cjs'),{version}=require('../package.json');
const {correctCadenceDisplay}=require('./cadence.cjs');
const {refine:refineSegments}=require('./segment-readout.cjs');
const overlay7=require('./generated/overlay-engine.cjs'),overlayProfile=require('./models/overlay7k.json');
const profile4=require('./models/4k.json'),profile7=require('./models/7k.json');
const native4=require('./models/native4k.json'),native6=require('./models/native6k.json'),native7=require('./models/native7k.json');
const perception=require('./perception.cjs'),perceptionProfiles=require('./models/perception7k.json').profiles;
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
function describe(w) {
  const m=w.metrics;
  return {anchor:m.anchorShare,trill:m.trillFraction,roll:m.rollFraction,
    repetition:1-m.maskEntropy,occupancy:m.occupancy,releaseIndependence:m.releaseIndependence};
}
function stable7(osuText,opts,e) {
  const perceptual=opts.aggregation!=='classic';
  const name=opts.ratingEvidence?(opts.ratingEvidence.modelIdentity===perception.COMPACT_RATING?'compact':'detail'):'actions',p=perceptionProfiles[name];
  const r=overlay7.analyzeMap(osuText,{...opts,_rawEvidence:true,_segmentReadout:(raw,resolveBpm)=>refineSegments(e.ctx,raw,p.classes,{resolveBpm,speedRate:opts.speedRate??1,supportPower:p.supportPower??1})}),geometryContext=geometry.prepare(e.ctx);
  const classicPrior=r.debug.classicPrior,branchSupport=r.debug.readoutSupport,cutExclusions=r.debug.cutExclusions;
  delete r.debug.classicPrior;delete r.debug.readoutSupport;delete r.debug.cutExclusions;
  const raw=r.debug.rawSegments;delete r.debug.rawSegments;
  // The inherited segmenter can omit a short leading pickup when snapping.
  // Extend only the envelope; keep the learned tags and overall decision.
  if(r.segments.length){r.segments[0].start=r.map.firstNote;r.segments.at(-1).end=r.map.lastNote;}
  for(const s of r.segments) {
    const w=features(e.ctx,s.start,Math.max(s.start+1,s.end));
    const raw=s.scores??{[s.tag]:1},sum=Object.values(raw).reduce((a,b)=>a+b,0);
    const distribution=Object.fromEntries(Object.entries(raw).map(([k,v])=>[k,sum>0?v/sum:Number(k===s.tag)]));
    const ranked=Object.values(distribution).sort((a,b)=>b-a);
    s.distribution=distribution;
    const archivedEquivBpm=s._archivedEquivBpm;delete s._archivedEquivBpm;
    const cadence=s.evidence?.cadence;
    s.evidence={...(cadence?{cadence}:{}),...(archivedEquivBpm===undefined?{}:{archivedEquivBpm}),normalizedEntropy:ranked.length>1?-ranked.reduce((a,v)=>v>0?a+v*Math.log(v):a,0)/Math.log(ranked.length):0,
      margin:(ranked[0]||0)-(ranked[1]||0),effort:w.metrics.effort,occupancy:w.metrics.occupancy,
      featureWindows:1,contextNoteObservations:w.notes,legacyConfidence:s.confidence,
      geometry:geometry.describeInterval(geometryContext,s.start,Math.max(s.start+1,s.end))};
  }
  const total=Object.values(r.overall.weights).reduce((a,b)=>a+b,0);
  r.overall.composition=total>0?Object.fromEntries(Object.entries(r.overall.weights).map(([k,v])=>[k,v/total])):{break:1};
  r.overall.scoreKind='legacy decision scores; distribution is normalized affinity, not calibrated confidence';
  r.modelInfo={version,keyCount:7,variant:'legacy-stable',researchVariant:'anchored-ln-release',
    segmentEvidence:'archived affinities, regular-rest gate and two-thumb static-hand cut exclusion; stable cadence plateaus',temporalResolutionMs:250,
    segmentReadout:{stepMs:500,cutContextMs:4000,minCadencePlateauMs:1800,cutExcludedFrames:cutExclusions},
    overlayPolicy:{regularityRestRelativeNps:overlayProfile.gate.ratio,regularityRestRowRateMax:overlayProfile.gate.rowRateMax,
      rowRateUnit:'rows per converted second',aggregation:'unchanged legacy policy'}};
  if(perceptual){
    const rating=opts.ratingEvidence;
    const a=perception.infer(e.map,raw,rating,p,{speedRate:opts.speedRate??1,context:e.ctx,baseline:classicPrior??r.overall,_branchSupport:branchSupport});
    r.overall=a.overall;perception.decorateSegments(r.segments,a,p.classes);
    r.modelInfo.overlayPolicy.aggregation='support-weighted perceptual burden';
    r.modelInfo.perception={profile:name,recipe:{...p.recipe},supportPower:p.supportPower,
      family:{...p.family},policy:{...p.policy},ratingModelIdentity:p.modelIdentity};
  }
  r.descriptors=e.getWindows(4000).map(w=>({start:w.start,end:w.end,...describe(w)}));
  return r;
}
function analyzeMap(osuText,opts={}) {
  if(opts.aggregation!==undefined&&!['classic','perceptual'].includes(opts.aggregation))throw Error('UnknownAggregation: expected classic or perceptual');
  if((opts.aggregation==='perceptual'||opts.ratingEvidence)&&opts.algorithm&&opts.algorithm!=='stable')throw Error('PerceptionUnsupportedAlgorithm: requires stable 7K');
  if(opts.ratingEvidence&&opts.aggregation==='classic')throw Error('PerceptionUnusedRating: classic aggregation does not consume a difficulty curve');
  if(opts.algorithm==='legacy')return legacy.analyzeMap(osuText,opts);
  if(opts.algorithm && !['modern','stable'].includes(opts.algorithm))throw new Error('UnknownAlgorithm: expected modern, stable or legacy');
  const e=evidenceMap(osuText,opts),{map}=e;
  if(map.keys!==7&&(opts.aggregation==='perceptual'||opts.ratingEvidence))throw Error('PerceptionUnsupportedKeys: fitted profiles currently require native 7K');
  // Keep the mature LOCAL expert. The modern local adaptation remains opt-in;
  // stable 7K now applies the independently selected perceptual whole readout.
  if(map.keys===7){
    const r=opts.algorithm==='modern'?lnResult(osuText,native7,opts,e):stable7(osuText,opts,e);
    return correctCadenceDisplay(r,e.ctx,opts.speedRate??1);
  }
  return coarseResult(osuText,map.keys===4?native4:native6,opts);
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
module.exports={analyzeMap,extractIntermediate,buildCachedMap:legacy.buildCachedMap,buildEvidenceMap,ratingEvidenceFromSPM:perception.ratingEvidenceFromSPM,legacy};
