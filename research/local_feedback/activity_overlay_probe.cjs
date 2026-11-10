'use strict';
// An actual current-engine integration probe. No production/generated source
// is edited. If selected, these hooks must move into the maintained builder,
// not ship a dynamic-eval adapter or depend on research modules.
const {createRequire}=require('node:module');
const fs=require('node:fs'),path=require('node:path');
const {compileOverlayEngine}=require('../../scripts/build-overlay.cjs');
const {replaceOnce}=require('../rework/legacy-lab.cjs');
const {apply}=require('./activity_runtime.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {correctCadenceDisplay}=require('../../src/cadence.cjs');
const profiles=require('../../src/models/perception7k.json').profiles;
const aggregation=require('../../src/models/overlay7k.json').aggregation;
function compile(){
  let source=compileOverlayEngine();
  source=replaceOnce(source,'    const segments = smoothSegments(rawEvidence ? rawEvidence.map(s => ({...s, ...(s.scores ? {scores:{...s.scores}} : {})})) : classifySegments(cm, { scorer: "model" }));',`
    const priorOverall = rawEvidence ? aggregateOverall(smoothSegments(rawEvidence.map(s=>({...s}))), ${JSON.stringify(aggregation)}) : null;
    let activity = null;
    let visibleRaw = rawEvidence;
    if (opts._activityTransform) {
      const mapMedians = computeMapMedians(cm.windows);
      const medians = cm.rawSegs.map(s=>medianWindowFeatures(cm.windows,s.windowRange[0],s.windowRange[1],mapMedians));
      const candidates = medians.map((f,i)=>{
        const extras=cm.extras[i];
        return {rc:blendScores(modelScores("rc",f,extras,cm.cutDemandRef),scoreRc({f,extras,cutDemandRef:cm.cutDemandRef}),"model"),
          ln:blendScores(modelScores("ln",f,extras,cm.cutDemandRef),scoreLn({f,extras}),"model")};
      });
      activity = opts._activityTransform(rawEvidence,candidates);
      visibleRaw=activity.frames;
      for(const s of visibleRaw){
        const bpm=equivBpmFor(s.tag,medians[s.parent],cm.extras[s.parent]);
        if(bpm===undefined)delete s.equivBpm;else s.equivBpm=bpm;
      }
    }
    const segments = smoothSegments(visibleRaw ? visibleRaw.map(s=>({...s})) : classifySegments(cm,{scorer:"model"}));`);
  source=replaceOnce(source,'    if (rawEvidence) result.debug.rawSegments = rawEvidence;',
    '    if (rawEvidence) result.debug.rawSegments = visibleRaw;\n    result.debug.classicPrior=priorOverall;\n    result.debug.activity=activity;');
  const module={exports:{}};new Function('module','exports',source)(module,module.exports);return module.exports;
}
function perceptionProbe(){
  // Post-study compatibility: production now exposes the same internal
  // drop-only support input. Original report-protocol retains the old hook.
  const perception=require('../../src/perception.cjs');
  return {...perception,infer:(map,frames,rating,profile,options={})=>perception.infer(map,frames,rating,profile,{...options,_branchSupport:options._prototypeSupport})};
}
function createProbe(model,adapter=apply){
  const engine=compile(),perception=perceptionProbe();
  return function analyze(text,opts={}){
    const map=parseOsu(text,opts),ctx=prepare(map),name=opts.ratingEvidence?(opts.ratingEvidence.modelIdentity===perception.COMPACT_RATING?'compact':'detail'):'actions',profile=profiles[name];
    if(map.keys!==7)throw Error('Activity probe requires7K');
    const r=engine.analyzeMap(text,{...opts,_rawEvidence:true,_activityTransform:(raw,candidates)=>adapter(model,map,raw,candidates,profile.classes,ctx)});
    if(r.segments.length){r.segments[0].start=map.firstNote;r.segments.at(-1).end=map.lastNote;}
    for(const s of r.segments)s.evidence={sourceStart:s.start,sourceEnd:s.end};
    const inference=perception.infer(map,r.debug.rawSegments,opts.ratingEvidence,profile,{speedRate:opts.speedRate??1,context:ctx,baseline:r.debug.classicPrior,_prototypeSupport:r.debug.activity.support});
    r.overall=inference.overall;r.modelInfo={variant:'research-activity-probe'};perception.decorateSegments(r.segments,inference,profile.classes);
    return correctCadenceDisplay(r,ctx,opts.speedRate??1);
  };
}
module.exports={compile,createProbe};
