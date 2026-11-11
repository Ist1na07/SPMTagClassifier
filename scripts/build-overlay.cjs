'use strict';
const fs=require('node:fs'),{compileRestEngine}=require('../research/overlay-quality/compile-rest-engine.cjs');
const {replaceOnce}=require('../research/rework/legacy-lab.cjs');
const profile=require('../src/models/overlay7k.json');
const localProfile=require('../src/models/local7k.json');
if(profile.schemaVersion!==1)throw Error('Unsupported overlay gate schema');
function compileOverlayEngine({segmentReadout=false,localExpert=segmentReadout,breakPressure=localExpert}={}){
if(localExpert&&!segmentReadout)throw Error('Retuned local expert requires its maintained readout');
let source=compileRestEngine(profile.gate,profile.aggregation);
// Internal evidence request reuses this exact classification pass. Smoothing
// mutates envelopes, so preserve the raw states by smoothing copies only.
source=replaceOnce(source,'    const segments = smoothSegments(classifySegments(cm, { scorer: "model" }));',
  '    const rawEvidence = opts._rawEvidence ? classifySegments(cm, { scorer: "model" }) : null;\n'+
  '    const segments = smoothSegments(rawEvidence ? rawEvidence.map(s => ({...s, ...(s.scores ? {scores:{...s.scores}} : {})})) : classifySegments(cm, { scorer: "model" }));');
source=replaceOnce(source,'    return { map, segments, overall, debug: { windows: cm.windows, norm: void 0, ctx: void 0 } };',
  '    const result = { map, segments, overall, debug: { windows: cm.windows, norm: void 0, ctx: void 0 } };\n'+
  '    if (rawEvidence) result.debug.rawSegments = rawEvidence;\n    return result;');
if(segmentReadout){
  // LN Wildcard's deliberately low prototype affinity is not a probability
  // of inactivity. Explicit early rests and shortSparse remain unchanged.
  source=replaceOnce(source,'const easyAmbiguous = seg.confidence < 0.5 &&',
    'const easyAmbiguous = seg.tag !== "ln wildcard" && seg.confidence < 0.5 &&');
  const before='    const segments = smoothSegments(rawEvidence ? rawEvidence.map(s => ({...s, ...(s.scores ? {scores:{...s.scores}} : {})})) : classifySegments(cm, { scorer: "model" }));';
  source=replaceOnce(source,before,`
    const originalSegments = smoothSegments(rawEvidence ? rawEvidence.map(s=>({...s})) : classifySegments(cm,{scorer:"model"}));
    const classicPrior = aggregateOverall(originalSegments, ${JSON.stringify(profile.aggregation)});
    let readout = null;
    if (opts._segmentReadout) {
      const mapMedians=computeMapMedians(cm.windows),medians=new Map();
      readout=opts._segmentReadout(rawEvidence,(tag,parent)=>{
        if(!medians.has(parent)){const s=cm.rawSegs[parent];medians.set(parent,medianWindowFeatures(cm.windows,s.windowRange[0],s.windowRange[1],mapMedians));}
        return equivBpmFor(tag,medians.get(parent),cm.extras[parent]);
      });
    }
    const segments = readout ? readout.segments : originalSegments;`);
  source=replaceOnce(source,'    if (rawEvidence) result.debug.rawSegments = rawEvidence;',
    '    if (rawEvidence) result.debug.rawSegments = readout ? readout.frames : rawEvidence;\n    if(readout){result.debug.classicPrior=classicPrior;result.debug.readoutSupport=readout.support;result.debug.cutExclusions=readout.exclusions;}');
}
if(localExpert){
  if(localProfile.schemaVersion!==1)throw Error('Unsupported local expert schema');
  for(const f of ['rc','ln']){
    const m=localProfile.models[f];
    if(m.keys.length!==56||m.mean.length!==56||m.std.length!==56||m.W.length!==56*m.classes.length||m.b.length!==m.classes.length||m.scoreGain!==1.4||![...m.mean,...m.std,...m.W,...m.b].every(Number.isFinite)||m.std.some(s=>s<=0))throw Error('Invalid fixed-gain local expert: '+f);
  }
  source=replaceOnce(source,'  var MODEL = model_generated_default;',`  var MODEL = ${JSON.stringify(localProfile.models)};`);
  source=replaceOnce(source,'function modelScores(fam, f, extras, cutDemandRef) {','function modelScores(fam, f, extras, cutDemandRef, expert = MODEL) {');
  source=replaceOnce(source,'    const m = MODEL[fam];','    const m = expert[fam];');
  for(const f of ['rc','ln'])source=replaceOnce(source,`modelScores("${f}", f, extras, cm.cutDemandRef)`,`modelScores("${f}", f, extras, cm.cutDemandRef, opts._expertModel)`);
  // classic is the existing escape hatch; it keeps the archived local expert.
  for(const call of ['classifySegments(cm, { scorer: "model" })','classifySegments(cm,{scorer:"model"})'])source=replaceOnce(source,call,'classifySegments(cm,{scorer:"model",_expertModel:opts._archivedExpert?model_generated_default:undefined})');
  // The integral profiles and joint loss use this frozen reference predictor.
  // Do not inadvertently change both the fitted local branch and its prior.
  source=replaceOnce(source,`    const classicPrior = aggregateOverall(originalSegments, ${JSON.stringify(profile.aggregation)});`,
    `    const classicPrior = aggregateOverall(smoothSegments(classifySegments(cm,{scorer:"model",_expertModel:model_generated_default})), ${JSON.stringify(profile.aggregation)});`);
}
if(breakPressure){
  const currentCall='classifySegments(cm,{scorer:"model",_expertModel:opts._archivedExpert?model_generated_default:undefined})';
  if(source.split(currentCall).length!==3)throw Error('Expected both local classification entry points');
  source=source.split(currentCall).join('classifySegments(cm,{scorer:"model",_expertModel:opts._archivedExpert?model_generated_default:undefined,_breakPolicy:opts._archivedExpert?undefined:opts._breakPolicy})');
  source=replaceOnce(source,'  function classifySegments(cm, opts = {}) {',
    '  function classifySegments(cm, opts = {}) {\n    const breakPolicy=opts._breakPolicy, lowNpsScale=breakPolicy?.lowNpsScale??1;');
  source=replaceOnce(source,'f.nps < 18.5 && ex.segIvLogStd','f.nps < 18.5 * lowNpsScale && ex.segIvLogStd');
  source=replaceOnce(source,'f.nps < 22.5 && ex.segIvLogStd','f.nps < 22.5 * lowNpsScale && ex.segIvLogStd');
  source=replaceOnce(source,'if (lowStakes) {','if (lowStakes && !breakPolicy?.protects?.(s.start,s.end,"low")) {');
  source=replaceOnce(source,'seg.confidence < 0.5 && diffs[i]','seg.confidence < (breakPolicy?.confidenceFloor??0.5) && diffs[i]');
  source=replaceOnce(source,'if (easyAmbiguous || shortSparse) {','if ((easyAmbiguous && !breakPolicy?.protects?.(seg.start,seg.end,"late")) || shortSparse) {');
}
return source;
}
if(require.main===module){
fs.mkdirSync('src/generated',{recursive:true});
const source=compileOverlayEngine({segmentReadout:true});
fs.writeFileSync('src/generated/overlay-engine.cjs','/* Generated by node scripts/build-overlay.cjs; MIT; archive unchanged. */\n'+source);
console.log('Built src/generated/overlay-engine.cjs: retuned56D local expert, fixed reference prior and selected rest gate');
}
module.exports={compileOverlayEngine};
