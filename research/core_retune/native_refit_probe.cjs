'use strict';
// In-memory research replacement of only MODEL and its explicit family gains.
// Uses the current maintained builder + unmodified public entry point/readout.
// This dynamic adapter is never bundled or installed as production code.
const fs=require('node:fs'),path=require('node:path'),{createRequire}=require('node:module');
const {compileOverlayEngine}=require('../../scripts/build-overlay.cjs');
const {replaceOnce}=require('../rework/legacy-lab.cjs');
function createProbe(models,{preservePrior=false}={}){
 let source=compileOverlayEngine({segmentReadout:true,localExpert:false});
 if(preservePrior){
  source=replaceOnce(source,'function modelScores(fam, f, extras, cutDemandRef) {','function modelScores(fam, f, extras, cutDemandRef, expert = MODEL) {');
  source=replaceOnce(source,'    const m = MODEL[fam];','    const m = expert[fam];');
  for(const family of ['rc','ln'])source=replaceOnce(source,`modelScores("${family}", f, extras, cm.cutDemandRef)`,`modelScores("${family}", f, extras, cm.cutDemandRef, opts._expertModel)`);
  source=replaceOnce(source,'function blendScores(mdl, rule, mode) {','function blendScores(mdl, rule, mode, historicalExpert = false) {');
  const call='scores = blendScores(mdl, rule, scorer);';if(source.split(call).length!==3)throw Error('Expected both expert blends');
  source=source.split(call).join('scores = blendScores(mdl, rule, scorer, !!opts._expertModel);');
  source=replaceOnce(source,`    const classicPrior = aggregateOverall(originalSegments, ${JSON.stringify(require('../../src/models/overlay7k.json').aggregation)});`,
   `    const classicPrior = aggregateOverall(smoothSegments(classifySegments(cm,{scorer:"model",_expertModel:model_generated_default})), ${JSON.stringify(require('../../src/models/overlay7k.json').aggregation)});`);
 }
 source=replaceOnce(source,'  return __toCommonJS(index_exports);',
  `  MODEL = ${JSON.stringify(models)};\n  return __toCommonJS(index_exports);`);
 const rc=models.rc.scoreGain??1.4,ln=models.ln.scoreGain??1.4;
 source=replaceOnce(source,'if (mode === "model") out[k] = (mdl[k] ?? 0) * 1.4;',
  `if (mode === "model") out[k] = (mdl[k] ?? 0) * (${preservePrior?'historicalExpert ? 1.4 : ':''}Object.prototype.hasOwnProperty.call(mdl,"release") ? ${ln} : ${rc});`);
 const engineModule={exports:{}};new Function('module','exports',source)(engineModule,engineModule.exports);
 const file=path.resolve(__dirname,'../../src/index.cjs'),nativeRequire=createRequire(file),module={exports:{}};
 new Function('require','module','exports',fs.readFileSync(file,'utf8'))(p=>p==='./generated/overlay-engine.cjs'?engineModule.exports:nativeRequire(p),module,module.exports);
 return module.exports;
}
module.exports={createProbe};
