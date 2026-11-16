'use strict';
const fs=require('node:fs'),vm=require('node:vm'),z=require('node:zlib');
const {compileOverlayEngine}=require('../../scripts/build-overlay.cjs'),{replaceOnce}=require('../rework/legacy-lab.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{musicalSegments}=require('../../src/musical-segmentation.cjs');
let source=compileOverlayEngine({segmentReadout:true});source=replaceOnce(source,'  return __toCommonJS(index_exports);','  Object.assign(index_exports,{normalizeWindows});\n  return __toCommonJS(index_exports);');
const box={module:{exports:{}}};vm.runInNewContext(source,box,{codeGeneration:{strings:false,wasm:false}});
const records=z.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r2.jsonl.gz')).toString().trim().split('\n').map(JSON.parse),differences=[];
for(const r of records){const ctx=prepare(parseOsu(fs.readFileSync(r.source,'utf8'))),input={segments:r.cm.rawSegs,windows:r.cm.windows,norm:box.module.exports.normalizeWindows(r.cm.windows),speedRate:1};
 for(const radius of [.5,1,2]){const cuts={};for(const mode of ['nearest','contrast'])cuts[mode]=musicalSegments(ctx,input,{maxMoveBeats:radius,mode}).map(({start,end,windowRange})=>({start,end,windowRange}));if(JSON.stringify(cuts.nearest)!==JSON.stringify(cuts.contrast))differences.push({uid:r.uid,radius,cuts});}
}
fs.writeFileSync('research/action_clock/results/music-ordering-audit.json',JSON.stringify({scope:'All 411 original native parent/window inputs, three radii; no rating or classifier recomputation needed for candidate-order geometry.',inputs:records.length,comparisons:records.length*3,differences},null,2));console.log('ordering audit',records.length*3,'comparisons',differences.length,'different');
