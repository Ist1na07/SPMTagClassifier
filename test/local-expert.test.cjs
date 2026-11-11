'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),vm=require('node:vm');
const {compileOverlayEngine}=require('../scripts/build-overlay.cjs');
const {replaceOnce}=require('../research/rework/legacy-lab.cjs');
const profile=require('../src/models/local7k.json');
function load(localExpert){
 const source=replaceOnce(compileOverlayEngine({segmentReadout:true,localExpert}),'  return __toCommonJS(index_exports);','  Object.assign(index_exports,{testModels:{MODEL,model_generated_default}});\n  return __toCommonJS(index_exports);');
 const box={module:{exports:{}}};vm.runInNewContext(source,box,{codeGeneration:{strings:false,wasm:false}});return box.module.exports;
}
const chart='[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n'+Array.from({length:200},(_,i)=>`${(i%7+.5)*512/7},192,${1000+i*83},1,0,0:0:0:0:`).join('\n');
test('static expert contains the selected actual56D coefficients without mutating the reference predictor',()=>{
 const now=load(true),old=load(false),plain=x=>JSON.parse(JSON.stringify(x));
 assert.deepEqual(plain(now.testModels.MODEL),profile.models);
 assert.deepEqual(plain(now.testModels.model_generated_default),plain(old.testModels.MODEL));
 assert.notDeepEqual(plain(now.testModels.MODEL.rc.W),plain(old.testModels.MODEL.rc.W));
 assert.notDeepEqual(plain(now.testModels.MODEL.ln.W),plain(old.testModels.MODEL.ln.W));
});
test('classic explicit reference selection preserves the existing full local path at all supported rates',()=>{
 const now=load(true),old=load(false);
 for(const speedRate of [.75,1,1.5]){
  const expected=old.analyzeMap(chart,{speedRate,_rawEvidence:true});
  const actual=now.analyzeMap(chart,{speedRate,_rawEvidence:true,_archivedExpert:true});
  assert.equal(JSON.stringify(actual),JSON.stringify(expected));
 }
});
