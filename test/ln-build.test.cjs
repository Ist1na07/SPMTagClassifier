const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {fixtures}=require('../research/rework/coarse_sanity.cjs'),{analyzeLnAdapter}=require('../research/rework/ln-adapter-direct.cjs');
const {parseOsu,prepare}=require('../src/parser.cjs'),{lnResult}=require('../src/ln-result.cjs'),profile=require('../src/models/native7k.json');
test('static anchored LN consumer matches real research code without Node globals',()=>{
 const box={module:{exports:{}}};vm.createContext(box,{codeGeneration:{strings:false,wasm:false}});
 vm.runInContext(fs.readFileSync('src/generated/ln-engine.cjs','utf8'),box);
 const cases=fixtures().filter(f=>f.K===7);
 for(const f of[cases[0],cases[10],cases.at(-1)])for(const speedRate of[1,1.5]){
  assert.equal(JSON.stringify(box.module.exports.analyzeLnAdapter(f.text,profile,{speedRate})),JSON.stringify(analyzeLnAdapter(f.text,profile,{speedRate})),f.id);
 }
});
test('LN public adapter distinguishes display support from archival confidence and original aggregation intervals',()=>{
 const f=fixtures().find(f=>f.K===7),map=parseOsu(f.text),r=lnResult(f.text,profile,{}, {map,ctx:prepare(map,10)});
 assert.equal(r.segments[0].start,map.firstNote);assert.equal(r.segments.at(-1).end,map.lastNote+1);
 assert.equal(r.modelInfo.defaultReplacementAccepted,false);
 for(const s of r.segments){
  assert.ok(Math.abs(Object.values(s.distribution).reduce((a,b)=>a+b)-1)<1e-10);
  assert.equal(s.confidence,s.distribution[s.tag]);assert.ok(s.aggregationWeight>=0&&Number.isFinite(s.aggregationWeight));
  assert.ok(Object.values(s.evidence.geometry.release).every(Number.isFinite));
 }
});
