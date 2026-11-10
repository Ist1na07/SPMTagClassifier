'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const api=require('..'),{EXPECTED_RATING,COMPACT_RATING}=require('../src/perception.cjs'),{parseOsu}=require('../src/parser.cjs');
function chart(k=7){return `[General]\nMode:3\n[Difficulty]\nCircleSize:${k}\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n`+Array.from({length:160},(_,i)=>`${(i%k+.5)*512/k},192,${1001+i*83},1,0,0:0:0:0:`).join('\n');}
function rating(text,speedRate,modelIdentity){const map=parseOsu(text,{speedRate});const query=[...new Set(map.notes.map(n=>n.t))];return {schemaVersion:1,keyCount:7,speedRate,clock:'post-rate-ms',modelIdentity,query,demand:query.map(()=>2)};}
test('stable 7K integrates matching Detailed and Compact curves on native fractional-rate clock',()=>{
 const text=chart();for(const speedRate of [.75,1,1.5])for(const [id,name]of [[EXPECTED_RATING,'detail'],[COMPACT_RATING,'compact']]){
  const curve=rating(text,speedRate,id),before=structuredClone(curve),r=api.analyzeMap(text,{speedRate,ratingEvidence:curve});
  assert.equal(r.modelInfo.perception.profile,name);assert(r.overall.perception.difficultyAvailable);
  assert.deepEqual(curve,before);assert(!Object.hasOwn(r.debug,'rawSegments'));
 }
});
test('perceptual default preserves visible local decisions; branches are independent and weighted',()=>{
 const text=chart(),a=api.analyzeMap(text),b=api.analyzeMap(text,{aggregation:'classic'});
 assert.equal(a.modelInfo.perception.profile,'actions');assert(!a.overall.perception.difficultyAvailable);
 assert.deepEqual(a.segments.map(s=>[s.tag,s.confidence,s.start,s.end,s.equivBpm]),b.segments.map(s=>[s.tag,s.confidence,s.start,s.end,s.equivBpm]));
 assert(a.segments.every(s=>Object.values(s.branches).every(x=>x>=0&&x<=1)));
 assert(Math.abs(a.segments.reduce((a,s)=>a+s.aggregationWeight,0)-1)<1e-10);
});
test('explicit incompatible options and unidentified curves never silently fall back',()=>{
 const text=chart(),curve=rating(text,1,'custom');
 assert.throws(()=>api.analyzeMap(text,{ratingEvidence:curve}),/RatingModel/);
 assert.throws(()=>api.analyzeMap(text,{aggregation:'classic',ratingEvidence:curve}),/UnusedRating/);
 assert.throws(()=>api.analyzeMap(text,{aggregation:'perceptual',algorithm:'modern'}),/UnsupportedAlgorithm/);
 assert.throws(()=>api.analyzeMap(chart(4),{aggregation:'perceptual'}),/UnsupportedKeys/);
 assert.throws(()=>api.analyzeMap(text,{aggregation:'other'}),/UnknownAggregation/);
});
test('browser API has identical rated integration and rejects runtime code generation',()=>{
 const box={};vm.createContext(box,{codeGeneration:{strings:false,wasm:false}});vm.runInContext(fs.readFileSync('dist/tag-engine.js','utf8'),box);
 const text=chart(),opts={speedRate:1.5,ratingEvidence:rating(text,1.5,COMPACT_RATING)};
 assert.equal(JSON.stringify(box.TagClassifier.analyzeMap(text,opts)),JSON.stringify(api.analyzeMap(text,opts)));
});
