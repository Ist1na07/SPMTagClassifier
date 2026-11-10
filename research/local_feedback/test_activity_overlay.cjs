'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {createProbe,compile}=require('./activity_overlay_probe.cjs');
const stable=require('../../src/generated/overlay-engine.cjs');
const {apply}=require('./activity_runtime.cjs');
const {analyzeMap:product}=require('../../src/index.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const profiles=require('../../src/models/perception7k.json').profiles;
const physical=fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n').map(JSON.parse);
const experts=fs.readFileSync('.cache/local-feedback-routing-r3.jsonl','utf8').trim().split('\n').map(JSON.parse);
const candidate=require('./results/activity-feasibility-r6.json');
const analyze=createProbe(candidate.model);
test('research hook without adapter preserves actual engine',()=>{
  const engine=compile();for(const i of [0,348,380]){
    const text=fs.readFileSync(physical[i].source,'utf8'),a=engine.analyzeMap(text,{_rawEvidence:true}),b=stable.analyzeMap(text,{_rawEvidence:true});
    assert.deepEqual(a.segments,b.segments);assert.deepEqual(a.overall,b.overall);assert.deepEqual(a.debug.classicPrior,b.overall);assert.deepEqual(a.debug.rawSegments,b.debug.rawSegments);
  }
});
test('real actual-engine branch extraction and visible family consume native model',()=>{
  for(const uid of ['query/B01','query/B02','query/CTX-H',require('../frontier/annotations-r6.json').cases.G02.sourceUid]){
    const i=physical.findIndex(r=>r.uid===uid);assert(i>=0,uid);
    const input=physical[i],text=fs.readFileSync(input.source,'utf8'),r=analyze(text),map=parseOsu(text),ctx=prepare(map),expected=apply(candidate.model,map,input.frames,experts[i].candidates,profiles.actions.classes,ctx);
    assert.deepEqual(r.debug.activity.probabilities,expected.probabilities);assert.deepEqual(r.debug.activity.support,expected.support);
    assert.equal(r.debug.rawSegments.length,expected.frames.length);
    expected.frames.forEach((f,j)=>{for(const key of ['start','end','family','tag','secondary'])assert.equal(r.debug.rawSegments[j][key],f[key]);});
    const old=stable.analyzeMap(text);assert.deepEqual(r.debug.classicPrior,old.overall);
    if(uid==='query/B01')assert(r.segments.every(s=>s.tag==='break'));
    if(uid==='query/B02')assert(r.segments.some(s=>s.tag==='release'));
  }
});
test('ordinary existing product is still unchanged and probe handles DT physical clock',()=>{
  const row=physical.find(r=>r.uid==='query/B02'),text=fs.readFileSync(row.source,'utf8');const old=product(text);
  assert(old.segments.every(s=>s.tag==='break'));const r=analyze(text,{speedRate:1.5}),map=parseOsu(text,{speedRate:1.5});
  assert.equal(r.segments[0].start,map.firstNote);assert.equal(r.segments.at(-1).end,map.lastNote);
  assert(r.debug.activity.probabilities.every(p=>p.every(Number.isFinite)));
});
