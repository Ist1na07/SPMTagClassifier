'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {createProbe}=require('./activity_overlay_probe.cjs');
const {createReadoutProbe,smoothCadenceFrames}=require('./cadence_readout.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {fixture,adapter}=require('./cadence_merge_probe.cjs');
test('actual engine fine-frame readout preserves two known cadence regions',()=>{
  const text=fixture(),old=createProbe({},adapter)(text),r=createReadoutProbe({},adapter)(text);
  assert.equal(old.segments.length,1);assert.equal(old.segments[0].equivBpm,200);
  const at=t=>r.segments.find(s=>s.start<=t&&s.end>t);
  assert.equal(at(10000).equivBpm,150);assert.equal(at(30000).equivBpm,200);
  assert.deepEqual(r.overall,old.overall);assert.deepEqual(r.debug.rawSegments,old.debug.rawSegments);
  fs.writeFileSync('research/local_feedback/results/cadence-readout-repair.json',JSON.stringify({scope:'Fixed Speed labels, actual generated engine, old uniform-row meter reused before merging; classifier and aggregate unchanged',before:old.segments.map(({start,end,equivBpm})=>({start,end,equivBpm})),after:r.segments.map(({start,end,equivBpm,evidence})=>({start,end,equivBpm,evidence}))},null,2));
});
test('cadence readout uses post-rate clock, not timing point BPM guesses',()=>{
  const r=createReadoutProbe({},adapter)(fixture(),{speedRate:1.5}),at=t=>r.segments.find(s=>s.start<=t&&s.end>t);
  assert.equal(at(5000).equivBpm,225);assert.equal(at(20000).equivBpm,300);
});
test('a short reliable plateau cannot overwrite a long irregular fallback region',()=>{
  let text=fixture().split('[HitObjects]')[0]+'[HitObjects]\n',t=1000;
  for(let i=0;i<350;i++,t+=i<100?100:[53,91,167,73,119][i%5])text+=`36,192,${t},1,0,0:0:0:0:\n`;
  const ctx=prepare(parseOsu(text)),raw=[];
  for(let start=1000;start<t-200;start+=500)raw.push({start,end:Math.min(start+500,t-200),family:'rc',tag:'speed',equivBpm:200});
  const r=smoothCadenceFrames(raw,ctx);
  assert.equal(r.find(s=>s.start<=5000&&s.end>5000).equivBpm,150);
  assert.equal(r.find(s=>s.start<=25000&&s.end>25000).equivBpm,200);
});
