'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{test}=require('node:test');
const runtime=require('./coarse-runtime.cjs'),{composition}=require('./composition.cjs');
const {ROOT}=require('./data.cjs');
const profile=JSON.parse(fs.readFileSync(path.join(ROOT,'research/rework/results/coarse-4k-baseline-r1/weak4_only/0/outer-model.json'),'utf8'));
const template=()=>Object.fromEntries([...new Set([...profile.model.keys,'effort','occupancy','release:tailRatePerKey'])].map(k=>[k,0]));
const near=(a,b)=>a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<1e-10,`${v} != ${b[i]}`));
test('Break contributes no fine vote and coalescing preserves uncertainty/diversity',()=>{
 const metrics={...template(),effort:10},w={start:0,end:2000,metrics,notes:20,structurallyEmpty:false};
 const result=runtime.analyzeWindows([w,{...w,start:2000,end:4000},{...w,start:4000,end:600000,notes:0,metrics:template(),structurallyEmpty:true}],profile.model,6);
 const direct=composition(result.windows.map(w=>w.probability),result.windows.map(w=>w.aggregationWeight),{breakIndex:5});
 const merged=runtime.aggregateSegments(result.segments,[...profile.model.classes,'break']);
 near(direct.distribution,merged.distribution);assert.ok(Math.abs(direct.diversity-merged.diversity)<1e-12);
 assert.equal(result.segments.at(-1).tag,'break');assert.equal(result.segments.at(-1).aggregationWeight,0);
 const empty=runtime.analyzeWindows([{...w,metrics:template(),notes:0,structurallyEmpty:true}],profile.model,6);
 assert.equal(empty.coarse.dominant,null);assert.equal(Object.values(empty.coarse.composition).reduce((a,b)=>a+b,0),0);
});
test('held-only windows are physical activity and all keys remain native',()=>{
 for(const K of[4,6,7]){
  const objects=Array.from({length:K},(_,c)=>`${Math.floor((c+.5)*512/K)},192,0,128,0,10000:0:0:0:0:`);
  const text=`osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:${K}\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n${objects.join('\n')}`;
  const result=runtime.classifyCoarse(text,profile),held=result.windows.find(w=>w.contextStart>1000&&w.contextEnd<9000);
  assert.ok(held);assert.equal(held.notes,0);assert.equal(held.metrics.occupancy,1);assert.notEqual(held.tag,'break');
  assert.equal(result.modelInfo.nativeKeyCount,K);assert.equal(result.map.keys,K);
 }
});
test('speedRate converts real times and sixteenth-note equivalent BPM',()=>{
 const objects=Array.from({length:97},(_,i)=>`42,192,${i*125},1,0,0:0:0:0:`);
 const text=`osu file format v14\n[General]\nMode:3\n[Difficulty]\nCircleSize:6\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n${objects.join('\n')}`;
 const base=runtime.classifyCoarse(text,profile),fast=runtime.classifyCoarse(text,profile,{speedRate:1.5});
 assert.equal(base.map.lastNote,12000);assert.equal(fast.map.lastNote,8000);
 assert.equal(base.windows[0].equivBpm,120);assert.equal(fast.windows[0].equivBpm,180);
 assert.equal(fast.windows[0].metrics.rowBpm,180);assert.equal(fast.segments.at(-1).end,8001);
 for(const result of[base,fast])for(let i=1;i<result.segments.length;i++)assert.equal(result.segments[i-1].end,result.segments[i].start);
});
