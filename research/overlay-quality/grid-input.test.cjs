'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {GRID,gridPatch}=require('./grid-input.cjs'),{loadLab,context,inspectPrepared,replaceOnce}=require('../rework/legacy-lab.cjs');
function chart(){return '[General]\nMode:3\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n'+
  Array.from({length:200},(_,i)=>[0,2,4].map(c=>`${(c+.5)*512/7},192,${1000+i*50},1,0,0:0:0:0:`).join('\n')).join('\n');}
test('declared grid is a complete nonduplicate tolerance/floor Cartesian comparison',()=>{
  assert.equal(GRID.length,10);assert.equal(new Set(GRID.map(JSON.stringify)).size,10);
  const source=fs.readFileSync('src/legacy/tag-engine.cjs','utf8').replace(/\r\n/g,'\n');
  assert.equal(gridPatch(GRID[0]).apply(source,replaceOnce),source);
  assert.throws(()=>gridPatch({physicalFloor:true,relativeTolerance:.9}));
});
test('period-relative phase tolerance removes the 50ms-to75ms feature alias',()=>{
  const old=loadLab(),next=loadLab({patches:[gridPatch({physicalFloor:false,relativeTolerance:.15})]});
  const a=inspectPrepared(old,context(old.lab,chart())),b=inspectPrepared(next,context(next.lab,chart()));
  assert.equal(a.segments[0].f.equivBpm,200);assert.equal(b.segments[0].f.equivBpm,300);
});
test('row-resolution spacing floor permits fast physical periods rather than a333BPM ceiling',()=>{
  const api=loadLab({patches:[gridPatch({physicalFloor:true,relativeTolerance:.15})]});
  const view=inspectPrepared(api,context(api.lab,chart(),1.5));
  assert.ok(view.segments.some(s=>s.f.equivBpm>333.34));
});
