const test=require('node:test'),assert=require('node:assert/strict');
const {coarseResult}=require('../src/coarse-result.cjs');
const model={keys:['rowRate','occupancy'],classes:['jack','chordstream','speed','tech','ln'],mean:[0,0],scale:[1,1],
 theta:[.1,0,.2,0,.3,0,.4,0,0,1,0,0,0,0,0,.5,.5],clip:6,config:{windowMs:4000,burdenGate:.2},spec:{representation:'physical'}};
function chart(K,objects){return `[General]\nMode:3\n[Difficulty]\nCircleSize:${K}\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n${objects.join('\n')}`;}
test('coarse API adapter preserves one local distribution through display and overall',()=>{
 for(const K of[4,6]){
  const text=chart(K,Array.from({length:97},(_,i)=>`32,192,${i*125},1,0,0:0:0:0:`));
  const r=coarseResult(text,{model,postprocess:{physicalSupport:'both'}});
  assert.deepEqual(r.overall.tags,['jack']);assert.equal(r.overall.composition.jack,1);
  assert.equal(r.segments[0].start,r.map.firstNote);assert.equal(r.segments.at(-1).end,r.map.lastNote+1);
  for(const s of r.segments){assert.equal(s.tag,'jack');assert.equal(s.confidence,1);assert.equal(s.distribution.jack,1);assert.deepEqual(s.scores,s.distribution);}
  assert.ok(r.descriptors.every(d=>Object.values(d).every(Number.isFinite)));
 }
});
test('source-bucket stamina compatibility does not rename the internal morphology',()=>{
 const m={...model,theta:Array(17).fill(0)};m.theta[11]=10;
 const r=coarseResult(chart(4,['32,192,0,1,0,0:0:0:0:','160,192,125,1,0,0:0:0:0:']),{model:m});
 assert.equal(r.coarse.dominant,'chordstream');assert.deepEqual(r.overall.tags,['stamina']);
 assert.ok(r.overall.composition.stamina>0);assert.equal(r.overall.composition.chordstream,undefined);
 assert.equal(r.modelInfo.sortMeaning,'family of dominant coarse class');
});
