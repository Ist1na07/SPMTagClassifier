'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {parseOsu,prepare}=require('../src/parser.cjs');
const {jackCadence,switchCadence,describeActions}=require('../src/action-clock.cjs');
const {measureCadence,correctCadenceDisplay}=require('../src/cadence.cjs');
function context(events,rate=1,mirror=false){
 const text='[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n0,500,4,1,0,100,1,0\n[HitObjects]\n'+events.flatMap(([t,cols])=>cols.map(c=>`${Math.floor(((mirror?6-c:c)+.5)*512/7)},192,${Math.round(t)},1,0,0:0:0:0:`)).join('\n');
 return prepare(parseOsu(text,{speedRate:rate}),10/rate);
}
const stream=(step,masks,count=80)=>Array.from({length:count},(_,i)=>[1000+i*step,masks[i%masks.length]]);

test('same events have per-hand Jack75 but global substitution stream150; type determines clock',()=>{
 for(const rate of [.75,1,1.5])for(const mirror of [false,true]){
  const ctx=context(stream(100,[[0,1],[4,5],[1,2],[5,6]]),rate,mirror),a=ctx.map.firstNote,b=ctx.map.lastNote+1/rate;
  assert(Math.abs(jackCadence(ctx,a,b,'chordjack',rate).equivBpm-75*rate)<=.5);
  assert(Math.abs(switchCadence(ctx,a,b,rate).equivBpm-150*rate)<=.5);
 }
});
test('global single fillers do not double coactive hand strikes; actual faster strikes stay faster',()=>{
 for(const step of [50,75,100,150]){
  const base=stream(step,[[0,1,4],[1,2,5,6]]),fill=stream(step,[[0],[4],[1],[6]],79).map(([t,c])=>[t+step/2,c]);
  const ctx=context([...base,...fill].sort((a,b)=>a[0]-b[0]));
  assert.equal(jackCadence(ctx,1000,ctx.map.lastNote+1,'chordjack').equivBpm,15000/step);
 }
});
test('same-finger Minijack clock survives other-hand interleaving but rejects a finger roll',()=>{
 const ctx=context([...stream(100,[[0]]),...stream(100,[[4],[5],[6]]).map(([t,c])=>[t+50,c])].sort((a,b)=>a[0]-b[0]));
 assert.equal(jackCadence(ctx,1000,ctx.map.lastNote+1,'minijack').equivBpm,150);
 const roll=context(stream(50,[[0],[1],[2]]));
 assert.equal(jackCadence(roll,1000,roll.map.lastNote+1,'minijack'),null);
 assert.equal(jackCadence(roll,1000,roll.map.lastNote+1,'chordjack'),null);
});
test('fixed hand grips do not create a substitution clock',()=>{
 const ctx=context(stream(100,[[0,1],[4,5]]));assert.equal(switchCadence(ctx,1000,ctx.map.lastNote+1),null);
 const description=describeActions(ctx,1000,ctx.map.lastNote+1);
 for(const view of description.thumbAssignments)for(const hand of view.hands){assert.equal(hand.substitutionFraction,0);assert.equal(hand.fixedGripFraction,1);}
});
test('a very short faster hand cannot determine the sustained Jack tempo',()=>{
 const ctx=context([...stream(50,[[0,1]],4),...stream(200,[[4,5]])].sort((a,b)=>a[0]-b[0]));
 assert.equal(jackCadence(ctx,1000,ctx.map.lastNote+1,'chordjack').equivBpm,75);
});
test('mixed unsupported rhythms omit a scalar BPM instead of retaining an archived grid guess',()=>{
 let t=1000;const ctx=context([0,83,106,171,229,310,449,157,191,337,277].map(dt=>[t+=dt,[0,1]]));
 for(const tag of ['chordjack','minijack','speed','tech','wildcard']){
  const result={segments:[{start:1000,end:ctx.map.lastNote+1,tag,equivBpm:263,evidence:{}}],modelInfo:{}};
  correctCadenceDisplay(result,ctx);assert.equal(result.segments[0].equivBpm,undefined);assert(result.segments[0].evidence.cadenceUnresolved);
 }
});
test('a mixed-type display may carry separate primary and secondary action BPMs',()=>{
 const ctx=context(stream(100,[[0,1],[4,5],[1,2],[5,6]]));
 const result={segments:[{start:1000,end:ctx.map.lastNote+1,tag:'chordjack',secondary:'dense chordstream',equivBpm:150,evidence:{}}],modelInfo:{}};
 correctCadenceDisplay(result,ctx);assert.equal(result.segments[0].equivBpm,75);assert.equal(result.segments[0].evidence.secondaryCadence.equivBpm,150);
 assert.equal(measureCadence(ctx,1000,ctx.map.lastNote+1,'release'),null);
});


test('sparse chord accents inside a finger roll do not invent a slower Jack octave',()=>{
 const events=Array.from({length:120},(_,i)=>[1000+i*50,i%4===0?[0,1]:[i%3]]),ctx=context(events);
 assert.equal(jackCadence(ctx,1000,ctx.map.lastNote+1,'chordjack'),null);
});


test('coactive Jack backbone can coexist with faster common-finger Anchor retriggers',()=>{
 const ctx=context(stream(50,[[0,1],[0]])),a=ctx.map.firstNote,b=ctx.map.lastNote+1;
 assert.equal(jackCadence(ctx,a,b,'chordjack').equivBpm,150);
 assert.equal(jackCadence(ctx,a,b,'minijack').equivBpm,300);
});

test('secondary cadence survives even when the primary action clock is unresolved',()=>{
 const ctx=context(stream(50,[[0],[1],[2]]));
 const result={segments:[{start:1000,end:ctx.map.lastNote+1,tag:'minijack',secondary:'speed',equivBpm:263,evidence:{}}],modelInfo:{}};
 correctCadenceDisplay(result,ctx);assert.equal(result.segments[0].equivBpm,undefined);
 assert.equal(result.segments[0].evidence.secondaryCadence.equivBpm,300);
});
