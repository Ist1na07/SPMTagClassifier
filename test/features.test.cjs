const test=require('node:test'),assert=require('node:assert/strict');
const {parseOsu,prepare}=require('../src/parser.cjs');
const {features,windows}=require('../src/features.cjs');
const header=k=>`[General]\nMode:3\n[Difficulty]\nCircleSize:${k}\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n`;
const line=(c,t,end=null,k=7)=>`${(c+.5)*512/k},192,${t},${end===null?1:128},0,${end===null?'0:0:0:0:':end+':0:0:0:0:'}`;
test('overlapping LN intervals have bounded union occupancy',()=>{
 const map=parseOsu(header(7)+[line(0,0,2000),line(0,1000,3000)].join('\n'));
 const f=features(prepare(map),0,3000);assert.equal(f.metrics.occupancy,1/7);assert.ok(f.metrics.occupancy<=1);
});
test('touching tail/head needs a release and is not pre-existing held support',()=>{
 const map=parseOsu(header(7)+[line(0,0,1000),line(0,1000,2000)].join('\n'));
 const ctx=prepare(map);assert.equal(ctx.holds[0].length,2);
 assert.equal(features(ctx,1000,1100).metrics.heldAtHead,0);
});
test('tail-only events never sever consecutive head adjacency',()=>{
 const text=header(7)+[line(0,0,150),line(1,0),line(0,250),line(1,250)].join('\n');
 const f=features(prepare(parseOsu(text)),0,500);assert.equal(f.metrics.overlap,1);assert.equal(f.rows,2);
});
test('held intervals crossing both boundaries contribute once',()=>{
 const map=parseOsu(header(4)+line(0,0,3000,4));
 assert.equal(features(prepare(map),1000,2000).metrics.occupancy,.25);
});
test('timing phase resets at the current uninherited timing point',()=>{
 const text=header(4).replace('0,500,4,2,1,100,1,0','0,500,4,2,1,100,1,0\n1033,600,4,2,1,100,1,0')+
 [1033,1183,1333,1483].map((t,i)=>line(i,t,null,4)).join('\n');
 assert.equal(features(prepare(parseOsu(text)),1033,1600).metrics.gridResidual,0);
});
test('duplicate taps remain taps and parser rejects nonfinite rates',()=>{
 const text=header(4)+[line(1,1000,null,4),line(1,1000,null,4)].join('\n');
 assert.equal(parseOsu(text).notes.length,1);assert.equal(parseOsu(text).notes[0].lnEnd,null);
 for(const speedRate of [0,-1,Infinity,NaN])assert.throws(()=>parseOsu(text,{speedRate}),RangeError);
});
test('window support includes a hold-only tail and is a contiguous partition',()=>{
 const ctx=prepare(parseOsu(header(6)+line(0,500,10000,6))),ws=windows(ctx,4000,2000);
 assert.equal(ws[0].start,500);assert.equal(ws.at(-1).end,10001);
 for(let i=1;i<ws.length;i++)assert.equal(ws[i-1].end,ws[i].start);
 assert.ok(ws.every(w=>w.values.every(Number.isFinite)));
});
test('nonfinite timing points are ignored with an explicit diagnostic',()=>{
 const text=header(6).replace('[HitObjects]','1000,Infinity,4,2,1,100,1,0\n[HitObjects]')+line(0,2000,null,6);
 const m=parseOsu(text);assert.equal(m.timingPoints.length,1);assert.deepEqual(m.warnings,['IgnoredNonfiniteTimingPoint']);
});
