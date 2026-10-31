'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {releaseContext,releaseGeometry,KEYS}=require('./release-geometry.cjs');
const {extract,CHANNELS}=require('./mechanics-probe.cjs');
function chart(K,notes,rate=1,mirror=false){
 const text='[General]\nMode:3\n[Difficulty]\nCircleSize:'+K+'\nOverallDifficulty:6\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n'+notes.map(([t,c,e])=>
  `${Math.floor(((mirror?K-1-c:c)+.5)*512/K)},192,${t},${e?128:1},0,${e?e+':0:0:0:0':'0:0:0:0:'}`).join('\n');
 return {text,ctx:releaseContext(prepare(parseOsu(text,{speedRate:rate})))};
}
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
test('isolated tail between two heads remains observable even when sampled Rbar is zero',
 {skip:!fs.existsSync('.upstream/spm_algorithm.js')},()=>{
 const {text,ctx}=chart(7,[[0,0,500],[1000,2]]),m=releaseGeometry(ctx,0,1000).metrics;
 close(m.headIsolation,.8);assert.equal(m.tailAsynchrony,1);
 const r=extract(text,[{start:0,end:1000}])[0],j=CHANNELS.indexOf('Rbar')*3;
 assert.deepEqual(r.slice(j,j+3),[0,0,0]);
});
test('simultaneous release/head has no head isolation or same-finger recovery gap',()=>{
 const {ctx}=chart(7,[[0,0,500],[500,0],[1000,2]]),m=releaseGeometry(ctx,0,1000).metrics;
 assert.equal(m.headIsolation,0);assert.equal(m.sameFingerRecovery,0);
});
test('release-only sequence is not erased when no head falls between its tail events',
 {skip:!fs.existsSync('.upstream/spm_algorithm.js')},()=>{
 const {text,ctx}=chart(7,[[0,0,1000],[0,2,1125],[0,1,1250],[2000,6]]);
 const m=releaseGeometry(ctx,0,2000).metrics,r=extract(text,[{start:0,end:2000}])[0],j=CHANNELS.indexOf('Rbar')*3;
 assert.ok(m.headIsolation>.8);assert.ok(m.sameHandOrder>0);assert.deepEqual(r.slice(j,j+3),[0,0,0]);
});
test('native active fingers, not a vacant embedded thumb, define release load and width',()=>{
 for(const K of [4,6,7]){
  const notes=Array.from({length:K},(_,c)=>[0,c,500]).concat([[1000,0]]),{ctx}=chart(K,notes);
  const m=releaseGeometry(ctx,0,1000).metrics;
  assert.equal(m.heldOtherAtTail,1);assert.equal(m.releaseChordFraction,1);assert.equal(m.tailAsynchrony,0);
 }
});
test('geometry is mirror invariant and timing ratios survive rate conversion in 4K, 6K and 7K',()=>{
 for(const K of [4,6,7]){
  const notes=[[0,0,500],[0,1,650],[0,2,575],[800,K-1],[1000,1,1600],[1200,0],[1800,K-2]];
  const a=releaseGeometry(chart(K,notes).ctx,0,1900),b=releaseGeometry(chart(K,notes,1,true).ctx,0,1900),c=releaseGeometry(chart(K,notes,2).ctx,0,950);
  a.values.forEach((v,j)=>close(v,b.values[j]));close(c.values[0],2*a.values[0]);
  a.values.slice(1).forEach((v,j)=>close(v,c.values[j+1]));
 }
});
test('same-hand order excludes tied release times and flexible centre assignments',()=>{
 const ordered=releaseGeometry(chart(7,[[0,0,500],[0,2,600],[0,1,700],[1000,6]]).ctx,0,1000).metrics;
 assert.ok(ordered.sameHandOrder>0);
 for(const notes of [[[0,0,500],[0,2,500],[0,1,700],[1000,6]],[[0,2,500],[0,3,600],[0,1,700],[1000,6]]]){
  assert.equal(releaseGeometry(chart(7,notes).ctx,0,1000).metrics.sameHandOrder,0);
 }
});
test('a chord at either endpoint cannot acquire an order from its sorted last/first column',()=>{
 for(const K of[6,7]){
  const ordered=[[0,0,1000],[0,2,1125],[0,1,1250],[2000,K-1]];
  assert.ok(releaseGeometry(chart(K,ordered).ctx,0,2000).metrics.sameHandOrder>0);
  for(const tail of[1000,1125,1250]){
   const notes=[...ordered,[0,K-1,tail]];
   const a=releaseGeometry(chart(K,notes).ctx,0,2000),b=releaseGeometry(chart(K,notes,1,true).ctx,0,2000);
   assert.equal(a.metrics.sameHandOrder,0);assert.equal(b.metrics.sameHandOrder,0);
   a.values.forEach((v,j)=>close(v,b.values[j]));
  }
 }
});
