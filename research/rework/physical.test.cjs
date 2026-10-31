'use strict';
const {test}=require('node:test'),assert=require('node:assert/strict');
const {prepare}=require('../../src/parser.cjs'),physical=require('./physical.cjs');
function context(K,notes){return physical.prepare(prepare({keys:K,notes,timingPoints:[{t:0,beatLength:500}],firstNote:0,
 lastNote:notes.reduce((a,n)=>Math.max(a,n.lnEnd??n.t),0)},0));}
const close=(a,b)=>a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-10,`${i}: ${x} != ${b[i]}`));
test('rotating overlapping chords do not constitute one continuous finger run',()=>{
 const masks=[3,6,12,9,3,6,12,9],notes=[];
 masks.forEach((m,i)=>{for(let c=0;c<4;c++)if(m&(1<<c))notes.push({t:i*125,col:c,lnEnd:null});});
 const f=physical.features(context(4,notes),0,1000);
 assert.equal(f.metrics.repeatRun,1);assert.equal(f.metrics.runCoverage,.25);assert.equal(f.metrics.peakExcess,0);
});
test('all real keys, including unused keys, participate in occupancy',()=>{
 for(const K of[4,6,7]){
  const notes=Array.from({length:K},(_,col)=>({t:0,col,lnEnd:1000}));
  let f=physical.features(context(K,notes),0,1000);
  assert.equal(f.metrics.occupancy,1);assert.equal(f.metrics.minOccupancy,1);
  assert.equal(f.metrics['release:releaseChordFraction'],0); // half-open tail at end
  f=physical.features(context(K,notes.slice(0,4)),0,1000);
  assert.equal(f.metrics.occupancy,4/K);assert.equal(f.metrics.minOccupancy,K===4?1:0);
 }
});
test('all physical descriptors are mirror invariant at 4K, 6K and 7K',()=>{
 for(const K of[4,6,7]){
  const notes=Array.from({length:36},(_,i)=>({t:i*125,col:(i*i+3*i)%K,lnEnd:i%3===0?i*125+290:null}));
  const mirrored=notes.map(n=>({...n,col:K-1-n.col}));
  close(physical.features(context(K,notes),0,5000).values,physical.features(context(K,mirrored),0,5000).values);
 }
});
test('context windows partition display time and keep headless held activity',()=>{
 const ctx=context(6,[{t:0,col:0,lnEnd:10000},{t:12000,col:4,lnEnd:null}]),rows=physical.windows(ctx,2000);
 assert.equal(rows[0].start,0);assert.equal(rows.at(-1).end,12001);
 for(let i=1;i<rows.length;i++)assert.equal(rows[i-1].end,rows[i].start);
 const held=rows.find(r=>r.contextStart===4000);
 assert.equal(held.notes,0);assert.equal(held.structurallyEmpty,false);assert.equal(held.metrics.sustain,0);
 assert.equal(held.metrics.occupancy,1/6);assert.equal(new Set(physical.KEYS).size,physical.KEYS.length);
});
