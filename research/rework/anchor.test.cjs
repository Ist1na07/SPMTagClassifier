'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),{describe}=require('./anchor.cjs');
test('a rotating shared-column chain is not a sustained single-finger anchor',()=>{
 const rotating=[3,6,12,9,3,6,12,9].map((mask,i)=>({t:i*125,mask}));
 const anchored=[3,5,9,3,5,9,3,5].map((mask,i)=>({t:i*125,mask}));
 const a=describe(rotating,4),b=describe(anchored,4);
 // Both have overlapping consecutive rows; only the latter retains one lane.
 assert.ok(b[1]>a[1]);assert.ok(b[3]>a[3]);assert.ok(b[6]>a[6]);
});
test('full chords are balanced jacks, not a dominant-finger anchor',()=>{
 const result=describe(Array.from({length:16},(_,i)=>({t:i*100,mask:127})),7);
 assert.equal(result[1],0);assert.equal(result[2],0);assert.equal(result[3],0);assert.equal(result[6],0);
});
test('anchor morphology is invariant to mirror and uniform time scaling',()=>{
 const rows=[3,5,9,3,5,9,3,5].map((mask,i)=>({t:i*125,mask}));
 const mirror=mask=>Array.from({length:4},(_,i)=>mask&(1<<i)?1<<(3-i):0).reduce((a,b)=>a|b,0);
 const a=describe(rows,4),b=describe(rows.map(r=>({t:r.t/1.5,mask:mirror(r.mask)})),4);
 a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-12));
});
test('ties in finger frequency must not introduce left/right bias',()=>{
 const rows=[[0,9],[100,9],[200,9],[300,1],[700,8]].map(([t,mask])=>({t,mask}));
 const mirror=mask=>Array.from({length:4},(_,i)=>mask&(1<<i)?1<<(3-i):0).reduce((a,b)=>a|b,0);
 const a=describe(rows,4),b=describe(rows.map(r=>({...r,mask:mirror(r.mask)})),4);
 a.forEach((x,i)=>assert.ok(Math.abs(x-b[i])<1e-12));
});
