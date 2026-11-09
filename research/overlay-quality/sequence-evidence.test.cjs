const test=require('node:test'),assert=require('node:assert/strict');
const {sequenceEvidence}=require('./sequence-evidence.cjs');
const ctx=(masks,times=masks.map((_,i)=>1000+i*100))=>({map:{keys:7},rows:masks.map((mask,i)=>({mask,t:times[i]}))});
test('ordered chord support distinguishes equal row/note histograms',()=>{
  const a=sequenceEvidence(ctx([3,4,48,64]),1000,1400),b=sequenceEvidence(ctx([3,48,4,64]),1000,1400);
  assert.equal(a[0],0);assert.equal(b[0],1/3);assert.notEqual(a[1],b[1]);
});
test('hand timing can vary despite perfectly steady global rows; mirror/rate invariant',()=>{
  const masks=[1,2,16,4,32,64,1,16,2,32,4,64];const a=sequenceEvidence(ctx(masks),1000,2200);
  assert.ok(a[2]>.1);assert.ok(a[3]>0);assert.ok(Math.abs(a[4])<1e-12);assert.equal(a[5],0);
  const mirror=x=>[...Array(7)].reduce((a,_,i)=>a|((x>>i&1)<<(6-i)),0);
  const b=sequenceEvidence(ctx(masks.map(mirror)),1000,2200);
  const c=sequenceEvidence(ctx(masks,masks.map((_,i)=>(1000+i*100)/1.5)),1000/1.5,2200/1.5);
  a.forEach((v,i)=>{assert.ok(Math.abs(v-b[i])<1e-12);assert.ok(Math.abs(v-c[i])<1e-12);});
});
