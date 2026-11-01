const test=require('node:test'),assert=require('node:assert/strict');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{releaseActions}=require('./release-actions.cjs');
function chart(K,stagger=false,mirror=false,shift=0,step=150){
 const cs=Array.from({length:K},(_,c)=>c).filter(c=>K!==7||c!==3);
 const offsets=cs.map((_,i)=>(i-(cs.length-1)/2)*step);
 return `[General]\nMode:3\n[Difficulty]\nCircleSize:${K}\n[HitObjects]\n`+cs.map((c,i)=>
  `${((mirror?K-1-c:c)+.5)*512/K},192,${1000+shift},128,0,${4000+shift+(stagger?offsets[i]:0)}:0:0:0:0:`).join('\n');
}
for(const K of[4,6,7])test(K+'K distinguishes full simultaneous release from retained-finger release without calling either a tag',()=>{
 const measure=(stagger,mirror=false,shift=0)=>releaseActions(prepare(parseOsu(chart(K,stagger,mirror,shift))),1000+shift,5000+shift).metrics;
 const a=measure(false),b=measure(true);assert.equal(a.retainedFingerPairsPerKeySecond,0);assert.ok(b.retainedFingerPairsPerKeySecond>0);
 assert.equal(a.allHeldReleasedGroupFraction,1);assert.ok(b.allHeldReleasedGroupFraction<1);
 assert.deepEqual(b,measure(true,true));assert.deepEqual(b,measure(true,false,2000));
});
test('continuous retained-pair input has no simultaneous-release jitter cliff',()=>{
 const get=step=>releaseActions(prepare(parseOsu(chart(7,true,false,0,step))),1000,5000).continuous;
 const exact=get(0),tiny=get(1e-6),large=get(150);
 for(let i=0;i<exact.length;i++){
  assert.equal(exact[i].retainedPairsPerKeySecond,0);
  assert.ok(tiny[i].retainedPairsPerKeySecond<1e-6);
  assert.ok(large[i].retainedPairsPerKeySecond>tiny[i].retainedPairsPerKeySecond);
 }
});
test('rescaled large-tau shape converges to the analytic limit',()=>{
 const ctx=prepare(parseOsu(chart(7,true))),tau=1e8;
 const a=releaseActions(ctx,1000,5000,[tau]);
 for(const key of ['retainedPairsPerKeySecond','sameHandPairsPerKeySecond']){
  const observed=a.continuous[0][key]*tau*tau/1e6,limit=a.largeScaleLimit[key];
  assert.ok(Math.abs(observed/limit-1)<5e-5);
 }
 const exact=releaseActions(prepare(parseOsu(chart(7,false))),1000,5000);
 assert.equal(exact.largeScaleLimit.retainedPairsPerKeySecond,0);
});
