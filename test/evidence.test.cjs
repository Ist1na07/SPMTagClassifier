const test=require('node:test'),assert=require('node:assert/strict');
const {scores}=require('../src/model.cjs');
const {aggregateEvidence,transferDistribution}=require('../src/aggregate.cjs');
const {segmentsFrom,decode}=require('../src/segment.cjs');
const {FEATURE_KEYS}=require('../src/features.cjs');
const window=(start,end,effort)=>({start,end,values:FEATURE_KEYS.map(k=>k==='effort'?effort:0),
 metrics:{effort,occupancy:0},notes:10,rows:5,equivBpm:120});
test('softmax scores are normalized and finite at extreme logits',()=>{
 const m={mean:[0],scale:[1],W:[[1000],[-1000]],b:[0,0]};const p=scores(m,[100]);
 assert.equal(p[0]+p[1],1);assert.ok(p.every(Number.isFinite));
});
test('confidence weighting uses the same denominator as its numerator',()=>{
 const p=aggregateEvidence([window(0,1000,1),window(1000,3000,2)],[[.9,.1],[.4,.6]],'full',true);
 assert.ok(Math.abs(p[0]+p[1]-1)<1e-12);assert.ok(Math.abs(p[0]-(.9*.9+.4*2*2*.6)/(.9+2*2*.6))<1e-12);
});
test('merged segments retain all windows evidence rather than the first confidence',()=>{
 const s=segmentsFrom([window(0,1000,1),window(1000,3000,2)],[[.9,.1],[.6,.4]],['speed','tech'],0);
 assert.equal(s.length,1);assert.equal(s[0].end,3000);assert.equal(s[0].confidence,.7);
});
test('Viterbi discourages isolated changes without inventing probabilities',()=>{
 assert.deepEqual(decode([[.9,.1],[.45,.55],[.9,.1]],3),[0,0,0]);
});
test('6K interpolation preserves normalized coarse mass and fine conditional structure',()=>{
 const classes=['chordjack','minijack','speed','break'],p=transferDistribution(classes,[.4,.2,.3,.1],['jack','speed'],[.2,.8],2/3);
 assert.ok(Math.abs(p.reduce((a,b)=>a+b)-1)<1e-12);assert.ok(Math.abs(p[0]/p[1]-2)<1e-12);
 assert.ok(p[2]>.3);assert.ok(Math.abs(p[3]-.1)<1e-12);
});
