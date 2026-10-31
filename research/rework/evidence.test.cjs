'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {aggregateEvidence,coalesceForDisplay}=require('./evidence.cjs');
const {CLASSES}=require('./native.cjs');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-10,`${a} != ${b}`);
function segment(start,end,values,burden=1){
 const probability=CLASSES.map(k=>values[k]||0),tag=CLASSES[probability.indexOf(Math.max(...probability))];
 return {start,end,probability,tag,burden,confidence:Math.max(...probability),family:tag==='break'?'break':'rc'};
}
test('hard legacy consumer preserves all cached complete original overall decisions',()=>{
 const file='.cache/rework-legacy-full.json';if(!fs.existsSync(file))return;
 for(const r of JSON.parse(fs.readFileSync(file,'utf8'))){
  const result=aggregateEvidence(r.segments,{mode:'legacy',confidencePower:1});
  assert.deepEqual(result.tags,r.overall.tags,r.uid);assert.equal(result.sort,r.overall.sort,r.uid);
  for(const [k,w]of Object.entries(r.overall.weights))close(result.weights[k],w);
 }
});
test('weighted numerator and denominator share the same confidence factor',()=>{
 const rows=[segment(0,6000,{chordjack:1}),segment(6000,8000,{minijack:1}),segment(8000,10000,{'dense chordstream':1})].map(s=>({...s,confidence:.4}));
 assert.deepEqual(aggregateEvidence(rows,{mode:'legacy',confidencePower:1}).tags,['rc mix']);
 assert.deepEqual(aggregateEvidence(rows,{mode:'normalized',confidencePower:1}).tags,['chordjack']);
});
test('soft evidence and diversity survive subdivision and display coalescing',()=>{
 const a=[segment(0,1000,{chordjack:.8,minijack:.2},3),segment(1000,1500,{chordjack:.6,minijack:.4},1),segment(1500,3000,{minijack:1},2)];
 const split=[{...a[0],end:1},{...a[0],start:1},...a.slice(1)],options={mode:'soft',effortPower:1,confidencePower:1};
 const first=aggregateEvidence(a,options),second=aggregateEvidence(split,options),display=coalesceForDisplay(a,options),third=aggregateEvidence(display,options);
 assert.equal(display.length,2);assert.equal(display[0].components.length,2);
 for(const other of [second,third]){
  assert.deepEqual(first.tags,other.tags);close(first.diversity,other.diversity);
  first.distribution.forEach((v,k)=>close(v,other.distribution[k]));
 }
 display.forEach(s=>assert.equal(CLASSES[s.probability.indexOf(Math.max(...s.probability))],s.tag));
});
test('uncertain local mixtures are not evidence of three distinct temporal patterns',()=>{
 const p={chordjack:1/3,minijack:1/3,'dense chordstream':1/3};
 const uncertain=[segment(0,1000,p),segment(1000,2000,p),segment(2000,3000,p)];
 const distinct=[segment(0,1000,{chordjack:1}),segment(1000,2000,{minijack:1}),segment(2000,3000,{'dense chordstream':1})];
 const opts={mode:'soft',minimumDiversity:.1};
 assert.deepEqual(aggregateEvidence(uncertain,opts).tags,[]);
 assert.deepEqual(aggregateEvidence(distinct,opts).tags,['rc mix']);
 const empty=aggregateEvidence([segment(0,1000,{break:1})],opts);
 assert.deepEqual(empty.tags,[]);assert.equal(empty.activeWeight,0);
});
test('fractional fine support from displayed Break segments cannot dominate an active section',()=>{
 const rest=segment(0,1000000,{break:.6,chordjack:.4}),active=segment(1000000,1001000,{speed:1});
 assert.deepEqual(aggregateEvidence([rest],{mode:'soft'}).tags,[]);
 assert.deepEqual(aggregateEvidence([rest,active],{mode:'soft'}).tags,['speed']);
 const display=coalesceForDisplay([rest,{...rest,start:1000000,end:2000000}],{});
 assert.equal(display[0].aggregationWeight,0);assert.ok(display[0].probability.every(Number.isFinite));
});
test('Vibro does not supply the third ordinary type needed for Mix',()=>{
 const rows=[segment(0,4000,{chordjack:1}),segment(4000,7000,{speed:1}),segment(7000,10000,{vibro:1})];
 assert.deepEqual(aggregateEvidence(rows,{mode:'legacy'}).tags,['rc mix'],'faithful historical control');
 for(const mode of ['normalized','soft'])assert.ok(!aggregateEvidence(rows,{mode}).tags.includes('rc mix'));
 assert.deepEqual(aggregateEvidence(rows,{mode:'soft',excludeVibroFromMix:false}).tags,['rc mix']);
});
test('abstaining on LN fine subtype preserves the observed LN family',()=>{
 const row={...segment(0,10000,{'ln wildcard':1}),family:'ln'};
 const normalized=aggregateEvidence([row],{mode:'normalized'}),soft=aggregateEvidence([row],{mode:'soft'});
 for(const value of[normalized,soft]){assert.deepEqual(value.tags,[]);assert.equal(value.sort,'ln');}
 assert.equal(aggregateEvidence([row],{mode:'legacy'}).sort,'rc','exact historical control remains archived');
});
