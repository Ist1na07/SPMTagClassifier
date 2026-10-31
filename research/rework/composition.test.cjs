'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {composition}=require('./composition.cjs');
const close=(a,b)=>assert.ok(Math.abs(a-b)<1e-12,`${a} != ${b}`);
test('same composition entropy, opposite temporal heterogeneity',()=>{
 const uncertain=composition([[.5,.5],[.5,.5]],[1,1]);
 const diverse=composition([[1,0],[0,1]],[1,1]);
 assert.deepEqual(uncertain.distribution,diverse.distribution);
 close(uncertain.entropy,1);close(diverse.entropy,1);
 close(uncertain.diversity,0);close(diverse.diversity,1);
});
test('Break evidence is excluded and non-Break mass weights conditional distributions',()=>{
 const result=composition([[.5,0,.5],[0,1,0],[0,0,1]],[2,1,30],{breakIndex:2});
 assert.deepEqual(result.distribution,[.5,.5]);close(result.activeWeight,2);close(result.diversity,1);
 const empty=composition([[0,0,1]],[1],{breakIndex:2});assert.equal(empty.activeWeight,0);assert.deepEqual(empty.distribution,[0,0]);
});
test('uniform evidence rescaling and equal-distribution subdivision preserve diversity',()=>{
 const a=composition([[.8,.2],[.1,.9]],[3,1]);
 const b=composition([[.8,.2],[.8,.2],[.1,.9]],[30,30,20]);
 for(const k of ['entropy','meanLocalEntropy','diversity'])close(a[k],b[k]);
 for(let i=0;i<2;i++)close(a.distribution[i],b.distribution[i]);
});
test('reject raw affinities and negative/nonfinite evidence',()=>{
 assert.throws(()=>composition([[1.4,.6]],[1]),/InvalidProbability/);
 for(const w of [-1,Infinity,NaN])assert.throws(()=>composition([[.5,.5]],[w]),/InvalidWeight/);
});
