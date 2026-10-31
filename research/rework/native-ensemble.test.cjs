'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {CLASSES}=require('./native.cjs');
const {combineSegments}=require('./native-ensemble.cjs');
const row=(start,end,tag)=>({start,end,tag,family:'rc',confidence:1,probability:CLASSES.map(k=>k===tag?1:0),burden:2});
test('reflection group average is exactly view-order invariant on the native union partition',()=>{
 const a=[row(0,1000,'chordjack')],b=[row(0,500,'speed'),row(500,1000,'chordjack')];
 const ab=combineSegments(a,b),ba=combineSegments(b,a);assert.deepEqual(ab,ba);
 assert.equal(ab.length,2);assert.equal(ab[0].viewDisagreement,1);assert.equal(ab[1].viewDisagreement,0);
 assert.deepEqual(ab.map(s=>[s.start,s.end]),[[0,500],[500,1000]]);
});
test('missing native intervals are not hallucinated as a pattern or a rest',()=>{
 const result=combineSegments([row(10,20,'speed')],[row(30,40,'tech')]);
 assert.deepEqual(result.map(s=>[s.start,s.end,s.tag,s.availableViews]),[[10,20,'speed',1],[30,40,'tech',1]]);
});
