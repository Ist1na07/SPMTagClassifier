'use strict';
// Default 7K must preserve mature decisions on the complete actual corpus.
const fs=require('node:fs'),assert=require('node:assert/strict'),api=require('../..'),{records7}=require('./data.cjs');
const old=new Map(JSON.parse(fs.readFileSync('.cache/rework-legacy-full.json','utf8')).map(r=>[r.uid,r]));
let charts=0,segments=0,repairs=0;
for(const r of records7()){
 const actual=api.analyzeMap(fs.readFileSync(r.path,'utf8')),expected=old.get(r.uid);
 assert.deepEqual(actual.overall.tags,expected.overall.tags,r.uid);assert.deepEqual(actual.overall.weights,expected.overall.weights,r.uid);
 assert.equal(actual.overall.sort,expected.overall.sort,r.uid);assert.equal(actual.segments.length,expected.segments.length,r.uid);
 for(let i=0;i<actual.segments.length;i++){
  const a=actual.segments[i],b=expected.segments[i];
  for(const key of['tag','family','confidence','secondary','equivBpm'])assert.deepEqual(a[key],b[key],r.uid+' '+key);
  assert.equal(a.start,i===0?actual.map.firstNote:b.start);assert.equal(a.end,i===actual.segments.length-1?actual.map.lastNote:b.end);
  repairs+=Number(a.start!==b.start||a.end!==b.end);segments++;
  for(const values of [a.evidence.geometry.anchor,a.evidence.geometry.release])assert.ok(Object.values(values).every(Number.isFinite),r.uid+' new geometry');
 }
 charts++;
}
const result={scope:'default public7K vs complete archived cached outputs on actual corpus',charts,segments,defaultTagsFamiliesConfidenceSecondaryBpmOverall:'exact',
 displayEnvelopeRepairs:repairs,newGeometry:'all finite; additive diagnostics only',source:'research/rework/public_preservation.cjs'};
fs.mkdirSync('research/rework/results/public-default7-preservation-r1',{recursive:true});fs.writeFileSync('research/rework/results/public-default7-preservation-r1/REPORT.json',JSON.stringify(result,null,2)+'\n');console.log(result);
