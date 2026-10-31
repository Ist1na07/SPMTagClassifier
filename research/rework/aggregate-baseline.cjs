'use strict';
// Cheap, interpretable comparator: retune original aggregation and the
// numerator/denominator correction, preserving the COMPLETE original head.
const fs=require('node:fs');
const {records7}=require('./data.cjs'),{loadLab}=require('./legacy-lab.cjs'),{weightedShares}=require('./patches.cjs');
const original=loadLab(),normalized=loadLab({patches:[weightedShares]});
const grid=[];
for(const dominantShare of [.4,.52,.65])for(const mixShare of [.04,.07,.12])for(const secondRel of [.6,.75,.9])grid.push({dominantShare,mixShare,secondRel});
const file='.cache/rework-legacy-full.json';
let records;
if(fs.existsSync(file))records=JSON.parse(fs.readFileSync(file,'utf8'));
else{
 records=[];
 for(const r of records7()){
  const result=original.analyzeMap(fs.readFileSync(r.path,'utf8'));
  records.push({uid:r.uid,segments:result.segments,overall:result.overall});
 }
 fs.writeFileSync(file,JSON.stringify(records)+'\n');
}
const rows=records.map(r=>({uid:r.uid,original:r.overall.tags,
 normalizedDefault:normalized.lab.aggregateOverall(r.segments).tags,
 originalGrid:grid.map(p=>original.lab.aggregateOverall(r.segments,p).tags),
 normalizedGrid:grid.map(p=>normalized.lab.aggregateOverall(r.segments,p).tags)}));
fs.writeFileSync('.cache/rework-aggregate-grid.json',JSON.stringify({grid,rows})+'\n');
console.log('computed aggregation treatments',rows.length);
