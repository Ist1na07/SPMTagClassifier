'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{handCuts,KEYS}=require('./cut-features.cjs');
const root=path.resolve(__dirname,'../..'),source=path.join(root,'.cache/overlay-bilateral-partitions-r1');
const output=path.join(root,'.cache/overlay-partition-hands-r1');fs.mkdirSync(output,{recursive:true});
assert.equal(JSON.parse(fs.readFileSync(path.join(source,'complete.json'),'utf8')).partitions,18);
const contexts=new Map();let count=0;
for(const variant of['original','physical'])for(let partition=0;partition<9;partition++){
 const file=path.join(output,`${variant}-${partition}.jsonl`);
 if(fs.existsSync(file+'.complete.json'))continue;
 const rows=fs.readFileSync(path.join(source,`${variant}-${partition}.jsonl`),'utf8').trim().split('\n').map(JSON.parse);
 const result=rows.map(r=>{
  if(!contexts.has(r.source))contexts.set(r.source,prepare(parseOsu(fs.readFileSync(path.join(root,r.source),'utf8'))));
  return{uid:r.uid,intervals:r.intervals,values:r.intervals.map(([a,b])=>handCuts(contexts.get(r.source),a,b).values)};
 });
 fs.writeFileSync(file+'.tmp',result.map(r=>JSON.stringify(r)).join('\n')+'\n');fs.renameSync(file+'.tmp',file);
 fs.writeFileSync(file+'.complete.json',JSON.stringify({records:result.length})+'\n');count++;
}
fs.writeFileSync(path.join(output,'complete.json'),JSON.stringify({keys:KEYS,partitions:18,newPartitions:count,predictorCalls:0,
 sourceSnapshot:fs.readFileSync(__filename,'utf8'),featureSnapshot:fs.readFileSync(path.join(__dirname,'cut-features.cjs'),'utf8')},null,2)+'\n');
console.log('Partition hand geometry complete',count);
