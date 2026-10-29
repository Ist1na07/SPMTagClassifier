'use strict';
const fs=require('node:fs'),path=require('node:path');
const {parseOsu,prepare}=require('../src/parser.cjs');
const {windows}=require('../src/features.cjs');
const {scores,named}=require('../src/model.cjs');
const {pooledVector,localProbabilities,coarseSupports}=require('../src/aggregate.cjs');
const p4=require('../src/models/4k.json'),p7=require('../src/models/7k.json');
const splits={...require('./results/4k-split.json'),...require('./results/7k-split.json')};
function support(map,p) {
 const ws=windows(prepare(map,p.rowMergeMs??10),p.windowMs,2000),local=localProbabilities(ws,p);
 return scores(p.map,pooledVector(ws,p,local));
}
const out=fs.openSync('research/results/6k-projections.jsonl','w');let n=0;
for(const folder of ['spm_dataset-1.1.1','spm_dataset-4k-0.2.0']) {
 const records=JSON.parse(fs.readFileSync(path.join('dataset',folder,'labels.json'),'utf8'));
 records.forEach((r,i)=>{
  const uid=folder+'/'+r.id+'/'+i;if(!splits[uid])return;
  const map=parseOsu(fs.readFileSync(path.join('dataset',folder,r.file),'utf8')),sourceKeys=map.keys;
  let truth;
  if(sourceKeys===4)truth=[r.subset];
  else truth=Object.keys(coarseSupports(r.tags,r.tags.map(()=>1))).filter(c=>coarseSupports(r.tags,r.tags.map(()=>1))[c]>0);
  if(!truth.length)return; // Composite impression labels do not identify a unique projected pattern.
  const projected={...map,keys:6,notes:map.notes.filter(n=>sourceKeys!==7||n.col!==3).map(n=>({...n,
   col:sourceKeys===4?[0,2,3,5][n.col]:n.col>3?n.col-1:n.col}))};
  if(!projected.notes.length)return;
  const record={uid,sourceKeys,truth,...splits[uid],
   p4:named(p4.map.classes,support(projected,p4)),p7:coarseSupports(p7.map.classes,support(projected,p7))};
  const unadapted={...projected,keys:7};
  record.noKeyAdaptation={p4:named(p4.map.classes,support(unadapted,p4)),p7:coarseSupports(p7.map.classes,support(unadapted,p7))};
  fs.writeSync(out,JSON.stringify(record)+'\n');if(++n%100===0)console.log('projected',n);
 });
}
fs.closeSync(out);console.log('Projected source-labelled charts:',n);
