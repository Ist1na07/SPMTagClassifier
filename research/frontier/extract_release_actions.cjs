'use strict';
const fs=require('node:fs'),path=require('node:path'),{parseOsu,prepare}=require('../../src/parser.cjs');
const {releaseActions}=require('./release-actions.cjs');
const records=fs.readFileSync('.cache/rework7.jsonl','utf8').trim().split('\n').map(JSON.parse);
const output=process.argv[2]||'.cache/frontier-release-actions-r1.jsonl';
const scales=process.argv[3]?process.argv[3].split(',').map(Number):[15,30,60,120];
if(fs.existsSync(output))throw Error('Use a new output path; do not overwrite completed evidence');
const fd=fs.openSync(output,'w');let count=0;
for(const r of records){
 const ctx=prepare(parseOsu(fs.readFileSync(r.source,'utf8'))),actions=r.segments.map(s=>releaseActions(ctx,s.start,s.end,scales));
 fs.writeSync(fd,JSON.stringify({uid:r.uid,source:r.source,intervals:r.segments.map(s=>[s.start,s.end]),
  values:actions.map(a=>a.continuous),largeScaleLimit:actions.map(a=>a.largeScaleLimit)})+'\n');
 if(++count%50===0)console.log('release actions',count);
}
fs.closeSync(fd);
fs.writeFileSync(output+'.schema.json',JSON.stringify({records:count,scalesMs:scales,
 features:['retainedPairsPerKeySecond','sameHandPairsPerKeySecond'],
 semantics:'per-tail retained-finger pairs, smoothly vanish at synchronized starts/ends; candidate physical inputs, no hard tags',
 partition:'exact same original7K raw intervals as rework7.jsonl',
 sourceSnapshots:Object.fromEntries(['extract_release_actions.cjs','release-actions.cjs'].map(n=>[n,fs.readFileSync(path.join(__dirname,n),'utf8')]))},null,2)+'\n');
console.log('complete',count);
