'use strict';
// Inspect existing entry points, not a fitted candidate and not an accuracy test.
const fs=require('node:fs'),path=require('node:path'),api=require('../..');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {features}=require('../../src/features.cjs');
const {loadLab,inspect}=require('../rework/legacy-lab.cjs');
const lab=loadLab(),folder='artifacts/frontier-annotation-r2';
const manifest=JSON.parse(fs.readFileSync(path.join(folder,'manifest.json'),'utf8'));
const records=[];
for(const [id,description]of Object.entries(manifest.cases)){
 const text=fs.readFileSync(path.join(folder,id+'.osu'),'utf8'),map=parseOsu(text),ctx=prepare(map,10);
 const [start,end]=description.labelInterval;
 const outputs={};
 for(const algorithm of ['stable','modern']){
  const r=api.analyzeMap(text,{algorithm});
  outputs[algorithm]={overall:r.overall,segments:r.segments.map(s=>({start:s.start,end:s.end,tag:s.tag,confidence:s.confidence}))};
 }
 const r=inspect(lab,text,{intervals:[{start,end}]}),s=r.segments[0];
 records.push({id,targetInterval:[start,end],targetHeads:map.notes.filter(n=>start<=n.t&&n.t<end).map(n=>[n.t,n.col]),
  targetMetrics:features(ctx,start,end).metrics,outputs,
  inheritedTarget:{scope:'forced single target interval diagnosis, NOT automatic segmentation or public path',preMark:s.family0,classified:r.classified.map(s=>({tag:s.tag,family:s.family})),
   headRate:s.extra.headRate,medianWindowNps:s.f.nps}});
}
const root=process.argv[2]||'research/frontier/results/density-probe-r2';fs.mkdirSync(root,{recursive:true});
fs.writeFileSync(path.join(root,'tag-probe.json'),JSON.stringify({scope:'human conceptual contrasts; existing classifiers only, no accuracy estimate',records,
 sourceSnapshots:{'probe_density.cjs':fs.readFileSync(__filename,'utf8')}},null,2)+'\n');
for(const r of records)console.log(r.id,'stable',r.outputs.stable.overall.tags,'modern',r.outputs.modern.overall.tags,'target preMark',r.inheritedTarget.preMark);
