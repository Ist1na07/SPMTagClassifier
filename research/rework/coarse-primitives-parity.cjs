'use strict';
const fs=require('node:fs'),{fixtures}=require('./coarse_sanity.cjs'),runtime=require('./coarse-runtime.cjs');
const profile=JSON.parse(fs.readFileSync('research/rework/results/coarse-4k-baseline-r1/weak4_only/0/outer-model.json','utf8')),model=profile.model;
const chosen=fixtures().filter(f=>(f.expected==='jack'&&f.bpm===120&&f.columns.length===f.K)||f.expected==='ln');
const output=[];
for(const f of chosen)for(const mode of['none','jack','ln','both']){
 const result=runtime.classifyCoarse(f.text,profile,{physicalSupport:mode});
 const windows=result.windows.map(w=>({start:w.start,end:w.end,contextStart:w.contextStart,contextEnd:w.contextEnd,
  x:w.values,base:w.baseValues,empty:w.structurallyEmpty,notes:w.notes,occupancy:w.metrics.occupancy,effort:w.metrics.effort}));
 output.push({mode,record:{uid:f.id,keys:f.K,dataset:'synthetic-definition',windows:{[model.config.windowMs]:windows}},
  expected:{local:result.windows.map(w=>w.probability),weights:result.windows.map(w=>w.aggregationWeight),composition:Object.values(result.coarse.composition),
   rules:result.windows.map(w=>w.structuralSupport?.tag??null)}});
}
fs.writeFileSync(process.argv[2],JSON.stringify({model,cases:output})+'\n');
