'use strict';
// Batch the real parse -> native feature -> local head -> composition caller.
const fs=require('node:fs'),path=require('node:path');
const {ROOT}=require('./data.cjs'),runtime=require('./coarse-runtime.cjs');
const requests=JSON.parse(fs.readFileSync(process.argv[2],'utf8'));
const results=requests.map(r=>{
 const value=runtime.classifyCoarse(fs.readFileSync(path.join(ROOT,r.source),'utf8'),r.model,r.options||{});
 const merged=runtime.aggregateSegments(value.segments,[...r.model.classes,'break']);
 return {uid:r.uid,keys:value.map.keys,composition:Object.values(value.coarse.composition),
  mergedComposition:merged.distribution,diversity:value.coarse.evidence.diversity,mergedDiversity:merged.diversity,
  dominant:value.coarse.dominant,mass:value.coarse.evidence.activeWeight,
  windows:value.windows.map(w=>({start:w.start,end:w.end,contextStart:w.contextStart,contextEnd:w.contextEnd,
   tag:w.tag,probability:w.probability,weight:w.aggregationWeight,empty:w.structurallyEmpty,structuralSupport:w.structuralSupport?.tag??null,
   features:r.model.keys.map(k=>w.metrics[k])})),
  intervals:value.segments.map(s=>({start:s.start,end:s.end,tag:s.tag}))};
});
fs.writeFileSync(process.argv[3],JSON.stringify(results)+'\n');
