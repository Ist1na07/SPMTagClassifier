'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),api=require('../..');
const box={};vm.runInNewContext(fs.readFileSync('dist/tag-engine.js','utf8'),box);
const requests=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),results=[];
for(const r of requests){
 const text=fs.readFileSync(r.source,'utf8'),options={algorithm:r.keys===7?'modern':'stable'};
 const a=api.analyzeMap(text,options),b=box.TagClassifier.analyzeMap(text,options);
 assert.equal(JSON.stringify(a),JSON.stringify(b),r.uid+' browser/API');
 results.push({uid:r.uid,keys:a.map.keys,first:a.map.firstNote,last:a.map.lastNote,overall:a.overall,coarse:a.coarse,
  segments:a.segments.map(s=>({start:s.start,end:s.end,tag:s.tag,family:s.family,confidence:s.confidence,
   sourceStart:s.evidence.sourceStart,sourceEnd:s.evidence.sourceEnd,legacyConfidence:s.evidence.legacyConfidence,
   probability:s.probability,weight:s.aggregationWeight})),
  ...(a.map.keys===7?{}:{windows:a.debug.windows.map(w=>({start:w.start,end:w.end,probability:w.probability,weight:w.aggregationWeight}))})});
}
fs.writeFileSync(process.argv[3],JSON.stringify(results)+'\n');
