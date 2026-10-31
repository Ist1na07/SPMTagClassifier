'use strict';
const fs=require('node:fs'),path=require('node:path'),{ROOT}=require('./data.cjs'),{analyzeLnAdapter}=require('./ln-adapter-direct.cjs');
const atoms=rows=>rows.flatMap(r=>r.components?atoms(r.components):[r]);
const requests=JSON.parse(fs.readFileSync(process.argv[2],'utf8')),results=[];
for(const request of requests){
 const r=analyzeLnAdapter(fs.readFileSync(path.join(ROOT,request.source),'utf8'),request.profile);
 results.push({uid:request.uid,tags:r.overall.tags,shares:r.overall.shares,distribution:r.overall.distribution,diversity:r.overall.diversity,
  rows:atoms(r.segments).map(s=>({start:s.start,end:s.end,tag:s.tag,family:s.family,confidence:s.confidence,probability:s.probability,burden:s.burden}))});
}
fs.writeFileSync(process.argv[3],JSON.stringify(results)+'\n');
