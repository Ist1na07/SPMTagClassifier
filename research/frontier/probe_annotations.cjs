'use strict';
const fs=require('node:fs'),path=require('node:path'),api=require('../..');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const geometry=require('../rework/release-geometry.cjs'),physical=require('../rework/physical.cjs');
const coarse=require('../rework/coarse-runtime.cjs');
const native6=require('../../src/models/native6k.json');
const {releaseActions}=require('./release-actions.cjs');
const root=process.argv[2]||'research/frontier/results/annotation-probe-r1';
const records=[];
for(const id of ['A01','A02','B01','B02']){
 const text=fs.readFileSync(`artifacts/frontier-annotation-r1/${id}.osu`,'utf8');
 const rates=id==='A01'?[1,2,3,4,6]:[1];
 for(const speedRate of rates){
  const map=parseOsu(text,{speedRate}),ctx=geometry.releaseContext(prepare(map,10));
  const outputs={};
  for(const algorithm of ['stable','modern']){
   const r=api.analyzeMap(text,{speedRate,algorithm});
   outputs[algorithm]={overall:r.overall,segments:r.segments.map(s=>({start:s.start,end:s.end,tag:s.tag,confidence:s.confidence,
    ...(s.evidence?.geometry?{geometry:s.evidence.geometry}:{})}))};
  }
  const plain=coarse.classifyCoarse(text,native6,{speedRate,physicalSupport:'none'});
  const supported=coarse.classifyCoarse(text,native6,{speedRate,physicalSupport:'both'});
  const f=physical.features(ctx,map.firstNote,map.lastNote+1);
  records.push({id,speedRate,keys:map.keys,heads:map.notes.length,
   noteRate:map.notes.length/((map.lastNote-map.firstNote+1)/1000),outputs,
   experimentalSharedCoarse7:{scope:'shared 4K/7K morphology head, not public 7K fine classification',
    none:plain.coarse,both:supported.coarse},
   physicalMetrics:f.metrics,releaseGeometry:geometry.releaseGeometry(ctx,map.firstNote,map.lastNote+1).metrics,
   releaseActions:releaseActions(ctx,map.firstNote,map.lastNote+1)});
 }
}
fs.mkdirSync(root,{recursive:true});
fs.writeFileSync(path.join(root,'tag-probe.json'),JSON.stringify({scope:'diagnosis on user-annotated conceptual contrasts; not accuracy estimation',records},null,2)+'\n');
for(const r of records)console.log(r.id,r.speedRate,'stable',r.outputs.stable.overall.tags,'modern',r.outputs.modern.overall.tags,'coarse',r.experimentalSharedCoarse7.both.dominant);
