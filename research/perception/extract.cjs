'use strict';
// New natural v1.2.2 frames; no annotation boundaries enter extraction.
const fs=require('node:fs');
const {loadLab,inspect}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {features}=require('../../src/features.cjs');
const profile=require('../../src/models/overlay7k.json');
const output='.cache/perception-inputs-r1.jsonl';
if(fs.existsSync(output))throw Error('Preserve existing input; choose a new revision');
const records=fs.readFileSync('.cache/rework7.jsonl','utf8').trim().split('\n').map(JSON.parse);
for(const [uid,source] of [
 ['CTX-H','artifacts/frontier-annotation-r2/CTX-H.osu'],['CTX-L','artifacts/frontier-annotation-r2/CTX-L.osu'],
 ['LN-H','artifacts/frontier-annotation-r3/LN-H.osu'],['LN-L','artifacts/frontier-annotation-r3/LN-L.osu'],
 ['B01','artifacts/frontier-annotation-r1/B01.osu'],['B02','artifacts/frontier-annotation-r1/B02.osu'],
 ['C16','artifacts/frontier-annotation-r2/C16.osu'],['C24','artifacts/frontier-annotation-r2/C24.osu'],
 ['F01','artifacts/frontier-annotation-r5/F01.osu'],['F02','artifacts/frontier-annotation-r5/F02.osu']
])records.push({uid:'query/'+uid,source});
const derived=JSON.parse(fs.readFileSync('research/frontier/annotations-r10.json','utf8')).derivedWholeCharts.cases;
records.push(...Object.values(derived));
const api=loadLab({patches:[actionRatePatch(profile.gate)]});let n=0,frames=0;
const schema={id:'natural-v122-perception-r1',gate:profile.gate,
 sourceSnapshot:fs.readFileSync(__filename,'utf8'),
 description:'Natural raw frames before smoothing; all source endpoints unchanged, no forced local intervals; physical notes use public parser duplicate-head contract.'};
fs.writeFileSync(output+'.schema.json',JSON.stringify(schema,null,2)+'\n');
for(const record of records){
 const text=fs.readFileSync(record.source,'utf8'),ctx=prepare(parseOsu(text)),r=inspect(api,text);
 const fs0=r.classified.map(s=>({...s,scores:s.scores??{[s.tag]:1},secondary:s.secondary??null}));
 const geometry=fs0.map(s=>features(ctx,s.start,Math.max(s.start+1,s.end)).metrics);
 const row={uid:record.uid,source:record.source,map:ctx.map,frames:fs0,geometry,
  legacyOverall:api.lab.aggregateOverall(r.smoothed,profile.aggregation)};
 fs.appendFileSync(output,JSON.stringify(row)+'\n');frames+=fs0.length;
 if(++n%50===0)console.log('perception inputs',n,frames);
}
fs.writeFileSync(output+'.complete.json',JSON.stringify({records:n,frames})+'\n');
console.log('complete',n,frames);
