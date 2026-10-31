'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const {test}=require('node:test'),assert=require('node:assert/strict');
const {ROOT,records7}=require('./data.cjs'),{analyzeNative}=require('./native-direct.cjs'),{evaluateCachedNative}=require('./native-policy.cjs');
const root=path.join(ROOT,'research/rework/results/native-metric-r1'),protocol=JSON.parse(fs.readFileSync(path.join(root,'protocol.json'),'utf8'));
const choices=JSON.parse(fs.readFileSync(path.join(root,'metric_repaired/selection.json'),'utf8')).selections;
const rows=records7(),manual=rows.filter(r=>r.dataset==='manual'),weak=rows.filter(r=>r.dataset!=='manual');
function cached(r,c){
 let folder='.cache/rework-native-r1';
 if(protocol.sourceCatalogue[c.source])folder=protocol.sourceCatalogue[c.source].root;
 return JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(ROOT,folder,c.source,r.uid.replaceAll('/','__')+'.json.gz'))));
}
test('fresh single-choice caller reproduces actual cached policy on all selected outer configurations',()=>{
 let cases=0;
 for(const choice of choices)for(const r of[manual[choice.outer],weak[choice.outer*53]]){
  const c=choice.config,profile={config:c,sourceParameters:protocol.sourceParameters[c.source]};
  const text=fs.readFileSync(r.path,'utf8'),direct=analyzeNative(text,profile),expected=evaluateCachedNative(cached(r,c),c);
  assert.deepEqual(direct.overall,expected.overall,`${r.uid} overall`);
  assert.deepEqual(direct.segments,expected.segments,`${r.uid} segments`);
  assert.equal(direct.segments[0].start,direct.map.firstNote);assert.equal(direct.segments.at(-1).end,direct.map.lastNote);cases++;
 }
 assert.equal(cases,12);
});
