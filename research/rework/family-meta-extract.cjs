'use strict';
// Additional, label-free inputs needed to refit the ORIGINAL two-family heads.
// Existing vectors are reused; their alignment is checked while extracting the
// genuinely missing rule/gating inputs. No fixed prior is used as a fit target.
const fs=require('node:fs'),assert=require('node:assert/strict');
const {records7}=require('./data.cjs');
const {loadLab,inspect}=require('./legacy-lab.cjs');
const patches=require('./patches.cjs');
const api=loadLab(),fixed=loadLab({patches:patches.logical.concat(patches.releaseIsolation)});
const paths=new Map(records7().map(r=>[r.uid,r.path]));
const records=fs.readFileSync('.cache/rework7.jsonl','utf8').trim().split('\n').map(JSON.parse);
const fd=fs.openSync('.cache/rework-family-meta.jsonl','w');let count=0;
function compact(view){return view.segments.map(s=>({family0:s.family0,windowRange:s.windowRange,
 gates:{nps:s.f.nps,lnWeightRatio:s.f.lnWeightRatio,lockRatio:s.f.lockRatio,heldCoverage:s.f.heldCoverage,
  chordMean:s.f.chordMean,jackSpeedBpm:s.f.jackSpeedBpm,segIvLogStd:s.extra.segIvLogStd,
  vibroFraction:s.extra.vibroFraction,segRhythmChangeRate:s.extra.segRhythmChangeRate,rowCount:s.extra.rowCount},
 ruleRc:s.ruleRc,ruleLn:s.ruleLn}));}
for(const r of records){
 const text=fs.readFileSync(paths.get(r.uid),'utf8'),base=inspect(api,text),corrected=inspect(fixed,text,{intervals:base.segments});
 assert.deepEqual(base.segments.map(s=>[s.start,s.end,s.vector]),r.segments.map(s=>[s.start,s.end,s.original]),r.uid+' original alignment');
 assert.deepEqual(corrected.segments.map(s=>s.vector),r.segments.map(s=>s.corrected),r.uid+' corrected alignment');
 fs.writeSync(fd,JSON.stringify({uid:r.uid,original:compact(base),corrected:compact(corrected)})+'\n');
 if(++count%50===0)console.log('original-family fit metadata',count);
}
fs.closeSync(fd);console.log('original-family fit metadata complete',count);
