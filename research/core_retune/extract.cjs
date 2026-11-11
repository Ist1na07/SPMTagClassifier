'use strict';
// Research only. Natural intervals and original expert inputs, no label cuts.
const fs=require('node:fs'),zlib=require('node:zlib');
const {loadLab,inspect}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const OUT='.cache/core-retune-parents-r1.jsonl.gz';
if(fs.existsSync(OUT))throw Error('Preserve existing core input');
const api=loadLab({patches:[actionRatePatch(profile.gate)]}),lab=api.lab;
const rows=fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n').map(JSON.parse);
const result=[];let n=0,parents=0,maxScoreError=0;
for(const row of rows){
 const r=inspect(api,fs.readFileSync(row.source,'utf8'));
 if(r.classified.length!==row.frames.length)throw Error('Interval count '+row.uid);
 for(let j=0;j<r.classified.length;j++){
  const a=r.classified[j],b=row.frames[j];
  if(a.start!==b.start||a.end!==b.end||a.tag!==b.tag||a.family!==b.family)throw Error('Native baseline '+row.uid+'/'+j);
  if(a.scores)for(const[k,v]of Object.entries(a.scores))maxScoreError=Math.max(maxScoreError,Math.abs(v-b.scores[k]));
 }
 result.push({uid:row.uid,source:row.source,cutDemandRef:r.cutDemandRef,cm:r.cache,
  parents:r.segments.map((s,j)=>({start:s.start,end:s.end,f:s.f,extra:s.extra,vector:s.vector,
   modelRc:s.modelRc,modelLn:s.modelLn,ruleRc:s.ruleRc,ruleLn:s.ruleLn,baseline:r.classified[j]}))});
 parents+=r.classified.length;if(++n%50===0)console.log('core natural parents',n,parents);
}
fs.writeFileSync(OUT,zlib.gzipSync(result.map(r=>JSON.stringify(r)).join('\n')+'\n'));
fs.writeFileSync(OUT+'.schema.json',JSON.stringify({records:n,parents,maxScoreError,keys:lab.VEC_KEYS,model:lab.MODEL,
 rcDefaults:lab.RC_DEFAULTS,lnDefaults:lab.LN_DEFAULTS,config:lab.CONFIG,gate:profile.gate,
 source:fs.readFileSync(__filename,'utf8'),classifier:lab.classifySegments.toString(),blend:lab.blendScores.toString()},null,2)+'\n');
console.log('complete',n,parents,'max native baseline score error',maxScoreError);
