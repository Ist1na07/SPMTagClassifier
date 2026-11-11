'use strict';
// Original 56-dimensional expert input on label-independent 500ms clock.
// Measure features in centered 1/2/4s physical contexts, classify that evidence,
// then remap predictions onto nonoverlapping readout cells before smoothing.
const fs=require('node:fs'),zlib=require('node:zlib');
const {loadLab,context}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const api=loadLab({patches:[actionRatePatch(profile.gate)]}),lab=api.lab;
const early=loadLab({patches:[actionRatePatch(profile.gate),{apply(s,r){return r(s,'if (easyAmbiguous || shortSparse) {','if (false && (easyAmbiguous || shortSparse)) {');}}]}).lab;
const manifest=JSON.parse(fs.readFileSync('.cache/local-feedback-activity-r6b/manifest.json','utf8'));
const wanted=new Set(manifest.manualUids);
for(const rnd of ['r4','r5','r6'])for(const a of Object.values(JSON.parse(fs.readFileSync('research/frontier/annotations-'+rnd+'.json')).cases))wanted.add(a.sourceUid);
for(const a of Object.values(JSON.parse(fs.readFileSync('research/frontier/annotations-r10.json')).originalLocalRevision.cases))wanted.add(a.sourceUid);
const records=manifest.records.filter(r=>wanted.has(r.uid));
const widths=[1000,2000,4000],results=widths.map(()=>[]);
for(const r of records){
 const c=context(lab,fs.readFileSync(r.source,'utf8')),base=c.map.firstNote,last=c.map.lastNote;
 const ds=c.windows.map(w=>w.cutDemand).filter(v=>v>0).sort((a,b)=>a-b),ref=ds.length?ds[Math.floor(ds.length*.8)]:1;
 widths.forEach((width,wi)=>{
  const cells=[],windows=[],rawSegs=[],extras=[];
  for(let start=base;start<last;start+=500){
   const end=Math.min(start+500,last),center=(start+end)/2,a=Math.max(base,center-width/2),b=Math.min(last,center+width/2),i=cells.length;
   cells.push({start,end});windows.push(lab.computeWindowFeatures(c.ctx,a,b));extras.push(lab.computeSegmentExtras(c.ctx,a,b));rawSegs.push({start:a,end:b,windowRange:[i,i+1]});
  }
  const marks=lab.markBreaks(c.map,c.rows,rawSegs);rawSegs.forEach((s,i)=>s.family0=marks[i].family);
  const cm={windows,rawSegs,extras,cutDemandRef:ref},classified=lab.classifySegments(cm,{scorer:'model'}),pre=early.classifySegments(cm,{scorer:'model'}),mapmed=lab.computeMapMedians(windows);
  const parents=cells.map((cell,j)=>{
   const f=lab.medianWindowFeatures(windows,j,j+1,mapmed),ex=extras[j];
   return {...cell,context:[rawSegs[j].start,rawSegs[j].end],f,extra:ex,vector:lab.VEC_KEYS.map(k=>lab.buildVector(f,ex,ref)[k]),modelRc:lab.modelScores('rc',f,ex,ref),modelLn:lab.modelScores('ln',f,ex,ref),ruleRc:lab.scoreRc({f,extras:ex,cutDemandRef:ref}),ruleLn:lab.scoreLn({f,extras:ex}),baseline:{...classified[j],...cell},preLate:{...pre[j],...cell}};
  });
  results[wi].push({uid:r.uid,source:r.source,width,cutDemandRef:ref,parents});
 });
 console.log('local context',r.uid);
}
widths.forEach((width,wi)=>{
 const out='.cache/core-retune-local-context-'+width+'.jsonl.gz';if(fs.existsSync(out))throw Error('Preserve context input '+out);
 fs.writeFileSync(out,zlib.gzipSync(results[wi].map(r=>JSON.stringify(r)).join('\n')+'\n'));fs.writeFileSync(out+'.schema.json',JSON.stringify({width,step:500,records:records.length,annotationBoundariesUsed:false,source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
});
