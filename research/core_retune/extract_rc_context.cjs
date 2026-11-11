'use strict';
// Attribution probe: preserve ALL old route/rest decisions and old LN experts.
// Only replace RC model input with local motion evidence at the inherited
// 500ms child clock. No manual annotation endpoint is a feature boundary.
const fs=require('node:fs'),zlib=require('node:zlib');
const {loadLab,context}=require('../rework/legacy-lab.cjs');
const api=loadLab(),lab=api.lab;
const all=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r2.jsonl.gz')).toString().trim().split('\n').map(JSON.parse);
const local=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-local-context-4000.jsonl.gz')).toString().trim().split('\n').map(JSON.parse),wanted=new Set(local.map(r=>r.uid));
const result=[];
for(const r of all.filter(r=>wanted.has(r.uid))){
 const ctx=context(lab,fs.readFileSync(r.source,'utf8'));const mapmed=lab.computeMapMedians(ctx.windows);const parents=[];
 for(let parent=0;parent<r.parents.length;parent++){
  const p=r.parents[parent];
  for(let start=p.start;start<p.end;start+=500){
   const end=Math.min(start+500,p.end),center=(start+end)/2,local={};
   if(p.baseline.family==='rc')for(const width of [1000,2000,4000]){
    const a=Math.max(ctx.map.firstNote,center-width/2),b=Math.min(ctx.map.lastNote,center+width/2),f={...lab.computeWindowFeatures(ctx.ctx,a,b),...mapmed},ex=lab.computeSegmentExtras(ctx.ctx,a,b),v=lab.buildVector(f,ex,r.cutDemandRef);local[width]=lab.VEC_KEYS.map(k=>v[k]);
   }
   parents.push({...p,start,end,context:[p.start,p.end],parent,baseline:{...p.baseline,start,end},preLate:{...p.baseline,start,end},localVectors:local});
  }
 }
 result.push({uid:r.uid,source:r.source,parents});console.log('RC-only physical contexts',r.uid);
}
const out='.cache/core-retune-rc-context-r1.jsonl.gz';if(fs.existsSync(out))throw Error('Preserve RC context input');
fs.writeFileSync(out,zlib.gzipSync(result.map(r=>JSON.stringify(r)).join('\n')+'\n'));
