'use strict';
// Native-key physical descriptors for the shared coarse model. This module
// does not select tags or treat an unobserved attribute as a negative label.
const base=require('../../src/features.cjs');
const {lowerBound:lb}=require('../../src/parser.cjs');
const anchor=require('./anchor.cjs');
const release=require('./release-geometry.cjs');
const REPLACED=new Set(['anchorShare','repeatRun']);
const SEQUENCE=anchor.KEYS.filter(k=>k!=='peakShare');
const BASE_KEYS=base.FEATURE_KEYS.filter(k=>!REPLACED.has(k));
const KEYS=[...BASE_KEYS,...SEQUENCE,...release.KEYS.map(k=>'release:'+k)];
const GROUPS={...Object.fromEntries(Object.entries(base.GROUPS).map(([g,ks])=>[g,ks.filter(k=>!REPLACED.has(k))])),
 sequence:[...base.GROUPS.sequence.filter(k=>!REPLACED.has(k)),...SEQUENCE],release:release.KEYS.map(k=>'release:'+k)};

function features(ctx,start,end){
 const f=base.features(ctx,start,end);
 const rows=ctx.rows.slice(lb(ctx.rows,start,r=>r.t),lb(ctx.rows,end,r=>r.t));
 const a=anchor.describe(rows,ctx.map.keys),r=release.releaseGeometry(ctx,start,end);
 const sequence=Object.fromEntries(anchor.KEYS.map((k,i)=>[k,a[i]]));
 const metrics={...f.metrics,...sequence,...Object.fromEntries(release.KEYS.map((k,i)=>['release:'+k,r.values[i]]))};
 // An empty head bin is not sustained head activity. Long held notes still
 // have physical occupancy and are not erased by the head-only count.
 if(!f.notes)metrics.sustain=0;
 const values=KEYS.map(k=>metrics[k]);
 if(values.some(x=>!Number.isFinite(x)))throw Error('Nonfinite native-key descriptors');
 return {...f,values,metrics,baseValues:base.FEATURE_KEYS.map(k=>k==='sustain'&&!f.notes?0:f.metrics[k]),
   tails:r.tails,structurallyEmpty:f.notes===0&&metrics.occupancy===0&&r.tails===0};
}

function windows(ctx,windowMs=4000,stepMs=2000){
 const {firstNote}=ctx.map,end=Math.max(firstNote+1,ctx.map.lastNote+1),out=[];
 for(let t=firstNote;t<end;t+=stepMs){
  const center=Math.min(end,t+stepMs/2),a=Math.max(firstNote,center-windowMs/2),b=Math.min(end,center+windowMs/2);
  out.push({...features(ctx,a,b),start:t,end:Math.min(t+stepMs,end),contextStart:a,contextEnd:b});
 }
 return out;
}
module.exports={KEYS,GROUPS,BASE_KEYS,SEQUENCE,features,windows,prepare:release.releaseContext};
