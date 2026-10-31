'use strict';
// Complete archived segment representation + same-interval modern comparator.
// Raw charts remain local; every feature is computed without reading labels.
const fs=require('node:fs'),path=require('node:path');
const {loadLab,inspect}=require('./legacy-lab.cjs');
const {ROOT,records7}=require('./data.cjs');
const patches=require('./patches.cjs');
const modern=require('../../src/features.cjs'),parser=require('../../src/parser.cjs');
const CLASSES=['chordjack','minijack','dense chordstream','fast chordstream','speed','tech','vibro','anchor',
 'coordination','density','inverse','release','technical','wildcard','ln wildcard','break'];
const CONTEXT=['relative_nps','relative_burden','nps_rank','burden_rank','relative_chord',
 'relative_ln','relative_area','log_duration','map_nps','relative_effort'];
const base=loadLab(),fixed=loadLab({patches:patches.logical.concat(patches.releaseIsolation)});
const file=process.env.REWORK_FEATURES||path.join(ROOT,'.cache/rework7.jsonl');
fs.mkdirSync(path.dirname(file),{recursive:true});const fd=fs.openSync(file,'w'),seen=new Map();
const quantile=(a,p)=>a.slice().sort((a,b)=>a-b)[Math.floor((a.length-1)*p)]||0;
let count=0;
for(const r of records7()){
 const text=fs.readFileSync(r.path,'utf8'),map=parser.parseOsu(text),ctx=parser.prepare(map);
 const b=inspect(base,text),c=inspect(fixed,text,{intervals:b.segments});
 const iv=ctx.rows.slice(1).map((r,i)=>r.t-ctx.rows[i].t).filter(x=>x>0),scale=quantile(iv,.5)||1;
 const signature=map.keys+'|'+map.notes.map(n=>[Math.round((n.t-map.firstNote)/scale*10),n.col,n.lnEnd===null?'':Math.round((n.lnEnd-n.t)/scale*10)].join(',')).join(';');
 const duplicateOf=seen.get(signature)||null;if(!duplicateOf)seen.set(signature,r.uid);
 const modernValues=b.segments.map(s=>modern.features(ctx,s.start,s.end).values);
 const effort=modernValues.map(v=>v[modern.FEATURE_KEYS.indexOf('effort')]),effortRef=quantile(effort,.5)||1;
 const nps=b.segments.map(s=>s.f.nps),npsRef=quantile(nps,.5)||1;
 const burdens=b.segments.map(s=>s.f.nps*(.5+s.f.chordMean/3)*(.6+s.f.jackSpeedBpm/150));
 const burdenRef=quantile(burdens,.5)||1;
 const chordRef=quantile(b.segments.map(s=>s.f.chordMean),.5)||1;
 const lnRef=quantile(b.segments.map(s=>s.f.lnWeightRatio),.5),areaRef=quantile(b.segments.map(s=>s.f.lnAreaRatio),.5);
 const rank=(values,value)=>values.filter(x=>x<value).length/values.length;
 const segments=b.segments.map((s,i)=>{
  const prediction=b.classified[i],scores=prediction.family==='break'?{break:1}:(prediction.scores||{[prediction.tag]:1});
  const mass=Object.values(scores).reduce((a,b)=>a+b,0)||1;
  const prior=CLASSES.map(k=>.99*(scores[k]||0)/mass+.01/CLASSES.length);
  const context=[Math.log1p(s.f.nps/npsRef),Math.log1p(burdens[i]/burdenRef),rank(nps,s.f.nps),rank(burdens,burdens[i]),
   s.f.chordMean/chordRef,s.f.lnWeightRatio-lnRef,s.f.lnAreaRatio-areaRef,Math.log1p((s.end-s.start)/1000),
   Math.log1p(npsRef),Math.log1p(effort[i]/effortRef)];
  return {start:s.start,end:s.end,original:s.vector,corrected:c.segments[i].vector,modern:modernValues[i],context,
   prior,legacy:prediction.tag,effort:effort[i],notes:s.extra.notes};
 });
 const {path:localPath,...record}=r;
 const result={...record,source:localPath.slice(ROOT.length+1).replace(/\\/g,'/'),meta:map.meta,
  duplicateOf,segments,legacyOverall:b.smoothed.length?base.lab.aggregateOverall(b.smoothed):{tags:[]}};
 if(!segments.every(s=>['original','corrected','modern','context','prior'].every(k=>s[k].every(Number.isFinite))))throw Error('Nonfinite features '+r.uid);
 fs.writeSync(fd,JSON.stringify(result)+'\n');if(++count%50===0)console.log('extracted',count);
}
fs.closeSync(fd);
fs.writeFileSync(path.join(ROOT,'.cache/rework-schema.json'),JSON.stringify({classes:CLASSES,original:base.lab.VEC_KEYS,corrected:base.lab.VEC_KEYS,modern:modern.FEATURE_KEYS,context:CONTEXT},null,2)+'\n');
// Later anchored/cold studies consume the complete archived experts. Generate
// this from the real archive on a fresh checkout, not a manual cache step.
fs.writeFileSync(path.join(ROOT,'.cache/archived-model-reference.json'),JSON.stringify(base.lab.MODEL)+'\n');
console.log('finished',count,file);
