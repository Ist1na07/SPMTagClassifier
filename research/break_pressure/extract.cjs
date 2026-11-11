'use strict';
const fs=require('node:fs'),zlib=require('node:zlib');
const {classifier}=require('./audit.cjs');
const {createPressureGuard}=require('../../src/break-pressure.cjs');
const OUT='.cache/break-pressure-input-r1.jsonl.gz';if(fs.existsSync(OUT))throw Error('Preserve prior extraction');
const rows=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r2.jsonl.gz')).toString().trim().split('\n').map(JSON.parse);
const resources=new Map(fs.readFileSync('.cache/perception-final-resource.jsonl','utf8').trim().split('\n').map(s=>{const r=JSON.parse(s);return[r.uid,r];}));
const raw=classifier({low:false,late:false}),base=classifier(),out=[];
for(const r of rows){
 const source=resources.get(r.uid),frames=raw(r.cm),baseline=base(r.cm),pressure={};
 for(const name of ['detail','compact'])pressure[name]=[.5,.75,.9].map(quantile=>{
  const guard=createPressureGuard(source.ratings[name],{quantile,fraction:.5});return{quantile,reference:guard.reference,segments:frames.map(s=>guard.measure(s.start,s.end))};
 });
 out.push({uid:r.uid,frames,baseline,pressure,parents:r.parents.map((p,j)=>({start:p.start,end:p.end,f:p.f,extra:p.extra,marked:r.cm.rawSegs[j].family0==='break'}))});
}
fs.writeFileSync(OUT,zlib.gzipSync(out.map(r=>JSON.stringify(r)).join('\n')+'\n'));console.log('Extracted',out.length);
