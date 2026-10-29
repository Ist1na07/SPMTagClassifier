'use strict';
// Optional, fixed-prior mechanism representation from the pinned plugin's
// difficulty engine. This is an extraction probe, NOT a validated classifier.
const fs=require('node:fs'),path=require('node:path');
const {ROOT,records7}=require('./data.cjs');
const source=path.join(ROOT,'.upstream/spm_algorithm.js');
let engine;
const getEngine=()=>engine??=new Function(fs.readFileSync(source,'utf8')+';return SPMEngine;')();
const CHANNELS=['Jm','Jc','Ja','Pbar','Xbar','Rbar','Cbar','Chord2','Recov'];
function lb(a,t){let lo=0,hi=a.length;while(lo<hi){const mid=(lo+hi)>>>1;if(a[mid]<t)lo=mid+1;else hi=mid;}return lo;}
function summarise(times,values,start,end){
 const pieces=[];let sum=0,total=0,max=-Infinity;
 for(let i=Math.max(0,lb(times,start)-1);i<times.length&&times[i]<end;i++){
  const duration=Math.max(0,Math.min(end,times[i+1]??end)-Math.max(start,times[i]));
  if(!duration)continue;
  const v=values[i];if(!Number.isFinite(v))throw Error('NonfiniteMechanism');
  sum+=duration*v;total+=duration;max=Math.max(max,v);pieces.push([v,duration]);
 }
 pieces.sort((a,b)=>a[0]-b[0]);let acc=0,q90=0;
 for(const [v,w]of pieces){acc+=w;q90=v;if(acc>=.9*total)break;}
 return [sum/Math.max(1,total),q90,pieces.length?max:0];
}
function extract(text,segments){
 const engine=getEngine();
 const notes=engine.parseOsu(text,1),batch=engine.buildBatch(notes),structure=engine.buildStruct(batch,engine.SPM_V120_PARAMS);
 const curves=engine.computeCurves(batch,structure,engine.SPM_V120_PARAMS);
 return segments.map(s=>CHANNELS.flatMap(c=>summarise(batch.t,curves[c],s.start,s.end).map(v=>Math.sign(v)*Math.log1p(Math.abs(v)))));
}
function run(){
 const byid=new Map(records7().map(r=>[r.uid,r]));
 const records=fs.readFileSync('.cache/rework7.jsonl','utf8').trim().split('\n').map(JSON.parse);
 const file='.cache/rework-mechanics.jsonl',fd=fs.openSync(file,'w');let count=0;
 for(const r of records){
  const text=fs.readFileSync(byid.get(r.uid).path,'utf8'),features=extract(text,r.segments);
  fs.writeSync(fd,JSON.stringify({uid:r.uid,features})+'\n');if(++count%50===0)console.log('mechanics',count);
 }
 fs.closeSync(fd);fs.writeFileSync('.cache/rework-mechanics-schema.json',JSON.stringify({
  source:'Ist1na07/spm_rating_map_analyser@1e99e4dc20f0c338de414e76afeaf73d8219d65c',
  features:CHANNELS.flatMap(c=>['timeMean','timeQ90','maximum'].map(s=>c+':'+s)),
  transform:'signed log1p; piecewise constant on [row_i,row_i+1), integrated over interval duration',
  evidence:'fixed difficulty-model representation only; classification utility not established'})+'\n');
}
if(require.main===module)run();
module.exports={extract,summarise,CHANNELS};
