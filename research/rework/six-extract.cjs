'use strict';
// Actual native-six-column feature checks. Difficulty numbers and empty
// pattern labels do not enter features or any supervised selection.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const parser=require('../../src/parser.cjs'),physical=require('./physical.cjs'),base=require('../../src/features.cjs');
const {ROOT}=require('./data.cjs');
const folder='spm_dataset_6k-0.1.0',dir=path.join(ROOT,'dataset',folder),out=path.resolve(ROOT,process.argv[2]||'.cache/rework-six-physical-r2');
fs.mkdirSync(out,{recursive:true});
const records=JSON.parse(fs.readFileSync(path.join(dir,'labels.json'),'utf8').replace(/^\uFEFF/,''));
if(records.some(r=>r.tags.length))throw Error('6K label availability changed: review the protocol before extraction');
const spec={id:'six-native-physical-r2',keys:physical.KEYS,baseKeys:base.FEATURE_KEYS,windowMs:[2000,4000,8000],stepMs:2000,rowMergeMs:10,
 scope:'actual 6K, no human pattern labels; finite features and physical interval contract only',
 sourceSnapshot:Object.fromEntries(['six-extract.cjs','physical.cjs','anchor.cjs','release-geometry.cjs'].map(n=>[n,fs.readFileSync(path.join(__dirname,n),'utf8')])),
 coreSnapshot:Object.fromEntries(['parser.cjs','features.cjs'].map(n=>[n,fs.readFileSync(path.join(ROOT,'src',n),'utf8')]))};
const manifest=path.join(out,'spec.json');
if(fs.existsSync(manifest))assert.deepEqual(JSON.parse(fs.readFileSync(manifest,'utf8')),spec,'6K cache changed; use a new revision');
else fs.writeFileSync(manifest,JSON.stringify(spec,null,2)+'\n');
const inventory=[],errors=[];
for(let i=0;i<records.length;i++){
 const r=records[i],uid=folder+'/'+r.id+'/'+i,file=path.join(out,uid.replaceAll('/','__')+'.json');
 try{
  if(!fs.existsSync(file)){
   const map=parser.parseOsu(fs.readFileSync(path.join(dir,r.file),'utf8'));assert.equal(map.keys,6);
   const ctx=physical.prepare(parser.prepare(map,spec.rowMergeMs)),windows={};
   for(const ms of spec.windowMs){
    windows[ms]=physical.windows(ctx,ms,spec.stepMs).map(w=>({start:w.start,end:w.end,contextStart:w.contextStart,contextEnd:w.contextEnd,
     x:w.values,base:w.baseValues,empty:w.structurallyEmpty,notes:w.notes,occupancy:w.metrics.occupancy,effort:w.metrics.effort}));
    const ws=windows[ms];assert.equal(ws[0].start,map.firstNote);assert.equal(ws.at(-1).end,map.lastNote+1);
    for(let j=1;j<ws.length;j++)assert.equal(ws[j].start,ws[j-1].end);
   }
   const value={uid,id:r.id,keys:6,dataset:folder,source:'dataset/'+folder+'/'+r.file,meta:map.meta,first:map.firstNote,last:map.lastNote,
    notes:map.notes.length,warnings:map.warnings,windows};
   fs.writeFileSync(file,JSON.stringify(value)+'\n');
  }
  const value=JSON.parse(fs.readFileSync(file,'utf8'));
  if(value.uid!==uid||value.keys!==6)throw Error('6K cached source identity differs');
  inventory.push({uid,source:value.source,notes:value.notes,warnings:value.warnings,windows:Object.fromEntries(Object.entries(value.windows).map(([k,ws])=>[k,ws.length]))});
 }catch(e){errors.push({uid,source:r.file,error:e.message});}
 if((i+1)%100===0)console.log('actual 6K checked',i+1,'valid',inventory.length,'rejected',errors.length);
}
const result={id:spec.id,requested:records.length,valid:inventory.length,records:inventory,errors,
 patternTruth:'none; these checks do not estimate human 6K classification accuracy'};
fs.writeFileSync(path.join(out,'inventory.json'),JSON.stringify(result,null,2)+'\n');
console.log('actual 6K complete',records.length,'valid',inventory.length,'rejected',errors.length);
