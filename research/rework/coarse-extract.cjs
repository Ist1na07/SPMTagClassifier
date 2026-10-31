'use strict';
// Label-independent shared representation. Cache files remain local; the
// published inventory contains record identities and counts, not note data.
const fs=require('node:fs'),path=require('node:path');
const parser=require('../../src/parser.cjs'),physical=require('./physical.cjs'),base=require('../../src/features.cjs');
const {ROOT,records7}=require('./data.cjs');
const CACHE=path.resolve(ROOT,process.argv[2]||'.cache/rework-coarse-r2');
function records(){
 const folder='spm_dataset-4k-0.2.0',dir=path.join(ROOT,'dataset',folder);
 const four=JSON.parse(fs.readFileSync(path.join(dir,'labels.json'),'utf8').replace(/^\uFEFF/,''))
  .map((r,i)=>({...r,uid:folder+'/'+r.id+'/'+i,dataset:folder,path:path.join(dir,r.file)}));
 return [...four,...records7()];
}
function run(){
 fs.mkdirSync(CACHE,{recursive:true});
 const spec={id:'coarse-native-physical-r2',windowMs:[2000,4000,8000],stepMs:2000,rowMergeMs:10,
  keys:physical.KEYS,baseKeys:base.FEATURE_KEYS,groups:physical.GROUPS,
  physicalKeys:'actual map CircleSize columns, including real unused columns; not only columns with notes',
  sourceSnapshot:Object.fromEntries(['coarse-extract.cjs','physical.cjs','anchor.cjs','release-geometry.cjs'].map(n=>[n,fs.readFileSync(path.join(__dirname,n),'utf8')])),
  coreSnapshot:Object.fromEntries(['parser.cjs','features.cjs'].map(n=>[n,fs.readFileSync(path.join(ROOT,'src',n),'utf8')]))};
 const specPath=path.join(CACHE,'spec.json');
 if(fs.existsSync(specPath)&&JSON.stringify(JSON.parse(fs.readFileSync(specPath,'utf8')))!==JSON.stringify(spec))throw Error('Cache protocol changed: use a new revision');
 fs.writeFileSync(specPath,JSON.stringify(spec,null,2)+'\n');
 const seen=new Map(),inventory=[],errors=[];
 for(const r of records()){
  const file=path.join(CACHE,r.uid.replaceAll('/','__')+'.json');
  try{
   const map=parser.parseOsu(fs.readFileSync(r.path,'utf8')),ctx=physical.prepare(parser.prepare(map,10));
   if(![4,7].includes(map.keys))throw Error('Unexpected labelled source key count');
   const gaps=ctx.rows.slice(1).map((x,i)=>x.t-ctx.rows[i].t).filter(x=>x>0).sort((a,b)=>a-b);
   const scale=gaps[Math.floor((gaps.length-1)/2)]||1;
   const signature=map.keys+'|'+map.notes.map(n=>[Math.round((n.t-map.firstNote)/scale*1e6),n.col,n.lnEnd===null?'':Math.round((n.lnEnd-n.t)/scale*1e6)].join(',')).join(';');
   const duplicateOf=seen.get(signature)||null;if(!duplicateOf)seen.set(signature,r.uid);
   const {path:localPath,...record}=r;
   const header={...record,keys:map.keys,source:path.relative(ROOT,localPath).replaceAll('\\','/'),meta:map.meta,
    duplicateOf,first:map.firstNote,last:map.lastNote,notes:map.notes.length,warnings:map.warnings};
   if(!fs.existsSync(file)){
    const windows={};
    for(const ms of spec.windowMs)windows[ms]=physical.windows(ctx,ms,spec.stepMs).map(w=>({
     start:w.start,end:w.end,contextStart:w.contextStart,contextEnd:w.contextEnd,x:w.values,base:w.baseValues,
     empty:w.structurallyEmpty,notes:w.notes,occupancy:w.metrics.occupancy,effort:w.metrics.effort}));
    fs.writeFileSync(file,JSON.stringify({...header,windows})+'\n');
   }else{
    const old=JSON.parse(fs.readFileSync(file,'utf8'));
    if(old.uid!==header.uid||old.source!==header.source||old.notes!==header.notes||old.duplicateOf!==duplicateOf)throw Error('Cached source identity differs');
   }
   inventory.push(header);
   if(inventory.length%100===0)console.log('coarse extracted',inventory.length);
  }catch(e){errors.push({uid:r.uid,error:e.stack});}
 }
 fs.writeFileSync(path.join(CACHE,'inventory.json'),JSON.stringify({id:spec.id,records:inventory,errors},null,2)+'\n');
 console.log('coarse complete',inventory.length,'errors',errors.length);if(errors.length)process.exitCode=1;
}
if(require.main===module)run();
module.exports={records,CACHE};
