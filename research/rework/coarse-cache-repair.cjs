'use strict';
// Revise one evidenced defective coordinate; preserve the original caches.
// All other release coordinates are recomputed and required BITWISE equal.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {ROOT}=require('./data.cjs'),parser=require('../../src/parser.cjs'),release=require('./release-geometry.cjs');
const specs=[['rework-coarse-r1','rework-coarse-r2','coarse-native-physical-r2'],
 ['rework-six-physical-r1','rework-six-physical-r2','six-native-physical-r2']];
function write(file,value){const temp=file+'.tmp';fs.writeFileSync(temp,JSON.stringify(value)+'\n');fs.renameSync(temp,file);}
const audits=[];
for(const [oldName,newName,id]of specs){
 const from=path.join(ROOT,'.cache',oldName),to=path.join(ROOT,'.cache',newName);fs.mkdirSync(to,{recursive:true});
 const previous=JSON.parse(fs.readFileSync(path.join(from,'spec.json'),'utf8')),inventory=JSON.parse(fs.readFileSync(path.join(from,'inventory.json'),'utf8'));
 const spec={...previous,id,parentSpecification:previous,
  cacheRepair:{from:oldName,meaning:'sameHandOrder: three consecutive unique timestamps, no column-order inference from release chords',
   source:fs.readFileSync(__filename,'utf8')},
  sourceSnapshot:{...previous.sourceSnapshot,'release-geometry.cjs':fs.readFileSync(path.join(__dirname,'release-geometry.cjs'),'utf8')}};
 const specPath=path.join(to,'spec.json');
 if(fs.existsSync(specPath))assert.deepEqual(JSON.parse(fs.readFileSync(specPath,'utf8')),spec);else write(specPath,spec);
 const index=spec.keys.indexOf('release:sameHandOrder'),releaseIndices=release.KEYS.map(k=>spec.keys.indexOf('release:'+k));
 const changed=[],counts={};let windows=0,changedWindows=0;
 for(const r of inventory.records){
  const filename=r.uid.replaceAll('/','__')+'.json',output=path.join(to,filename);
  if(fs.existsSync(output))continue;
  const value=JSON.parse(fs.readFileSync(path.join(from,filename),'utf8'));
  const map=parser.parseOsu(fs.readFileSync(path.join(ROOT,value.source),'utf8')),ctx=release.releaseContext(parser.prepare(map,10));
  let affected=0,maxDelta=0;
  for(const ws of Object.values(value.windows))for(const w of ws){
   const current=release.releaseGeometry(ctx,w.contextStart,w.contextEnd).values;windows++;
   for(let j=0;j<release.KEYS.length;j++){
    const old=w.x[releaseIndices[j]];
    if(releaseIndices[j]!==index)assert.equal(current[j],old,`${r.uid} ${release.KEYS[j]} changed unexpectedly`);
    else if(current[j]!==old){maxDelta=Math.max(maxDelta,Math.abs(current[j]-old));w.x[index]=current[j];affected++;changedWindows++;}
   }
  }
  counts[value.keys]=(counts[value.keys]||0)+1;
  if(affected)changed.push({uid:r.uid,keys:value.keys,windows:affected,maxDelta});
  write(output,value);
  if(Object.values(counts).reduce((a,b)=>a+b,0)%100===0)console.log('repair',newName,counts,'affected charts',changed.length);
 }
 // For a resumed repair, reconstruct the complete delta audit from old/new
 // persisted arrays, rather than misreporting only this process's new work.
 const allChanged=[];let totalWindows=0,totalChanged=0;const keyCounts={};
 for(const r of inventory.records){
  const filename=r.uid.replaceAll('/','__')+'.json',a=JSON.parse(fs.readFileSync(path.join(from,filename),'utf8')),b=JSON.parse(fs.readFileSync(path.join(to,filename),'utf8'));
  let affected=0,maxDelta=0;keyCounts[a.keys]=(keyCounts[a.keys]||0)+1;
  for(const ms of Object.keys(a.windows))for(let i=0;i<a.windows[ms].length;i++){
   const x=a.windows[ms][i],y=b.windows[ms][i];totalWindows++;
   for(let j=0;j<x.x.length;j++)if(j!==index)assert.equal(x.x[j],y.x[j]);
   assert.deepEqual(x.base,y.base);
   if(x.x[index]!==y.x[index]){affected++;totalChanged++;maxDelta=Math.max(maxDelta,Math.abs(x.x[index]-y.x[index]));}
  }
  if(affected)allChanged.push({uid:r.uid,keys:a.keys,windows:affected,maxDelta});
 }
 const audit={from:oldName,to:newName,records:inventory.records.length,keyCounts,windows:totalWindows,changedWindows:totalChanged,
  changedCharts:allChanged,unchanged:'all base40 coordinates and all physical coordinates except sameHandOrder are bitwise equal; intervals, gates and all labels are retained'};
 write(path.join(to,'repair-audit.json'),audit);write(path.join(to,'inventory.json'),{...inventory,id});audits.push(audit);
 console.log('repair complete',newName,'charts',inventory.records.length,'changed charts',allChanged.length,'changed windows',totalChanged);
}
const out=path.join(ROOT,'research/rework/results/release-order-repair-r1');fs.mkdirSync(out,{recursive:true});
write(path.join(out,'cache-audit.json'),audits);
