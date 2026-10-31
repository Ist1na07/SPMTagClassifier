'use strict';
// The missing cell of the neutral/logical x release x normalization study.
// logical_release_* ALREADY includes neutral hands and coverage (native.cjs).
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const {ROOT,records7}=require('./data.cjs'),{loadLab,context}=require('./legacy-lab.cjs');
const {SPEC,extractNative}=require('./native.cjs'),patches=require('./patches.cjs');
const name='logical_neutral_scale',applied=patches.logical.concat(patches.neutralHands,patches.coverage,patches.transformedScale);
const out=path.join(ROOT,'.cache/rework-native-factorial-r1'),directory=path.join(out,name);
fs.mkdirSync(directory,{recursive:true});
const spec={id:'native-factorial-r1',baseSpecification:SPEC,treatment:name,patches:applied.map(p=>p.id),
 sourceSnapshot:Object.fromEntries(['native-factorial-extract.cjs','native.cjs','legacy-lab.cjs','patches.cjs'].map(n=>[n,fs.readFileSync(path.join(__dirname,n),'utf8')]))};
const manifest=path.join(out,'spec.json');
if(fs.existsSync(manifest))assert.deepEqual(JSON.parse(fs.readFileSync(manifest,'utf8')),spec,'Factorial cache changed; use a new revision');
else fs.writeFileSync(manifest,JSON.stringify(spec,null,2)+'\n');
const api=loadLab({patches:applied});let count=0,created=0;
for(const r of records7()){
 const file=path.join(directory,r.uid.replaceAll('/','__')+'.json.gz');
 if(!fs.existsSync(file)){
  const prepared=context(api.lab,fs.readFileSync(r.path,'utf8'));
  const value={id:spec.id,uid:r.uid,treatment:name,firstNote:prepared.map.firstNote,lastNote:prepared.map.lastNote,partitions:extractNative(api,prepared)};
  const temp=file+'.tmp';fs.writeFileSync(temp,zlib.gzipSync(JSON.stringify(value)));fs.renameSync(temp,file);created++;
 }
 if(++count%25===0)console.log('factorial native charts',count,'created',created);
}
console.log('factorial native complete',count,'created',created);
