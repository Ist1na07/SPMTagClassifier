'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const {ROOT,records7}=require('./data.cjs');
const {mirrorText}=require('./mirror.cjs');
const {context}=require('./legacy-lab.cjs');
const {nativeApi,extractNative,SPEC}=require('./native.cjs');
const root=path.join(ROOT,'.cache/rework-native-mirror-r1');fs.mkdirSync(root,{recursive:true});
const spec={id:'native-mirror-r1',nativeSpecification:SPEC,treatment:'original',mirror:'physical-column mirror; labels never read by extraction'};
const file=path.join(root,'spec.json');
if(fs.existsSync(file))assert.deepEqual(JSON.parse(fs.readFileSync(file,'utf8')),spec);else fs.writeFileSync(file,JSON.stringify(spec,null,2)+'\n');
const api=nativeApi('original');let count=0;
for(const r of records7()){
 const file=path.join(root,r.uid.replaceAll('/','__')+'.json.gz');
 if(!fs.existsSync(file)){
  const text=mirrorText(fs.readFileSync(r.path,'utf8')),prepared=context(api.lab,text),partitions=extractNative(api,prepared);
  const value={id:spec.id,uid:r.uid,firstNote:prepared.map.firstNote,lastNote:prepared.map.lastNote,partitions};
  const temporary=file+'.tmp';fs.writeFileSync(temporary,zlib.gzipSync(JSON.stringify(value)));fs.renameSync(temporary,file);
 }
 if(++count%20===0)console.log('mirrored native charts',count);
}
console.log('mirrored native complete',count);
