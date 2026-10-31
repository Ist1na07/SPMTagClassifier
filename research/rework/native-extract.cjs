'use strict';
// Only labels/record IDs locate source files. No target label enters extraction.
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const {ROOT,records7}=require('./data.cjs');
const {context}=require('./legacy-lab.cjs');
const {SPEC,TREATMENTS,nativeApi,extractNative,decisionSupport}=require('./native.cjs');
const requested=(process.argv.find(a=>a.startsWith('--treatments='))||'').split('=').slice(1).join('=');
const names=requested?requested.split(','):Object.keys(TREATMENTS);
const limit=Number((process.argv.find(a=>a.startsWith('--limit='))||'--limit=0').split('=')[1]);
const out=path.join(ROOT,'.cache/rework-native-r1');fs.mkdirSync(out,{recursive:true});
const manifest=path.join(out,'spec.json');
if(fs.existsSync(manifest))assert.deepEqual(JSON.parse(fs.readFileSync(manifest,'utf8')),SPEC,'Native cache specification changed; use a new revision');
else fs.writeFileSync(manifest,JSON.stringify(SPEC,null,2)+'\n');
const apis=new Map(names.map(k=>[k,nativeApi(k)]));
const legacyCache=path.join(ROOT,'.cache/rework-legacy-full.json');
const originals=fs.existsSync(legacyCache)?new Map(JSON.parse(fs.readFileSync(legacyCache,'utf8')).map(r=>[r.uid,r])):null;
let count=0,created=0,parity=0;
function checkOriginal(value){
 if(!originals)return;
 const baseline=originals.get(value.uid).segments,current=value.partitions[4].cases[2].legacy;
 assert.equal(current.length,baseline.length,value.uid);
 current.forEach((s,i)=>{
  for(const key of ['start','end','tag','family','confidence'])assert.equal(s[key],baseline[i][key],value.uid+' '+key);
  assert.deepEqual(s.probability,decisionSupport(baseline[i]),value.uid+' support');
 });parity++;
}
for(const r of records7()){
 if(limit&&count>=limit)break;
 for(const name of names){
  const directory=path.join(out,name);fs.mkdirSync(directory,{recursive:true});
  const file=path.join(directory,r.uid.replaceAll('/','__')+'.json.gz');
  if(fs.existsSync(file)){
   if(name==='original')checkOriginal(JSON.parse(zlib.gunzipSync(fs.readFileSync(file))));
   continue;
  }
  const text=fs.readFileSync(r.path,'utf8'),api=apis.get(name),prepared=context(api.lab,text);
  const partitions=extractNative(api,prepared);
  const value={id:SPEC.id,uid:r.uid,treatment:name,firstNote:prepared.map.firstNote,lastNote:prepared.map.lastNote,partitions};
  if(name==='original')checkOriginal(value);
  const temporary=file+'.tmp';fs.writeFileSync(temporary,zlib.gzipSync(JSON.stringify(value)));fs.renameSync(temporary,file);created++;
 }
 if(++count%10===0)console.log('native charts',count,'new caches',created);
}
console.log('native extraction complete',count,'records',names.length,'treatments','created',created,'original parity',parity);
