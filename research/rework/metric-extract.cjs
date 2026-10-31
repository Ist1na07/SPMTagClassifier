'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const {ROOT,records7}=require('./data.cjs'),{context}=require('./legacy-lab.cjs'),{SPEC,extractNative}=require('./native.cjs');
const {metricApi,RELEASES}=require('./metric.cjs');
const treatments=RELEASES.flatMap(releaseMs=>[.05,.2].map(strength=>({name:'metric_'+(releaseMs??'none')+'_'+Math.round(100*strength),releaseMs,strength})));
const out=path.join(ROOT,'.cache/rework-native-metric-r1');fs.mkdirSync(out,{recursive:true});
const spec={id:'native-metric-r1',baseSpecification:SPEC,treatments,
 sourceSnapshot:Object.fromEntries(['metric-extract.cjs','metric.cjs','native.cjs','legacy-lab.cjs','patches.cjs'].map(n=>[n,fs.readFileSync(path.join(__dirname,n),'utf8')]))};
const manifest=path.join(out,'spec.json');
if(fs.existsSync(manifest))assert.deepEqual(JSON.parse(fs.readFileSync(manifest,'utf8')),spec,'Metric cache changed; use a new revision');
else fs.writeFileSync(manifest,JSON.stringify(spec,null,2)+'\n');
const apis=new Map(treatments.map(t=>[t.name,metricApi(t.releaseMs,t.strength)]));let count=0,created=0;
for(const r of records7()){
 for(const t of treatments){
  const directory=path.join(out,t.name);fs.mkdirSync(directory,{recursive:true});const file=path.join(directory,r.uid.replaceAll('/','__')+'.json.gz');
  if(fs.existsSync(file))continue;
  const api=apis.get(t.name),prepared=context(api.lab,fs.readFileSync(r.path,'utf8'));
  const value={id:spec.id,uid:r.uid,treatment:t.name,firstNote:prepared.map.firstNote,lastNote:prepared.map.lastNote,partitions:extractNative(api,prepared)};
  const temp=file+'.tmp';fs.writeFileSync(temp,zlib.gzipSync(JSON.stringify(value)));fs.renameSync(temp,file);created++;
 }
 if(++count%20===0)console.log('metric native charts',count,'created',created);
}
console.log('metric native complete',count,'created',created);
