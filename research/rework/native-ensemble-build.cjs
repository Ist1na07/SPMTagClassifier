'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const {records7}=require('./data.cjs');
const {SPEC}=require('./native.cjs');
const {combineSegments}=require('./native-ensemble.cjs');
const root='.cache/rework-native-ensemble-r1',directory=path.join(root,'mirror_ensemble');fs.mkdirSync(directory,{recursive:true});
const spec={id:'native-ensemble-r1',nativeSpecification:SPEC,
 composition:'pointwise arithmetic mean of the two native decision distributions on the union time partition; same parameter tuple for both views',
 eligibility:'primary Break remains excluded from aggregation; fractional support/disagreement remains visible',
 missingCoverage:'use the available-view mean when one branch is uncovered; emit no label when both are uncovered',
 smoothing:'coherent only; does not define a historical legacy smoothing path',
 cost:'two full native analyses per chart; exactly invariant to exchanging original/mirrored views'};
const manifest=path.join(root,'spec.json');
if(fs.existsSync(manifest))assert.deepEqual(JSON.parse(fs.readFileSync(manifest,'utf8')),spec);else fs.writeFileSync(manifest,JSON.stringify(spec,null,2)+'\n');
let count=0;
for(const r of records7()){
 const name=r.uid.replaceAll('/','__')+'.json.gz',target=path.join(directory,name);
 if(fs.existsSync(target)){count++;continue;}
 const a=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join('.cache/rework-native-r1/original',name))));
 const b=JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join('.cache/rework-native-mirror-r1',name))));
 const partitions=a.partitions.map((p,i)=>({boundary:p.boundary,cases:p.cases.map((c,j)=>({head:c.head,raw:combineSegments(c.raw,b.partitions[i].cases[j].raw)}))}));
 const value={id:spec.id,uid:r.uid,treatment:'mirror_ensemble',firstNote:a.firstNote,lastNote:a.lastNote,partitions};
 const temporary=target+'.tmp';fs.writeFileSync(temporary,zlib.gzipSync(JSON.stringify(value)));fs.renameSync(temporary,target);count++;
}
console.log('ensemble native complete',count);
