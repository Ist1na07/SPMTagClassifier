'use strict';
const fs=require('node:fs'),path=require('node:path'),zlib=require('node:zlib');
const assert=require('node:assert/strict'),{test}=require('node:test');
const {metricApi}=require('./metric.cjs'),{ROOT,records7}=require('./data.cjs');
const {context}=require('./legacy-lab.cjs'),{extractNative}=require('./native.cjs');
const record=records7().find(r=>r.dataset==='manual'),text=fs.readFileSync(record.path,'utf8');
function cached(root,source){return JSON.parse(zlib.gunzipSync(fs.readFileSync(path.join(ROOT,root,source,record.uid.replaceAll('/','__')+'.json.gz')))).partitions;}
test('both metric endpoints reproduce the complete existing native consumer',()=>{
 for(const release of[null,40])for(const strength of[0,1]){
  const api=metricApi(release,strength),actual=extractNative(api,context(api.lab,text));
  const source=release===null?(strength?'logical_neutral_scale':'logical_neutral'):(strength?'full_40':'logical_release_40');
  const root=source==='logical_neutral_scale'?'.cache/rework-native-factorial-r1':'.cache/rework-native-r1';
  assert.deepEqual(actual,cached(root,source));
 }
});
test('shrinkage changes only the four rate coordinates of the segment signal',()=>{
 const a=metricApi(null,0),ctx=context(a.lab,text),left=a.lab.normalizeWindows(ctx.windows),right=metricApi(null,1).lab.normalizeWindows(ctx.windows);
 const rates=new Set(['cutDemand','jackSpeedBpm','nps','tailRate']);
 for(const strength of[.05,.2]){
  const value=metricApi(null,strength).lab.normalizeWindows(ctx.windows);
  for(let i=0;i<value.values.length;i++)for(let j=0;j<value.featureNames.length;j++){
   const x=value.values[i][j],expected=(1-strength)*left.values[i][j]+strength*right.values[i][j];
   assert.equal(x,expected);assert.ok(x>=-1-1e-12&&x<=2+1e-12);
   if(!rates.has(value.featureNames[j]))assert.ok(Math.abs(x-left.values[i][j])<1e-12);
  }
 }
});
