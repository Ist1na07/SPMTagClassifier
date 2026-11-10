'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict'),runtime=require('./runtime.cjs');
const input=JSON.parse(fs.readFileSync('.cache/local-feedback-parity.json','utf8')),results=[];
for(const name of ['competition','routing']){
  const ref=input.cases[name],actual=name==='competition'?runtime.competition(input.original,ref.x,ref.model,input.classes):runtime.routing(input.original,ref.x,ref.model,input.candidates,input.classes);
  let max=0;for(let i=0;i<actual.support.length;i++)for(let j=0;j<input.classes.length;j++)max=Math.max(max,Math.abs(actual.support[i][j]-ref.support[i][j]));
  assert(max<1e-9,name+' support mismatch '+max);
  assert.deepEqual(actual.frames.map(f=>f.tag),ref.frames.map(f=>f.tag),name+' primary tags');
  assert.deepEqual(actual.frames.map(f=>f.family),ref.frames.map(f=>f.family),name+' family');
  assert.deepEqual(actual.frames.map(f=>f.secondary??null),ref.frames.map(f=>f.secondary??null),name+' secondary');
  results.push({model:name,frames:actual.frames.length,maximumSupportError:max,primaryAndSecondary:'identical'});
}
fs.writeFileSync('research/local_feedback/results/resource-runtime-parity.json',JSON.stringify({scorerCalls:0,fits:0,results},null,2)+'\n');
console.log(results);
