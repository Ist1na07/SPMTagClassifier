'use strict';
const fs=require('node:fs'),readline=require('node:readline'),assert=require('node:assert/strict');
const {apply}=require('./joint_runtime.cjs');
const root='.cache/local-feedback-joint-native-oracle.jsonl',meta=JSON.parse(fs.readFileSync(root+'.schema.json','utf8'));
async function* lines(file){for await(const line of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity}))if(line.trim())yield JSON.parse(line);}
(async()=>{
  const physical=lines('.cache/perception-inputs-r1.jsonl')[Symbol.asyncIterator](),experts=lines('.cache/local-feedback-routing-r3.jsonl')[Symbol.asyncIterator]();let records=0,frames=0,maxFeatureError=0,maxProbabilityError=0,maxSupportError=0;const begin=performance.now();
  for await(const row of lines(root)){
    const input=(await physical.next()).value,expert=(await experts.next()).value;assert.equal(row.uid,input.uid);assert.equal(row.uid,expert.uid);const out=apply(meta.model,input.map,input.frames,expert.candidates,meta.classes);assert.equal(out.frames.length,row.frames.length);
    out.frames.forEach((f,j)=>{
      for(const key of ['start','end','family','tag','secondary'])assert.equal(f[key],row.frames[j][key],row.uid+'/'+j+'/'+key);
      meta.features.forEach((key,k)=>maxFeatureError=Math.max(maxFeatureError,Math.abs(out.inputs[j][key]-row.inputs[j][k])));
      out.probabilities[j].forEach((v,k)=>maxProbabilityError=Math.max(maxProbabilityError,Math.abs(v-row.probabilities[j][k])));
      out.support[j].forEach((v,k)=>maxSupportError=Math.max(maxSupportError,Math.abs(v-row.support[j][k])));
    });records++;frames+=out.frames.length;if(records%75===0)console.log('joint native parity',records,frames);
  }
  assert.equal(records,meta.records);assert.equal(frames,meta.frames);assert((await physical.next()).done);assert((await experts.next()).done);assert(maxFeatureError<1e-10);assert(maxProbabilityError<1e-10);assert(maxSupportError<1e-10);
  const receipt={candidate:meta.candidate,records,frames,maxFeatureError,maxProbabilityError,maxSupportError,seconds:(performance.now()-begin)/1000,scorerCalls:0,scope:'Research-native actual physical extraction and JS learned route/type consumer, not product adoption or render acceptance'};fs.writeFileSync('research/local_feedback/results/joint-native-parity-r7.json',JSON.stringify(receipt,null,2)+'\n');console.log(receipt);
})().catch(e=>{console.error(e);process.exitCode=1;});
