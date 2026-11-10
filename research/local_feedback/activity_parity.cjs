'use strict';
const fs=require('node:fs'),readline=require('node:readline'),assert=require('node:assert/strict');
const {apply}=require('./activity_runtime.cjs');
const root='.cache/local-feedback-activity-native-oracle.jsonl';
const meta=JSON.parse(fs.readFileSync(root+'.schema.json','utf8'));
async function* lines(file){for await(const line of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity}))if(line.trim())yield JSON.parse(line);}
(async()=>{
  const a=lines('.cache/perception-inputs-r1.jsonl')[Symbol.asyncIterator](),b=lines('.cache/local-feedback-routing-r3.jsonl')[Symbol.asyncIterator]();let records=0,frames=0,maxFeatureError=0,maxProbabilityError=0,maxSupportError=0;const begin=performance.now();
  for await(const row of lines(root)){
    const input=(await a.next()).value,expert=(await b.next()).value;assert.equal(row.uid,input.uid);assert.equal(row.uid,expert.uid);
    const r=apply(meta.model,input.map,input.frames,expert.candidates,meta.classes);assert.equal(r.frames.length,row.frames.length);
    r.frames.forEach((f,j)=>{
      const old=row.frames[j];for(const key of ['start','end','family','tag','secondary'])assert.equal(f[key],old[key],row.uid+'/'+j+'/'+key);
      meta.features.forEach((key,k)=>{maxFeatureError=Math.max(maxFeatureError,Math.abs(r.inputs[j][key]-row.inputs[j][k]));});
      r.probabilities[j].forEach((v,k)=>maxProbabilityError=Math.max(maxProbabilityError,Math.abs(v-row.probabilities[j][k])));
      r.support[j].forEach((v,k)=>maxSupportError=Math.max(maxSupportError,Math.abs(v-row.support[j][k])));
    });
    records++;frames+=r.frames.length;if(records%75===0)console.log('native activity parity',records,frames);
  }
  assert.equal(records,meta.records);assert.equal(frames,meta.frames);assert((await a.next()).done);assert((await b.next()).done);
  assert(maxFeatureError<1e-10);assert(maxProbabilityError<1e-10);assert(maxSupportError<1e-10);
  const result={candidate:meta.candidate,records,frames,maxFeatureError,maxProbabilityError,maxSupportError,seconds:(performance.now()-begin)/1000,scorerCalls:0,scope:'Research-native physical extraction and JS parameter consumer; not production integration/render acceptance'};
  fs.writeFileSync('research/local_feedback/results/activity-native-parity-r6.json',JSON.stringify(result,null,2)+'\n');console.log(result);
})().catch(e=>{console.error(e);process.exitCode=1;});
