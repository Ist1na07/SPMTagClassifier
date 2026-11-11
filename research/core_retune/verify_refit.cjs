'use strict';
const fs=require('node:fs'),readline=require('node:readline'),assert=require('node:assert/strict');
const api=require('../../src/index.cjs'),{createProbe}=require('./native_refit_probe.cjs');
const originalModels=JSON.parse(fs.readFileSync('.cache/core-retune-parents-r1.jsonl.gz.schema.json','utf8')).model;
const classicControl=createProbe(originalModels);
async function* lines(file){for await(const line of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity}))if(line.trim())yield JSON.parse(line);}
(async()=>{
 const expected=lines('research/core_retune/results/deployment-refit-r1/config-6-fixed-prior-product.jsonl')[Symbol.asyncIterator]();
 let count=0,error=0,classic=0;const times=[];
 for await(const row of lines('.cache/perception-final-resource.jsonl')){
  const e=(await expected.next()).value;assert.equal(e.uid,row.uid);const text=fs.readFileSync(row.source,'utf8');
  for(const name of ['actions','detail','compact']){
   const rating=name==='actions'?undefined:row.ratings[name],start=performance.now(),r=api.analyzeMap(text,rating?{ratingEvidence:rating}:{});times.push(performance.now()-start);
   assert.equal(r.modelInfo.localExpert,'original56-joint-fixed-ridge01');
   assert.deepEqual(r.overall.tags,e.consumers[name].tags);
   for(const [c,v]of Object.entries(e.consumers[name].percentages)){error=Math.max(error,Math.abs(r.overall.percentages[c]-v));assert(Math.abs(r.overall.percentages[c]-v)<1e-8);}
   const frames=r.segments.map(({start,end,tag,secondary,equivBpm,confidence})=>({start,end,tag,secondary,equivBpm,confidence}));assert.deepEqual(JSON.parse(JSON.stringify(frames)),e.consumers[name].segments);
  }
  const r=api.analyzeMap(text,{aggregation:'classic'}),control=classicControl.analyzeMap(text,{aggregation:'classic'});
  assert.deepEqual(r.overall,control.overall);assert.deepEqual(r.segments,control.segments);classic++;
  if(++count%75===0)console.log('static final',count);
 }
 assert((await expected.next()).done);times.sort((a,b)=>a-b);const result={records:count,consumers:3,maxPercentagePointError:error,classicFullOutputMatches:classic,analysisMs:{median:times[Math.floor(times.length*.5)],p95:times[Math.floor(times.length*.95)],max:times.at(-1)}};
 fs.writeFileSync('research/core_retune/results/deployment-refit-r1/static-native.json',JSON.stringify(result,null,2)+'\n');console.log(result);
})().catch(e=>{console.error(e);process.exitCode=1;});
