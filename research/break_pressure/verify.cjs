'use strict';
const fs=require('node:fs'),readline=require('node:readline'),assert=require('node:assert/strict');
const api=require('../../src/index.cjs');
async function* lines(file){for await(const line of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity}))if(line.trim())yield JSON.parse(line);}
(async()=>{
 const old=lines('research/core_retune/results/deployment-refit-r1/config-6-fixed-prior-product.jsonl')[Symbol.asyncIterator](),out=fs.createWriteStream('.cache/break-pressure-product-r1.jsonl');let count=0,changed=0,protectedParents=0;const timings=[];
 for await(const r of lines('.cache/perception-final-resource.jsonl')){
  const before=(await old.next()).value;assert.equal(before.uid,r.uid);const text=fs.readFileSync(r.source,'utf8'),row={uid:r.uid,consumers:{}};
  for(const name of ['actions','detail','compact']){
   const rating=name==='actions'?null:r.ratings[name],start=performance.now(),a=api.analyzeMap(text,rating?{ratingEvidence:rating}:{});timings.push(performance.now()-start);
   const frames=a.segments.map(({start,end,tag,secondary,equivBpm,confidence})=>({start,end,tag,secondary,equivBpm,confidence}));
   if(name==='actions'){assert.deepEqual(a.overall.tags,before.consumers[name].tags);assert.deepEqual(a.overall.percentages,before.consumers[name].percentages);assert.equal(JSON.stringify(frames),JSON.stringify(before.consumers[name].segments));}
   else{changed+=JSON.stringify(frames)!==JSON.stringify(before.consumers[name].segments);protectedParents+=a.modelInfo.breakPolicy.protectedParents;}
   row.consumers[name]={tags:a.overall.tags,percentages:a.overall.percentages,segments:frames,breakPolicy:a.modelInfo.breakPolicy};
  }
  out.write(JSON.stringify(row)+'\n');if(++count%75===0)console.log('pressure native',count);
 }
 await new Promise(resolve=>out.end(resolve));assert((await old.next()).done);timings.sort((a,b)=>a-b);
 const result={records:count,changedRatedConsumerTimelines:changed,protectedParentConsumerPairs:protectedParents,actionConsumersUnchanged:count,analysisMs:{median:timings[Math.floor(timings.length*.5)],p95:timings[Math.floor(timings.length*.95)],max:timings.at(-1)}};
 fs.writeFileSync('research/break_pressure/results/native.json',JSON.stringify(result,null,2)+'\n');console.log(result);
})().catch(e=>{console.error(e);process.exitCode=1;});
