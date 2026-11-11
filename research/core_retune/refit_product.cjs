'use strict';
const fs=require('node:fs'),readline=require('node:readline'),path=require('node:path'),assert=require('node:assert/strict');
const {createProbe}=require('./native_refit_probe.cjs');
async function* lines(file){for await(const line of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity}))if(line.trim())yield JSON.parse(line);}
(async()=>{
 const file=process.argv[2];if(!file)throw Error('Provide full-data refit JSON');
 const preservePrior=process.argv.includes('--fixed-prior');
 const candidate=JSON.parse(fs.readFileSync(file,'utf8')),api=createProbe(candidate.models,{preservePrior}),stem=file.replace(/\.json$/,'')+(preservePrior?'-fixed-prior':''),target=stem+'-product.jsonl';
 if(fs.existsSync(target))throw Error('Preserve completed/partial replay; use a separate output after investigating');
 const out=fs.createWriteStream(target),old=lines('.cache/core-retune-product-candidate.jsonl')[Symbol.asyncIterator]();let count=0,changed=0;const times=[];
 for await(const r of lines('.cache/perception-final-resource.jsonl')){
  const previous=(await old.next()).value;assert.equal(previous.uid,r.uid);const text=fs.readFileSync(r.source,'utf8'),row={uid:r.uid,consumers:{}};
  for(const name of ['actions','detail','compact']){
   const rating=name==='actions'?undefined:r.ratings[name],start=performance.now(),actual=api.analyzeMap(text,rating?{ratingEvidence:rating}:{});times.push(performance.now()-start);
   const frames=actual.segments.map(({start,end,tag,secondary,equivBpm,confidence})=>({start,end,tag,secondary,equivBpm,confidence}));
   if(JSON.stringify(frames)!==JSON.stringify(previous.consumers[name].segments))changed++;
   row.consumers[name]={tags:actual.overall.tags,percentages:actual.overall.percentages,segments:frames};
  }
  out.write(JSON.stringify(row)+'\n');if(++count%75===0)console.log(path.basename(file),'actual product',count);
 }
 assert((await old.next()).done);await new Promise(resolve=>out.end(resolve));times.sort((a,b)=>a-b);
 const receipt={candidate:file,preservePrior,scope:'Actual current public entry point with in-memory original W/b replacement; '+(preservePrior?'keeps original expert classic prior as used during training':'recomputes classic priors')+', supports, cadence and three actual rating consumers. Complete-data development performance.',records:count,changedConsumerTimelines:changed,analysisMs:{median:times[Math.floor(times.length*.5)],p95:times[Math.floor(times.length*.95)],max:times.at(-1)}};
 fs.writeFileSync(stem+'-native.json',JSON.stringify(receipt,null,2)+'\n');console.log(receipt);
})().catch(e=>{console.error(e);process.exitCode=1;});
