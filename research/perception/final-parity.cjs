'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict'),readline=require('node:readline');
const api=require('../../src/index.cjs'),core=require('../../src/generated/overlay-engine.cjs');
async function main(){
 let cases=0,values=0,maxError=0,rawFrames=0,elapsed=performance.now();
 const timings=[];
 for await(const line of readline.createInterface({input:fs.createReadStream('.cache/perception-final-resource.jsonl')})){
  const c=JSON.parse(line),text=fs.readFileSync(c.source,'utf8');
  const classic=api.analyzeMap(text,{aggregation:'classic'});
  assert.deepEqual([...classic.overall.tags].sort(),[...c.baseline.tags].sort(),c.uid+'/classic');
  const raw=core.analyzeMap(text,{_rawEvidence:true});
  assert.equal(raw.debug.rawSegments.length,c.frames.length,c.uid+'/raw length');
  raw.debug.rawSegments.forEach((s,j)=>{const b=c.frames[j];for(const k of ['start','end','tag','family'])assert.equal(s[k],b[k],c.uid+'/'+k);rawFrames++;});
  const withoutRaw={...raw,debug:{...raw.debug}};delete withoutRaw.debug.rawSegments;
  assert.deepEqual(core.analyzeMap(text),withoutRaw,c.uid+'/internal evidence flag preserves classic output');
  for(const name of ['actions','detail','compact']){
   const start=performance.now(),r=api.analyzeMap(text,name==='actions'?{}:{ratingEvidence:c.ratings[name]});timings.push(performance.now()-start);
   assert.deepEqual([...r.overall.tags].sort(),[...c.expected[name].tags].sort(),c.uid+'/'+name+'/tags');
   for(const [tag,value]of Object.entries(c.expected[name].percentages)){
    const error=Math.abs(r.overall.percentages[tag]-value);maxError=Math.max(maxError,error);values++;
    assert(error<1e-8,c.uid+'/'+name+'/'+tag+': '+error);
   }
   assert.deepEqual(r.segments.map(s=>[s.start,s.end,s.tag,s.confidence,s.equivBpm]),classic.segments.map(s=>[s.start,s.end,s.tag,s.confidence,s.equivBpm]),c.uid+'/local unchanged');
   assert(!Object.hasOwn(r.debug,'rawSegments'));assert(r.segments.every(s=>s.aggregationWeight>=0&&s.aggregationWeight<=1+1e-10));
  }
  cases++;if(cases%100===0)console.log('final runtime parity',cases);
 }
 timings.sort((a,b)=>a-b);
 const receipt={cases,profiles:3,comparedPercentages:values,maxPercentagePointError:maxError,rawFrames,localTimeline:'unchanged vs v1.2.2 classic',seconds:(performance.now()-elapsed)/1000,analysisMs:{median:timings[Math.floor(timings.length*.5)],p95:timings[Math.floor(timings.length*.95)],max:timings.at(-1)}};
 fs.writeFileSync('research/perception/results/final-runtime/parity.json',JSON.stringify(receipt,null,2)+'\n');console.log(receipt);
}
main().catch(e=>{console.error(e);process.exitCode=1;});
