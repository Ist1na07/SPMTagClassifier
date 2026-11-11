'use strict';
// Actual maintained public/browser consumer, original physical input/rating
// fields, independent cached-cm classifier and current effective classic prior.
const fs=require('node:fs'),zlib=require('node:zlib'),readline=require('node:readline'),assert=require('node:assert/strict');
const api=require('../../src/index.cjs'),perception=require('../../src/perception.cjs'),{parseOsu,prepare}=require('../../src/parser.cjs');
const {refine}=require('../../src/segment-readout.cjs');
const {loadLab}=require('../rework/legacy-lab.cjs'),{actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const op=require('../../src/models/overlay7k.json'),profiles=require('../../src/models/perception7k.json').profiles;
async function* lines(path){for await(const line of readline.createInterface({input:fs.createReadStream(path),crlfDelay:Infinity}))if(line.trim())yield JSON.parse(line);}
(async()=>{
 const apiLab=loadLab({patches:[actionRatePatch(op.gate),{apply(s,r){return r(s,'const easyAmbiguous = seg.confidence < 0.5 &&','const easyAmbiguous = seg.tag !== "ln wildcard" && seg.confidence < 0.5 &&');}}]});
 const inputs=new Map(zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r2.jsonl.gz')).toString().trim().split('\n').map(s=>{const r=JSON.parse(s);return[r.uid,r];}));
 const old=lines('.cache/local-feedback-product-readout.jsonl')[Symbol.asyncIterator](),out=fs.createWriteStream('.cache/core-retune-product-candidate.jsonl');let count=0,error=0,changedPriors=0,changedParents=0;const times=[],changedMaps=[];
 for await(const r of lines('.cache/perception-final-resource.jsonl')){
  const previous=(await old.next()).value;assert.equal(previous.uid,r.uid);const input=inputs.get(r.uid),raw=apiLab.lab.classifySegments(input.cm,{scorer:'model'});let changes=0;
  raw.forEach((s,j)=>changes+=s.tag!==input.parents[j].baseline.tag);changedParents+=changes;
  const baseline=apiLab.lab.aggregateOverall(apiLab.lab.smoothSegments(structuredClone(raw)),op.aggregation);changedPriors+=JSON.stringify([...baseline.tags].sort())!==JSON.stringify([...r.baseline.tags].sort());
  const text=fs.readFileSync(r.source,'utf8'),map=parseOsu(text),ctx=prepare(map),row={uid:r.uid,source:r.source,consumers:{},changedParents:changes};
  for(const name of ['actions','detail','compact']){
   const rating=name==='actions'?undefined:r.ratings[name],start=performance.now(),actual=api.analyzeMap(text,rating?{ratingEvidence:rating}:{});times.push(performance.now()-start);
   const readout=refine(ctx,raw,profiles[name].classes),expected=perception.infer(map,readout.frames,rating,profiles[name],{context:ctx,baseline,_branchSupport:readout.support});assert.deepEqual([...actual.overall.tags].sort(),[...expected.overall.tags].sort(),r.uid+'/'+name+'/effective prior');
   for(const tag of profiles[name].classes){const e=Math.abs(actual.overall.percentages[tag]-expected.overall.percentages[tag]);error=Math.max(error,e);assert(e<1e-8,r.uid+'/'+name+'/'+tag);}
   row.consumers[name]={tags:actual.overall.tags,percentages:actual.overall.percentages,segments:actual.segments.map(({start,end,tag,secondary,equivBpm,confidence})=>({start,end,tag,secondary,equivBpm,confidence}))};
  }
  if(changes)changedMaps.push({uid:r.uid,parents:changes});out.write(JSON.stringify(row)+'\n');if(++count%75===0)console.log('actual candidate',count);
 }
 assert((await old.next()).done);await new Promise(resolve=>out.end(resolve));times.sort((a,b)=>a-b);const result={records:count,changedParents,changedMaps,changedClassicTagPriors:changedPriors,maxPercentagePointError:error,analysisMs:{median:times[Math.floor(times.length*.5)],p95:times[Math.floor(times.length*.95)],max:times.at(-1)}};fs.writeFileSync('research/core_retune/results/product-native.json',JSON.stringify(result,null,2)+'\n');console.log(result);
})().catch(e=>{console.error(e);process.exitCode=1;});
