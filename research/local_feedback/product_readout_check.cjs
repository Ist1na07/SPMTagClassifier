'use strict';
const fs=require('node:fs'),readline=require('node:readline'),assert=require('node:assert/strict');
const api=require('../../src/index.cjs'),{parseOsu,prepare}=require('../../src/parser.cjs'),perception=require('../../src/perception.cjs');
const {refine}=require('../../src/segment-readout.cjs');
const profiles=require('../../src/models/perception7k.json').profiles;
async function* lines(file){for await(const s of readline.createInterface({input:fs.createReadStream(file),crlfDelay:Infinity}))if(s.trim())yield JSON.parse(s);}
(async()=>{
  const masks=lines('.cache/local-feedback-static-cut-r1.jsonl')[Symbol.asyncIterator](),out=fs.createWriteStream('.cache/local-feedback-product-readout.jsonl');let count=0,max=0,frames=0,exclusions=0;const times=[],begin=performance.now();
  for await(const c of lines('.cache/perception-final-resource.jsonl')){
    const mask=(await masks.next()).value;assert.equal(mask.uid,c.uid);
    const text=fs.readFileSync(c.source,'utf8'),map=parseOsu(text),ctx=prepare(map),expectedFrames=[];
    c.frames.forEach((s,parent)=>{for(let start=s.start;start<s.end;start+=500)expectedFrames.push({...s,start,end:Math.min(s.end,start+500),parent});});
    assert.deepEqual(expectedFrames.map(s=>[s.start,s.end,s.parent]),mask.frames.map(s=>[s.start,s.end,s.parent]));
    const native=refine(ctx,c.frames,profiles.actions.classes);frames+=native.frames.length;
    native.frames.forEach((s,j)=>{
      const f=expectedFrames[j],certified=f.family==='rc'&&mask.masks[2][j];
      const scores=f.scores??{[f.tag]:1},div=Math.max(1,...Object.values(scores));
      profiles.actions.classes.forEach((tag,k)=>assert(Math.abs(native.support[j][k]-(f.family==='break'||certified&&['dense chordstream','fast chordstream'].includes(tag)?0:Math.min(.999,Math.max(0,scores[tag]??0)/div)))<1e-14,c.uid+'/physical certificate'));
    });exclusions+=native.exclusions;
    const row={uid:c.uid,source:c.source,consumers:{}};
    for(const name of ['actions','detail','compact']){
      const rating=name==='actions'?undefined:c.ratings[name],start=performance.now(),r=api.analyzeMap(text,rating?{ratingEvidence:rating}:{});times.push(performance.now()-start);
      const expected=perception.infer(map,expectedFrames,rating,profiles[name],{context:ctx,baseline:c.baseline,_branchSupport:native.support});
      assert.deepEqual([...r.overall.tags].sort(),[...expected.overall.tags].sort(),c.uid+'/'+name+'/tags');
      for(const tag of profiles[name].classes){const e=Math.abs(r.overall.percentages[tag]-expected.overall.percentages[tag]);max=Math.max(max,e);assert(e<1e-8,c.uid+'/'+name+'/'+tag);}
      assert(!r.debug.rawSegments&&!r.debug.readoutSupport&&!r.debug.classicPrior);
      row.consumers[name]={tags:r.overall.tags,percentages:r.overall.percentages,segments:r.segments.map(({start,end,tag,secondary,equivBpm,confidence})=>({start,end,tag,secondary,equivBpm,confidence}))};
    }
    out.write(JSON.stringify(row)+'\n');count++;if(count%75===0)console.log('product readout check',count);
  }
  assert((await masks.next()).done);out.end();times.sort((a,b)=>a-b);
  const receipt={scope:'Actual maintained static library, complete411 inventory, independent cached hand-certificate oracle plus real released Detailed/Compact curves; no new rating calls',records:count,frames,exclusions,maxPercentagePointError:max,seconds:(performance.now()-begin)/1000,analysisMs:{median:times[Math.floor(times.length*.5)],p95:times[Math.floor(times.length*.95)],max:times.at(-1)}};
  fs.writeFileSync('research/local_feedback/results/product-readout-native.json',JSON.stringify(receipt,null,2));console.log(receipt);
})().catch(e=>{console.error(e);process.exitCode=1;});
