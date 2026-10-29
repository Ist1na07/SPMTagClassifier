const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const api=require('..');
function chart(k){return `[General]\nMode:3\n[Difficulty]\nCircleSize:${k}\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n`+
 Array.from({length:160},(_,i)=>`${(i%k+.5)*512/k},192,${1000+i*125},1,0,0:0:0:0:`).join('\n');}
test('7K default preserves mature decisions and exposes corrected diagnostics',()=>{
 const text=chart(7),a=api.analyzeMap(text),b=api.legacy.analyzeMap(text);
 assert.equal(a.modelInfo.variant,'legacy-stable');assert.deepEqual(a.overall.tags,b.overall.tags);
 assert.deepEqual(a.segments.map(s=>[s.start,s.end,s.tag,s.confidence]),b.segments.map(s=>[s.start,s.end,s.tag,s.confidence]));
 for(const s of a.segments){assert.ok(s.evidence.occupancy>=0&&s.evidence.occupancy<=1);assert.ok(Math.abs(Object.values(s.distribution).reduce((a,b)=>a+b)-1)<1e-10);}
});
test('legacy cache contract and explicit raw legacy API remain unchanged',()=>{
 const text=chart(7);assert.deepEqual(api.buildCachedMap(text),api.legacy.buildCachedMap(text));
 assert.deepEqual(api.analyzeMap(text,{algorithm:'legacy'}),api.legacy.analyzeMap(text));
});
for(const k of [4,6,7])test(k+'K modern key path returns a continuous timeline and normalized scores',()=>{
 const r=api.analyzeMap(chart(k),{algorithm:'modern',speedRate:1.5});assert.equal(r.map.keys,k);
 assert.ok(r.segments.length);assert.ok(r.overall.tags.length);
 for(const s of r.segments)assert.ok(Math.abs(Object.values(s.scores).reduce((a,b)=>a+b)-1)<1e-10);
 assert.equal(r.segments[0].start,r.map.firstNote);assert.equal(r.segments.at(-1).end,r.map.lastNote+1);
});
test('browser IIFE is the actual tested standalone API, without Node globals',()=>{
 const box={};vm.runInNewContext(fs.readFileSync('dist/tag-engine.js','utf8'),box);
 for(const k of [4,6,7]){
  const text=chart(k);assert.equal(JSON.stringify(box.TagClassifier.analyzeMap(text)),JSON.stringify(api.analyzeMap(text)));
 }
});
test('CLI rejects invalid rate rather than producing nonfinite output',()=>{
 assert.throws(()=>api.analyzeMap(chart(4),{speedRate:NaN}),RangeError);
 assert.throws(()=>api.analyzeMap(chart(4),{algorithm:'not-a-model'}),/UnknownAlgorithm/);
});
