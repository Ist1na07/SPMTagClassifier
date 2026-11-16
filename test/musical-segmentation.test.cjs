'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {parseOsu,prepare}=require('../src/parser.cjs'),{measureLines,musicalSegments}=require('../src/musical-segmentation.cjs');
const engine=require('../src/generated/overlay-engine.cjs');
function chart(timing='1000,500,4,1,0,100,1,0',times=Array.from({length:161},(_,i)=>1000+i*100)){
 return '[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n'+timing+'\n[HitObjects]\n'+times.flatMap((t,i)=>[i%3,4+i%3].map(c=>`${Math.floor((c+.5)*512/7)},192,${t},1,0,0:0:0:0:`)).join('\n');
}
function input(ctx,rate=1){
 const n=Math.ceil(16000/rate/250),windows=Array.from({length:n},(_,i)=>({t:1000/rate+i*250})),norm={featureNames:['a'],values:windows.map((_,i)=>[i<16/rate?0:1])};
 return{segments:[{start:1000/rate,end:5150/rate,windowRange:[0,17]},{start:5150/rate,end:17000/rate,windowRange:[17,64]}],windows,norm,speedRate:rate};
}
test('meter, red-line phase reset, inherited timing and omit-first-bar semantics are preserved',()=>{
 const ctx=prepare(parseOsu(chart('1000,500,3,1,0,100,1,0\n1600,-50,7,1,0,100,0,0\n6000,400,7,1,0,100,1,8')));
 assert.equal(ctx.map.timingPoints.length,2);assert.equal(ctx.map.timingPoints[0].meter,3);assert.equal(ctx.map.timingPoints[1].omitFirstBarLine,true);
 assert.deepEqual(measureLines(ctx.map,1000,15000).map(p=>Math.round(p.t)),[1000,2500,4000,5500,8800,11600,14400]);
});
test('nearby musical cuts move before reclassification and respect the native rate clock',()=>{
 for(const rate of [.75,1,1.5]){
  const ctx=prepare(parseOsu(chart(),{speedRate:rate})),out=musicalSegments(ctx,input(ctx,rate));
  assert(Math.abs(out[0].end-5000/rate)<1e-8);assert(Math.abs(out[1].start-5000/rate)<1e-8);assert.equal(out[0].start,ctx.map.firstNote);assert.equal(out.at(-1).end,ctx.map.lastNote);
  assert.deepEqual(out.map(s=>s.windowRange),[[0,Math.round(16/rate)],[Math.round(16/rate),64]]);
  assert.equal(out[0].musicalEnd.originalTime,5150/rate);
 }
});
test('physical head gaps and headless held regions are not blindly snapped to a bar',()=>{
 const ctx=prepare(parseOsu(chart(undefined,[1000,2000,3000,4000,9000,10000,11000,12000,17000]))),i=input(ctx);i.segments[0].end=7150;i.segments[1].start=7150;
 assert.equal(musicalSegments(ctx,i)[0].end,7150);
});
test('missing timing makes the musical prior a no-op, not an invented 120-BPM grid',()=>{
 const ctx=prepare(parseOsu(chart(''))),i=input(ctx);assert.strictEqual(musicalSegments(ctx,i),i.segments);
});
test('musical segmentation never creates overlap, short manufactured parents or drops the envelope',()=>{
 const ctx=prepare(parseOsu(chart())),i=input(ctx);i.segments=[{start:1000,end:4100,windowRange:[0,12]},{start:4100,end:7200,windowRange:[12,25]},{start:7200,end:17000,windowRange:[25,64]}];
 for(const maxMoveBeats of [.5,1,2]){
  const out=musicalSegments(ctx,i,{maxMoveBeats});assert.equal(out[0].start,1000);assert.equal(out.at(-1).end,17000);
  out.forEach((s,j)=>{assert(s.end-s.start>=3000);if(j)assert.equal(s.start,out[j-1].end);});
 }
});
test('maintained static engine recomputes raw extras on new cuts while sharing old windows and prior',()=>{
 const text=chart(),ctx=prepare(parseOsu(text));
 const cm=engine.buildCachedMap(text,{_musicalSegmenter:i=>musicalSegments(ctx,{...i,segments:input(ctx).segments})});
 assert.equal(cm.rawSegs[0].end,5000);assert.equal(cm.extras[0].rowCount,40);
 assert.strictEqual(cm.windows,cm.referenceCm.windows);assert.notDeepEqual(cm.rawSegs,cm.referenceCm.rawSegs);
 assert.equal(cm.referenceCm.rawSegs[0].start,ctx.map.firstNote);
});


test('unmoved endpoints retain their original sequential feature-window assignments',()=>{
 const ctx=prepare(parseOsu(chart())),i=input(ctx);i.segments[0].windowRange=[0,18];i.segments[1].windowRange=[18,65];
 const out=musicalSegments(ctx,i,{maxMoveBeats:0});assert.deepEqual(out.map(s=>s.windowRange),[[0,18],[18,65]]);
});


test('a truncated fractional downbeat head belongs to the new section, not the previous one',()=>{
 const beat=60000/133,nominal=1000+8*beat,cut=Math.floor(nominal);
 const times=Array.from({length:161},(_,i)=>1000+i*100).filter(t=>Math.abs(t-cut)>20);times.push(cut);times.sort((a,b)=>a-b);
 for(const rate of [1,.75,1.5]){
  const text=chart(`1000,${beat},4,1,0,100,1,0`,times),ctx=prepare(parseOsu(text,{speedRate:rate})),i=input(ctx,rate);
  i.segments[0].end=(nominal+100)/rate;i.segments[1].start=(nominal+100)/rate;
  const out=musicalSegments(ctx,i);assert(Math.abs(out[0].end-cut/rate)<1e-8);assert.equal(out[1].start,out[0].end);
  assert(Math.abs(out[0].musicalEnd.nominalTime-nominal/rate)<1e-8);
  const downbeat=ctx.rows.find(r=>Math.abs(r.t-cut/rate)<1e-8);assert(downbeat);assert(downbeat.t>=out[1].start);assert(!(downbeat.t<out[0].end));
 }
});
