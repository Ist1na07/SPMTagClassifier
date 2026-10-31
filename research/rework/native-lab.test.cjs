'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {loadLab,context,inspect,inspectPrepared}=require('./legacy-lab.cjs');
const {records7}=require('./data.cjs');
const {nativeApi,decisionSupport,CLASSES}=require('./native.cjs');

test('prepared native sweep equals the complete entry path for every changed boundary configuration',()=>{
 const api=loadLab(),record=records7().find(r=>r.uid==='manual/AA'),text=fs.readFileSync(record.path,'utf8');
 const prepared=context(api.lab,text),base=inspectPrepared(api,prepared);
 for(const minSegLenMs of [1500,3000,6000])for(const segProminenceAbs of [.025,.05,.1]){
  Object.assign(api.lab.CONFIG,{minSegLenMs,segProminenceAbs});
  const reused=inspectPrepared(api,prepared),fresh=inspect(api,text);
  assert.deepEqual(reused.classified,fresh.classified);assert.deepEqual(reused.smoothed,fresh.smoothed);
  assert.deepEqual(reused.segments,fresh.segments);
 }
 Object.assign(api.lab.CONFIG,{minSegLenMs:3000,segProminenceAbs:.05});
 assert.deepEqual(inspectPrepared(api,prepared).classified,base.classified,'no configuration state leaks');
});
test('decision support respects explicit Break and wildcard-rejection gates',()=>{
 const p=decisionSupport({tag:'density',family:'ln',scores:{'ln wildcard':.8,density:.4,release:.1}});
 assert.equal(CLASSES[p.indexOf(Math.max(...p))],'density');assert.equal(p[CLASSES.indexOf('ln wildcard')],0);
 assert.equal(decisionSupport({tag:'break',family:'break',scores:{density:1}})[CLASSES.indexOf('break')],1);
});
test('native coverage correction assigns discarded sub-50 ms fragments without inventing overlap',()=>{
 const original=loadLab(),fixed=nativeApi('coverage');let missingBefore=0;
 for(const record of records7().filter(r=>r.dataset==='manual')){
  const text=fs.readFileSync(record.path,'utf8'),a=inspect(original,text),b=inspect(fixed,text);
  const span=a.map.lastNote-a.map.firstNote;
  missingBefore+=span-a.segments.reduce((sum,s)=>sum+s.end-s.start,0);
  assert.equal(b.segments[0].start,b.map.firstNote);assert.equal(b.segments.at(-1).end,b.map.lastNote);
  for(let i=1;i<b.segments.length;i++)assert.equal(b.segments[i].start,b.segments[i-1].end);
 }
 assert.equal(missingBefore,37.5,'exercise the actual diagnosed short fragment');
});
