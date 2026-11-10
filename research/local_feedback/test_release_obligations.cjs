'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {prepare}=require('../../src/parser.cjs');
const {evidence}=require('./release_obligations.cjs');
function ctx(notes,od=8){notes.sort((a,b)=>a.t-b.t||a.col-b.col);return prepare({keys:7,notes,meta:{od},firstNote:notes[0].t,lastNote:Math.max(...notes.map(n=>n.lnEnd??n.t)),timingPoints:[],breaks:[]});}
const hold=(t,col,lnEnd)=>({t,col,lnEnd});
test('simultaneous tails have no retained-finger release obligation',()=>{
  const f=evidence(ctx([hold(0,0,1000),hold(0,1,1000)]));
  for(const [a,b]of [[0,500],[500,1200]]){const v=f(a,b);assert.equal(v[0],0);assert.equal(v[1],0);assert.equal(v[3],0);assert.equal(v[4],0);}
});
test('small stagger tends continuously to simultaneous, body sees future tails',()=>{
  const samples=[0,.01,1,8,24,72,216].map(gap=>evidence(ctx([hold(0,0,1000),hold(0,1,1000+gap)]))(0,500));
  assert.equal(samples[0][3],0);for(let i=1;i<samples.length;i++)assert(samples[i][3]>samples[i-1][3]);
  assert(samples[1][3]<1e-4);assert(samples.at(-1)[4]>0);assert(samples.every(v=>v[0]===0));
});
test('own-column future-head guidance suppresses independent obligation',()=>{
  const base=[hold(0,0,1000),hold(0,1,1500)],unguided=evidence(ctx([...base]))(0,500),guided=evidence(ctx([...base,hold(1080,0,null)]))(0,500);
  assert.equal(unguided[3],guided[3]);assert(guided[4]<unguided[4]);
});
test('body integral is additive and lane-mirror invariant',()=>{
  const notes=[hold(0,0,1000),hold(200,1,1800),hold(900,3,2100)];const f=evidence(ctx(notes.map(n=>({...n})))),g=evidence(ctx(notes.map(n=>({...n,col:6-n.col}))));
  const a=f(0,700),b=f(700,2200),c=f(0,2200),m=g(0,2200);
  for(let j=0;j<5;j++){assert(Math.abs(c[j]-(a[j]*700+b[j]*1500)/2200)<1e-12);assert(Math.abs(c[j]-m[j])<1e-12);}
});
test('shared helper reproduces accepted cached extraction exactly',()=>{
  const schema=JSON.parse(fs.readFileSync('.cache/local-feedback-activity-r6b.jsonl.schema.json','utf8'));
  const physical=fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n');
  const cached=fs.readFileSync('.cache/local-feedback-activity-r6b.jsonl','utf8').trim().split('\n');
  for(const i of [0,16,55,123,234,348,355,400,410]){
    const row=JSON.parse(physical[i]),old=JSON.parse(cached[i]),f=evidence(prepare(row.map));assert.equal(row.uid,old.uid);
    for(let w=0;w<3;w++)for(let j=0;j<old.frames.length;j+=17){const frame=old.frames[j],mid=(frame.start+frame.end)/2,a=Math.max(row.map.firstNote,mid-schema.widths[w]/2),b=Math.max(a+1,Math.min(row.map.lastNote,mid+schema.widths[w]/2));const v=f(a,b);
      v.forEach((x,k)=>assert(Math.abs(x-old.values[w][j][36+k])<1e-12));
    }
  }
});
