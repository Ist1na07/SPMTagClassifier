'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {analyzeMap}=require('../src/index.cjs');
const reports=[],errors=[],counts={},started=Date.now();
function files(dir){return fs.readdirSync(dir,{withFileTypes:true}).flatMap(d=>d.isDirectory()?files(path.join(dir,d.name)):d.name.endsWith('.osu')?[path.join(dir,d.name)]:[]);}
function finite(o){if(typeof o==='number')assert.ok(Number.isFinite(o),'nonfinite numeric output');else if(o&&typeof o==='object')Object.values(o).forEach(finite);}
for(const file of files('dataset')) {
 const begin=performance.now();
 try {
  const r=analyzeMap(fs.readFileSync(file,'utf8'));finite(r);
  const s=r.segments;assert.ok(s.length);assert.equal(s[0].start,r.map.firstNote);
  assert.equal(s.at(-1).end,r.modelInfo.variant==='legacy-stable'?r.map.lastNote:r.map.lastNote+1);
  for(let i=0;i<s.length;i++){
   assert.ok(s[i].end>s[i].start);if(i)assert.equal(s[i-1].end,s[i].start);
   assert.ok(Math.abs(Object.values(s[i].distribution??s[i].scores).reduce((a,b)=>a+b,0)-1)<1e-9);
  }
  assert.ok(Math.abs(Object.values(r.overall.composition??r.overall.weights).reduce((a,b)=>a+b,0)-1)<1e-9);
  counts[r.map.keys]=(counts[r.map.keys]||0)+1;
  reports.push({file,keys:r.map.keys,tags:r.overall.tags,sort:r.overall.sort,segments:s.length,warnings:r.map.warnings??[],ms:performance.now()-begin});
 } catch(e){errors.push({file,error:e.message});}
 if((reports.length+errors.length)%250===0)console.log('smoke',reports.length,'accepted',errors.length,'rejected');
}
fs.writeFileSync('research/results/corpus-smoke.json',JSON.stringify({counts,errors,elapsedMs:Date.now()-started,charts:reports},null,2));
console.log(JSON.stringify({counts,errors,elapsedMs:Date.now()-started}));
if(errors.some(e=>!(e.file.includes('bm6k_0778.osu')&&e.error==='InvalidHoldTail')))process.exitCode=1;
