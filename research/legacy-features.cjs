const fs=require('node:fs'),path=require('node:path');
const legacy=require('../src/legacy/tag-engine.cjs');
const out=fs.openSync('.cache/legacy-features.jsonl','w');
let count=0;
for(const folder of ['spm_dataset-1.1.1','spm_dataset-4k-0.2.0']) {
 const data=JSON.parse(fs.readFileSync(path.join('dataset',folder,'labels.json'),'utf8'));
 for(let i=0;i<data.length;i++) {
  const r=data[i],cm=legacy.buildCachedMap(fs.readFileSync(path.join('dataset',folder,r.file),'utf8'));
  const keys=Object.keys(cm.windows[0]).filter(k=>!['t','end','notes','rows'].includes(k));
  const vector=[];
  for(const stat of ['mean','sd','q90']) for(const k of keys) {
   const a=cm.windows.map(w=>w[k]),m=a.reduce((s,x)=>s+x,0)/a.length;
   vector.push(stat==='mean'?m:stat==='sd'?Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/a.length):a.sort((x,y)=>x-y)[Math.floor((a.length-1)*.9)]);
  }
  fs.writeSync(out,JSON.stringify({uid:folder+'/'+r.id+'/'+i,keys,vector})+'\n');
  if(++count%100===0) console.log('legacy features',count);
 }
}
fs.closeSync(out);
