'use strict';
const fs=require('node:fs'),path=require('node:path');
const {ROOT,records7}=require('./data.cjs'),{parseOsu,prepare}=require('../../src/parser.cjs');
const {KEYS,pooled}=require('./anchor.cjs');
const dir=path.join(ROOT,'dataset/spm_dataset-4k-0.2.0'),labels=JSON.parse(fs.readFileSync(path.join(dir,'labels.json'),'utf8').replace(/^\uFEFF/,''));
const output=[];const identitySeen=new Map();
function save(r,file){
 const map=parseOsu(fs.readFileSync(file,'utf8')),ctx=prepare(map);
 const iv=ctx.rows.slice(1).map((r,i)=>r.t-ctx.rows[i].t).filter(x=>x>0).sort((a,b)=>a-b),scale=iv[Math.floor(iv.length/2)]||1;
 const signature=map.notes.map(n=>[Math.round((n.t-map.firstNote)/scale*10),n.col,n.lnEnd===null?'':Math.round((n.lnEnd-n.t)/scale*10)].join(',')).join(';');
 const duplicateOf=identitySeen.get(map.keys+'|'+signature)||null;if(!duplicateOf)identitySeen.set(map.keys+'|'+signature,r.uid);
 output.push({...r,keys:map.keys,meta:map.meta,duplicateOf,features:Object.fromEntries([2000,4000,8000].map(ms=>[ms,pooled(ctx,ms)]))});
}
labels.forEach((r,i)=>{
 if(r.subset!=='jack'||r.tags.includes('Unsigned'))return;
 save({...r,uid:'4k/'+r.id+'/'+i,dataset:'4k',target:r.tags.includes('anchor chordjack')?1:0},path.join(dir,r.file));
});
// 7K samples are a transfer diagnostic. Their missing Anchor tags do not prove
// absence. Keep them out of the source-domain training and model selection.
for(const r of records7().filter(r=>r.dataset!=='manual'&&r.tags.some(t=>['chordjack','minijack','anchor'].includes(t))))save(r,r.path);
fs.writeFileSync('.cache/rework-anchor.json',JSON.stringify({features:KEYS.flatMap(k=>['mean','sd','q90'].map(s=>k+':'+s)),records:output})+'\n');
console.log('anchor records',output.length,'4k',output.filter(r=>r.dataset==='4k').length);
