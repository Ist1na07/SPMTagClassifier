'use strict';
const fs=require('node:fs'),path=require('node:path');
const {createProbe}=require('./native_refit_probe.cjs'),product=require('../../src/index.cjs');
const file=process.argv[2];if(!file)throw Error('Provide full-data refit JSON');
const saved=JSON.parse(fs.readFileSync(file,'utf8')),analyze=createProbe(saved.models).analyzeMap;
function coverage(frames,a,b){const out={};for(const s of frames){const dt=Math.max(0,Math.min(b,s.end)-Math.max(a,s.start));if(dt)out[s.tag]=(out[s.tag]??0)+dt/(b-a);}return out;}
const sources=[...['A01','A02','B01','B02'].map(id=>({round:1,id})),...['C16','C24','C32','C48','CTX-L','CTX-H'].map(id=>({round:2,id})),...['D000','D008','D024','D072','D125','D216','LN-L','LN-H'].map(id=>({round:3,id})),...['H01','H02','H03','H04'].map(id=>({round:7,id}))],out=[];
for(const item of sources){
 const input=path.join('artifacts','frontier-annotation-r'+item.round,item.id+'.osu'),text=fs.readFileSync(input,'utf8'),old=product.analyzeMap(text),r=analyze(text),target=/^(H|CTX|LN-)/.test(item.id)?[9000,17000]:[r.map.firstNote,r.map.lastNote];
 const row={...item,file:input,target,baseline:coverage(old.segments,...target),candidate:coverage(r.segments,...target),overallTags:r.overall.tags,percentages:r.overall.percentages,
  scope:item.id==='D216'?'No user label; descriptive only':'Existing actively annotated mechanism, not randomly sampled test accuracy'};
 out.push(row);console.log(item.id,row.baseline,'=>',row.candidate,'whole',row.overallTags);
}
const output=file.replace(/\.json$/,'-mechanisms.json');fs.writeFileSync(output,JSON.stringify({candidate:file,cases:out},null,2)+'\n');
