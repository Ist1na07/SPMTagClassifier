'use strict';
const fs=require('node:fs'),path=require('node:path');
const api=require('../src/index.cjs');
const {family}=require('../src/segment.cjs');
const splits=require('./results/7k-split.json');
const data=JSON.parse(fs.readFileSync('dataset/spm_dataset-1.1.1/labels.json','utf8')),samples=[];
data.forEach((r,i)=>{
 const uid='spm_dataset-1.1.1/'+r.id+'/'+i;if(!splits[uid])return;
 const a=api.analyzeMap(fs.readFileSync(path.join('dataset/spm_dataset-1.1.1',r.file),'utf8'),{algorithm:'modern'});
 const ln=Object.entries(a.overall.weights).reduce((s,[tag,w])=>s+(family(tag)==='ln'?w:0),0);
 samples.push({uid,...splits[uid],truth:r.sort==='mix'?'hb':r.sort,ln});
});
const classes=['rc','hb','ln'];
function evaluate(rs,a,b){
 const per=classes.map(c=>{let tp=0,fp=0,fn=0;for(const r of rs){const p=r.ln<a?'rc':r.ln>b?'ln':'hb';if(p===c&&r.truth===c)tp++;else if(p===c)fp++;else if(r.truth===c)fn++;}return 2*tp/Math.max(1,2*tp+fp+fn);});
 return {macroF1:per.reduce((a,b)=>a+b)/3,perClass:Object.fromEntries(classes.map((c,i)=>[c,per[i]]))};
}
const grid=[],val=samples.filter(s=>s.split==='validation'),test=samples.filter(s=>s.split==='test');let best;
for(const rcMax of [.1,.18,.3,.4])for(const lnMin of [.5,.6,.68,.8,.9]){
 const v=evaluate(val,rcMax,lnMin);const r={rcMax,lnMin,validation:v};grid.push(r);if(!best||v.macroF1>best.validation.macroF1)best=r;
}
const p6=JSON.parse(fs.readFileSync('src/models/6k.json','utf8'));p6.sort={rcMax:best.rcMax,lnMin:best.lnMin};
fs.writeFileSync('src/models/6k.json',JSON.stringify(p6,null,2)+'\n');
fs.writeFileSync('research/results/sort-metrics.json',JSON.stringify({selected:best,test:evaluate(test,best.rcMax,best.lnMin),grid,samples},null,2));
console.log('sort thresholds',best);
