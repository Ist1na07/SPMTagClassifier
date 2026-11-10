'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const {infer}=require('./runtime.cjs');
const f=JSON.parse(fs.readFileSync('.cache/perception-runtime-resource.json','utf8'));
let maximum=0,values=0;
function compare(a,b,path){
 if(Array.isArray(a)){assert.equal(a.length,b.length,path);a.forEach((v,i)=>compare(v,b[i],path+'/'+i));return;}
 assert(Number.isFinite(a)&&Number.isFinite(b),path);const error=Math.abs(a-b);maximum=Math.max(maximum,error);values++;
 assert(error<=1e-9,`${path}: ${a} != ${b}, error ${error}`);
}
for(const c of f.cases){
 const r=infer(c.map,c.segments,c.rating,f.profile,{geometry:c.geometry});
 for(const k of ['mass','x','support','share','fine','slots'])compare(r[k],c.expected[k],c.uid+'/'+k);
 const own=infer(c.map,c.segments,c.rating,f.profile);
 compare(own.x,r.x,c.uid+'/native geometry');compare(own.overall.percentages&&Object.values(own.overall.percentages),Object.values(r.overall.percentages),c.uid+'/percentages');
}
const receipt={cases:f.cases.length,comparedValues:values,maxAbsoluteError:maximum,profile:'fixed nonzero resource residual, not a performance winner',scope:'real full-map curves, natural raw frames, direct consumer geometry; no scorer invocation'};
fs.mkdirSync('research/perception/results/runtime-resource-r1',{recursive:true});fs.writeFileSync('research/perception/results/runtime-resource-r1/parity.json',JSON.stringify(receipt,null,2)+'\n');console.log(receipt);
