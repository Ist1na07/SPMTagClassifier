'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {infer,integrate,EXPECTED_RATING,ratingEvidenceFromSPM,validateRating}=require('../src/perception.cjs');
const CLASSES=['anchor','chordjack','coordination','dense chordstream','density','fast chordstream','inverse','ln wildcard','minijack','release','speed','tech','technical','vibro'];
function fixture(){
 const notes=Array.from({length:101},(_,i)=>({t:i*100,col:i%7,lnEnd:null}));
 const map={keys:7,notes,firstNote:0,lastNote:10000,timingPoints:[],breaks:[],meta:{}},segments=[{start:0,end:10000,tag:'speed',family:'rc',confidence:1,scores:{speed:1,tech:.8}}];
 const rating={schemaVersion:1,keyCount:7,speedRate:1,clock:'post-rate-ms',modelIdentity:EXPECTED_RATING,query:notes.map(n=>n.t),demand:notes.map(()=>2),components:Object.fromEntries(['Rbar','Pbar','Cbar','Jm'].map(k=>[k,notes.map(()=>1)]))};
 const profile={schemaVersion:1,classes:CLASSES,components:true,mixKind:'distinct',supportPower:1,recipe:{alpha:1,beta:1,width:1000,hold:1},
  fine:{coef:CLASSES.map(()=>[0,1,0,0,0,0,0,0,0]),scaler:{mean:CLASSES.map(()=>Array(8).fill(0)),scale:CLASSES.map(()=>Array(8).fill(1)),keep:CLASSES.map(()=>[true,false,false,false,false,false,false,false])}},
  slot:{coef:Array.from({length:3},()=>[-10,0,0,0,0,0,0,0,0]),scaler:{mean:Array(8).fill(0),scale:Array(8).fill(1),keep:Array(8).fill(true)}}};
 return {map,segments,rating,profile};
}
function close(a,b){assert(Math.abs(a-b)<1e-10,`${a} != ${b}`);}
test('overlapping percentages are neither probabilities nor a partition',()=>{
 const f=fixture(),r=infer(f.map,f.segments,f.rating,f.profile);
 close(r.overall.percentages.speed,99.9);close(r.overall.percentages.tech,80);
 assert.deepEqual(r.overall.tags,['speed','tech']);assert(Object.values(r.overall.percentages).reduce((a,b)=>a+b,0)>100);
 close(Object.values(r.overall.composition).reduce((a,b)=>a+b,0),1);
});
test('identical support may be partitioned without changing its integral',()=>{
 const f=fixture(),original=infer(f.map,f.segments,f.rating,f.profile);
 const split=[[0,1700],[1700,3000],[3000,10000]].map(([start,end])=>({...f.segments[0],start,end}));
 const r=infer(f.map,split,f.rating,f.profile);
 for(const c of CLASSES)close(r.overall.percentages[c],original.overall.percentages[c]);
 assert.deepEqual(r.overall.tags,original.overall.tags);
});
test('short difficult passage gains burden share, not just elapsed time',()=>{
 const f=fixture();f.segments=[{...f.segments[0],end:8000,scores:{speed:1}},{...f.segments[0],start:8000,tag:'tech',scores:{tech:1}}];
 f.rating.demand=f.rating.query.map(t=>t>=8000?20:1);const r=infer(f.map,f.segments,f.rating,f.profile);
 assert(r.overall.percentages.tech>40);assert(r.overall.percentages.speed<80);
});
test('Break keeps physical burden in the denominator; no artificial 100 percent',()=>{
 const f=fixture();f.segments=[{...f.segments[0],end:2000,scores:{speed:1}},{start:2000,end:10000,tag:'break',family:'break',scores:{break:1}}];
 const r=infer(f.map,f.segments,f.rating,f.profile);assert(r.overall.percentages.speed<25);assert(r.overall.perception.coveredBurdenShare>0.99);
});
test('headless held time is not assigned zero action burden',()=>{
 const f=fixture();f.map={...f.map,notes:[{t:0,col:0,lnEnd:8000},{t:9000,col:1,lnEnd:null}],firstNote:0,lastNote:9000};
 f.segments=[{start:0,end:9000,tag:'release',family:'ln',scores:{release:1}}];f.rating.query=[0,9000];f.rating.demand=[2,2];f.rating.components=Object.fromEntries(['Rbar','Pbar','Cbar','Jm'].map(k=>[k,[1,1]]));
 const r=integrate(f.map,f.segments,f.rating,f.profile.recipe,{components:true});
 const covered=r.atoms.filter(a=>a.start<7000&&a.end>2000);assert(covered.length>0);assert(covered.every(a=>a.weight>0));
});
test('wrong clocks, incomplete components and missing required curves fail explicitly',()=>{
 const f=fixture();assert.throws(()=>infer(f.map,f.segments,null,f.profile),/MissingRating/);
 assert.throws(()=>infer(f.map,f.segments,{...f.rating,speedRate:1.5},f.profile),/Clock/);
 assert.throws(()=>infer(f.map,f.segments,{...f.rating,query:f.rating.query.map(t=>t+1)},f.profile),/Clock/);
 assert.throws(()=>infer(f.map,f.segments,{...f.rating,components:{Rbar:f.rating.demand}},f.profile),/Channel/);
});
test('translation and mirror preserve the same physical evidence',()=>{
 const f=fixture(),a=infer(f.map,f.segments,f.rating,f.profile),offset=500000;
 const map={...f.map,firstNote:offset,lastNote:10000+offset,notes:f.map.notes.map(n=>({...n,t:n.t+offset,col:6-n.col}))};
 const segments=f.segments.map(s=>({...s,start:s.start+offset,end:s.end+offset})),rating={...f.rating,query:f.rating.query.map(t=>t+offset)};
 const b=infer(map,segments,rating,f.profile);for(const c of CLASSES)close(a.overall.percentages[c],b.overall.percentages[c]);
});
test('action-only profile needs no rating and does not read demand',()=>{
 const f=fixture();f.profile.components=false;f.profile.recipe.alpha=0;
 const a=infer(f.map,f.segments,null,f.profile),b=infer(f.map,f.segments,f.rating,f.profile);
 assert.deepEqual(a.overall.percentages,b.overall.percentages);assert.equal(a.overall.perception.difficultyAvailable,false);
});
test('consumer does not mutate shared maps, curves, frames or model',()=>{
 const f=fixture(),before=structuredClone(f);
 function freeze(o){Object.freeze(o);for(const v of Object.values(o))if(v&&typeof v==='object'&&!Object.isFrozen(v))freeze(v);}
 freeze(f);infer(f.map,f.segments,f.rating,f.profile);assert.deepEqual(f,before);
});
test('SPM public-result adapter copies existing curves, never invents components',()=>{
 const f=fixture(),input={scope:'whole',key_count:7,rate:1,model_identity:EXPECTED_RATING,query:f.rating.query,field:f.rating.demand};
 const r=ratingEvidenceFromSPM(input);assert.equal(r.components,undefined);assert.notEqual(r.query,input.query);assert.notEqual(r.demand,input.field);
 validateRating(f.map,r);assert.throws(()=>ratingEvidenceFromSPM(input,{Rbar:input.field}),/together/);
 assert.throws(()=>ratingEvidenceFromSPM({...input,scope:'context'}),/Scope/);
});
