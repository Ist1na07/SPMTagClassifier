'use strict';
const {composition}=require('./composition.cjs');
const {CLASSES}=require('./native.cjs');
const NON_AGGR=new Set(['break','wildcard','ln wildcard','rc mix','ln mix','hybrid']);
const LN=new Set(['coordination','density','inverse','release','technical','ln wildcard']);
const BREAK=CLASSES.indexOf('break');
const defaults={mode:'soft',effortPower:0,confidencePower:0,dominantShare:.52,mixShare:.07,
 secondRel:.75,twoTagMaxShare:.55,lnLow:.18,lnHigh:.68,minEvidence:.25,minimumDiversity:0};
const settings=options=>({...defaults,...options,
 excludeVibroFromMix:options.excludeVibroFromMix??(options.mode!=='legacy')});
const atoms=segments=>segments.flatMap(s=>s.components?atoms(s.components):[s]);
function supportMetrics(p){
 const ranked=[...p].sort((a,b)=>b-a);
 return {maximum:ranked[0],margin:ranked[0]-ranked[1],concentration:p.reduce((s,x)=>s+x*x,0),
  entropy:-p.reduce((s,x)=>s+(x?x*Math.log(x):0),0)/Math.log(p.length)};
}
function evidence(segments,options={}){
 const p=settings(options),rows=atoms(segments);
 const active=rows.filter(s=>s.burden>0).map(s=>[s.burden,Math.max(0,s.end-s.start)]).sort((a,b)=>a[0]-b[0]);
 const mass=active.reduce((s,x)=>s+x[1],0);let acc=0,reference=1;
 for(const [value,time]of active){acc+=time;reference=value;if(acc>=.75*mass)break;}
 return rows.map(s=>{
  const duration=Math.max(0,s.end-s.start),effort=Math.max(1e-4,(s.burden||0)/reference)**p.effortPower;
  const metrics=s.probability?supportMetrics(s.probability):null;
  const concentration=p.mode==='soft'?metrics.concentration:Math.max(.4,s.confidence);
  const base=(p.mode==='soft'?duration/1000:Math.max(.5,duration/1000))*effort;
  // The native Break decision is an eligibility gate, not a source of tiny
  // residual fine-class votes that a very long rest can accumulate.
  return {...s,metrics,evidenceBase:base,evidenceWeight:p.mode==='soft'&&s.tag==='break'?0:base*concentration**p.confidencePower};
 });
}
function aggregateEvidence(segments,options={}){
 const p=settings(options),rows=evidence(segments,p),weights={};
 if(!rows.length)return {tags:[],sort:'rc',weights,shares:{},diversity:0,distribution:CLASSES.map(()=>0),activeWeight:0};
 let total=0,lnW=0,rcW=0,diversity=0,distribution=null;
 const familyTags={rc:new Set(),ln:new Set()};
 if(p.mode==='soft'){
  const summary=composition(rows.map(s=>s.probability),rows.map(s=>s.evidenceWeight),{breakIndex:BREAK});
  diversity=summary.diversity;distribution=CLASSES.map(()=>0);total=summary.activeWeight;
  summary.indices.forEach((k,j)=>distribution[k]=summary.distribution[j]);
  for(let k=0;k<CLASSES.length;k++){
   const tag=CLASSES[k],mass=total*distribution[k];
   if(LN.has(tag))lnW+=mass;else if(tag!=='break')rcW+=mass;
   if(!NON_AGGR.has(tag)&&mass>0)weights[tag]=mass;
  }
 }else{
  for(const s of rows){
   if(s.family==='break')continue;
   if(s.family==='ln')lnW+=s.evidenceBase;else if(s.family==='rc')rcW+=s.evidenceBase;
   if(NON_AGGR.has(s.tag))continue;
   if(!p.excludeVibroFromMix||s.tag!=='vibro')familyTags[s.family].add(s.tag);
   weights[s.tag]=(weights[s.tag]||0)+s.evidenceWeight;
   total+=p.mode==='legacy'?s.evidenceBase:s.evidenceWeight;
  }
 }
 const lnRatio=lnW/(p.mode==='soft'?(lnW+rcW||1):Math.max(1,lnW+rcW));
 const sort=lnRatio<p.lnLow?'rc':lnRatio>p.lnHigh?'ln':'hb';
 if(!(total>0))return {tags:[],sort:p.mode==='legacy'?'rc':sort,weights,shares:{},diversity,distribution,activeWeight:0};
 const ranked=Object.entries(weights).map(([tag,w])=>[tag,w/total]).sort((a,b)=>b[1]-a[1]);
 const top=ranked[0]?.[1]||0,significant=ranked.filter(([tag,share])=>share>=p.mixShare&&(!p.excludeVibroFromMix||tag!=='vibro'));
 let tags=[];
 const enoughFamilies=p.mode==='soft'?significant.length>=3:familyTags.rc.size+familyTags.ln.size>=3;
 if(p.mode==='soft'&&top<p.minEvidence)tags=[];
 else if(significant.length>=3&&enoughFamilies&&top<p.dominantShare){
  if(p.mode!=='soft'||diversity>=p.minimumDiversity)tags=[sort==='rc'?'rc mix':sort==='ln'?'ln mix':'hybrid'];
 }else if(ranked.length>=2&&ranked[1][1]>=p.secondRel*top&&top<p.twoTagMaxShare)tags=[ranked[0][0],ranked[1][0]];
 else if(ranked.length)tags=[ranked[0][0]];
 return {tags,sort,weights,shares:Object.fromEntries(ranked),diversity,distribution,activeWeight:total};
}
function coalesceForDisplay(segments,options={}){
 const rows=evidence(segments,{...options,mode:'soft'}),groups=[];
 for(const row of rows){
  const last=groups.at(-1);
  if(last&&last.end===row.start&&last.tag===row.tag&&last.family===row.family){last.end=row.end;last.components.push(row);}
  else groups.push({start:row.start,end:row.end,tag:row.tag,family:row.family,components:[row]});
 }
 return groups.map(g=>{
  const weight=g.components.reduce((s,r)=>s+r.evidenceWeight,0);
  const displayWeights=g.components.map(r=>weight?r.evidenceWeight:r.end-r.start),displayTotal=displayWeights.reduce((a,b)=>a+b,0);
  const probability=CLASSES.map((_,k)=>g.components.reduce((s,r,i)=>s+displayWeights[i]*r.probability[k],0)/displayTotal);
  return {...g,probability,metrics:supportMetrics(probability),aggregationWeight:weight};
 });
}
module.exports={aggregateEvidence,evidence,supportMetrics,coalesceForDisplay,defaults};
