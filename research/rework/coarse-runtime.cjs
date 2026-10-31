'use strict';
// Actual JavaScript consumer of the jointly trained native-key coarse model.
// It has no separate whole-map feature head and performs no 4K/7K output mix.
const {parseOsu,prepare}=require('../../src/parser.cjs');
const physical=require('./physical.cjs'),{composition}=require('./composition.cjs');
const {physicalSupport}=require('./physical-support.cjs');
const named=(classes,p)=>Object.fromEntries(classes.map((c,i)=>[c,p[i]]));
const entropy=p=>-p.reduce((sum,v)=>sum+(v>0?v*Math.log(v):0),0)/Math.log(5);

function localScores(metrics,model){
 const D=model.keys.length,K=model.classes.length,z=[],theta=model.theta;let clipped=0,maxMagnitude=0;
 const x=model.keys.map((key,j)=>{
  if(!Object.hasOwn(metrics,key))throw Error('Unsupported coarse feature: '+key);
  const value=(metrics[key]-model.mean[j])/model.scale[j];maxMagnitude=Math.max(maxMagnitude,Math.abs(value));
  if(Math.abs(value)>model.clip)clipped++;
  return Math.max(-model.clip,Math.min(model.clip,value));
 });
 for(let k=0;k<K;k++){
  let value=theta[K*D+k];for(let j=0;j<D;j++)value+=theta[k*D+j]*x[j];z.push(value);
 }
 const max=Math.max(...z),p=z.map(v=>Math.exp(v-max)),sum=p.reduce((a,b)=>a+b,0);
 return {p:p.map(v=>v/sum),clippedFeatureFraction:clipped/D,maxStandardizedMagnitude:maxMagnitude};
}

function relativeEffort(windows){
 const positive=windows.filter(w=>w.metrics.effort>0).map(w=>[w.metrics.effort,w.end-w.start]).sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
 const total=positive.reduce((s,[v,d])=>s+d,0);let cumulative=0,reference=1;
 for(const [value,duration]of positive){cumulative+=duration;reference=value;if(cumulative>=.75*total)break;}
 return reference;
}

function coalesce(windows,classes){
 const groups=[];
 for(const w of windows){
  let group=groups.at(-1);
  if(!group||group.tag!==w.tag||group.end!==w.start)groups.push(group={start:w.start,end:w.end,tag:w.tag,members:[]});
  group.end=w.end;group.members.push(w);
 }
 return groups.map(group=>{
  const {members}=group,total=members.reduce((s,w)=>s+w.aggregationWeight,0),duration=group.end-group.start;
  const p=Array(classes.length).fill(0);let entropyMass=0,effort=0,occupancy=0,notes=0,clipped=0,bpm=0,bpmDuration=0,structuralWindows=0;
  for(const w of members){
   const d=w.end-w.start,weight=total?w.aggregationWeight:d;
   w.probability.forEach((v,k)=>p[k]+=weight*v);entropyMass+=w.atomicEntropyMass;
   structuralWindows+=Number(!!w.structuralSupport);
   effort+=d*w.metrics.effort;occupancy+=d*w.metrics.occupancy;notes+=w.notes;clipped+=d*w.clippedFeatureFraction;
   if(w.equivBpm&&['jack','chordstream','speed'].includes(group.tag)){bpm+=d*w.equivBpm;bpmDuration+=d;}
  }
  for(let k=0;k<p.length;k++)p[k]/=total||duration;
  const order=p.map((v,i)=>[v,i]).sort((a,b)=>b[0]-a[0]||a[1]-b[1]);
  return {start:group.start,end:group.end,tag:group.tag,family:group.tag==='break'?'break':group.tag==='ln'?'ln':'rc',
   confidence:p[classes.indexOf(group.tag)],probability:p,distribution:named(classes,p),aggregationWeight:total,
   equivBpm:bpmDuration?Math.round(bpm/bpmDuration):undefined,
   evidence:{normalizedEntropy:entropy(p),margin:order[0][0]-order[1][0],atomicEntropyMass:entropyMass,
    meanLocalEntropy:total?entropyMass/total:0,effort:effort/duration,occupancy:occupancy/duration,
    featureWindows:members.length,contextNoteObservations:notes,clippedFeatureFraction:clipped/duration,structuralWindows}};
 });
}

function aggregateSegments(segments,classes){
 const result=composition(segments.map(s=>s.probability),segments.map(s=>s.aggregationWeight),{breakIndex:classes.length-1});
 // The merged support alone loses within-display-window uncertainty. Its
 // sufficient entropy statistic keeps between-window diversity unchanged.
 const mean=result.activeWeight?segments.reduce((s,r)=>s+r.evidence.atomicEntropyMass,0)/result.activeWeight:0;
 return {...result,meanLocalEntropy:mean,diversity:Math.max(0,result.entropy-mean)};
}

function analyzeWindows(windows,model,keyCount,options={}){
 const K=model.classes.length,D=model.keys.length;
 if(K!==5||model.theta.length!==K*(D+1)+2)throw Error('Invalid coarse5 model shape');
 if(!windows.length)throw Error('Empty coarse interval sequence');
 const classes=[...model.classes,'break'],reference=relativeEffort(windows),a=model.theta.at(-2),b=model.theta.at(-1);
 const annotated=windows.map(w=>{
  const score=localScores(w.metrics,model),effort=Math.max(1e-4,w.metrics.effort/reference),gate=model.config.burdenGate;
  const active=!w.structurallyEmpty&&(!gate||w.metrics.effort/reference>=gate||w.metrics.occupancy>0||w.metrics['release:tailRatePerKey']>0);
  const structuralSupport=active?physicalSupport(w,options.physicalSupport??'none'):null;
  const conditional=structuralSupport?model.classes.map(c=>Number(c===structuralSupport.tag)):score.p;
  const p=active?[...conditional,0]:classes.map(c=>Number(c==='break'));
  const weight=active?(w.end-w.start)/1000*effort**a*conditional.reduce((s,v)=>s+v*v,0)**b:0;
  const tag=classes[p.indexOf(Math.max(...p))];
  return {...w,probability:p,tag,aggregationWeight:weight,atomicEntropyMass:weight*entropy(p),structuralSupport,
   clippedFeatureFraction:score.clippedFeatureFraction,maxStandardizedMagnitude:score.maxStandardizedMagnitude};
 });
 const raw=composition(annotated.map(w=>w.probability),annotated.map(w=>w.aggregationWeight),{breakIndex:5});
 const segments=coalesce(annotated,classes),dominant=raw.activeWeight>0?model.classes[raw.distribution.indexOf(Math.max(...raw.distribution))]:null;
 return {windows:annotated,segments,coarse:{dominant,composition:named(model.classes,raw.distribution),
   familyMass:{ln:raw.distribution[model.classes.indexOf('ln')],rc:raw.distribution.reduce((s,p,i)=>s+(model.classes[i]==='ln'?0:p),0)},
   evidence:{activeWeight:raw.activeWeight,normalizedEntropy:raw.entropy,meanLocalEntropy:raw.meanLocalEntropy,diversity:raw.diversity},
   scoreKind:'weighted local morphology support; trained from weak whole-map labels and available interval labels; structural template support is not empirical correctness'},
  modelInfo:{taxonomy:'coarse5',nativeKeyCount:keyCount,featureCount:D,windowMs:model.config.windowMs,stepMs:2000,
   effortPower:a,concentrationPower:b,representation:model.spec.representation,weak4Only:!!model.spec.weak4Only,physicalSupport:options.physicalSupport??'none',
   clippedFeatureFraction:annotated.reduce((s,w)=>s+(w.end-w.start)*w.clippedFeatureFraction,0)/(annotated.at(-1).end-annotated[0].start)}};
}

function classifyCoarse(text,profile,options={}){
 const model=profile.model||profile,map=parseOsu(text,options),ctx=physical.prepare(prepare(map,10));
 const windows=physical.windows(ctx,model.config.windowMs,2000);
 return {map,...analyzeWindows(windows,model,map.keys,{physicalSupport:options.physicalSupport??profile.postprocess?.physicalSupport??'none'})};
}
module.exports={classifyCoarse,analyzeWindows,aggregateSegments,localScores};
