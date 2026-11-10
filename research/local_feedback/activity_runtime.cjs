'use strict';
// Research native consumer: recompute actual physical features, not cache them
// behind the test. No rating call or annotation boundary enters this path.
const {prepare}=require('../../src/parser.cjs');
const {features}=require('../../src/features.cjs');
const {evidence}=require('./release_obligations.cjs');
const {routing}=require('./runtime.cjs');
const GEOMETRY=['npsPerKey','rowRate','chordFraction','multiFraction','lnHeadFraction','overlap','repeatFraction','cutReturn','chordChange','columnEntropy','handImbalance','sameHand','crossCut','logIoiSd','ioiChange','graceFraction','gridResidual','ioiCv','occupancy','minOccupancy','occupancySd','heldAtHead','lockFraction','releaseIndependence','releaseScatter','tailRate','shortHoldFraction','inverseGap','lnHeadOverlap','anchorShare','repeatRun','maskEntropy','trillFraction','rollFraction','burstiness','activeFraction'];
const GUIDES=['continuousReleasePairRate','independentReleasePairRate','independentTailRate','heldReleaseObligation','independentHeldObligation'];
const RATE=new Set(['npsPerKey','rowRate','tailRate',...GUIDES]);
function measure(map,source,model,context){
  const ctx=context??prepare(map),release=model.config.obligations!==false?evidence(ctx):null,frames=[],inputs=[],parents=[];
  for(let parent=0;parent<source.length;parent++){
    const old=source[parent];
    for(let start=old.start;start<old.end;start+=model.stepMs){
      const end=Math.min(start+model.stepMs,old.end),mid=(start+end)/2,a=Math.max(map.firstNote,mid-model.windowMs/2),b=Math.max(a+1,Math.min(map.lastNote,mid+model.windowMs/2));
      const m=features(ctx,a,b).metrics,x=Object.fromEntries(GEOMETRY.map(k=>[k,m[k]]));
      if(release)release(a,b).forEach((v,j)=>x[GUIDES[j]]=v);
      for(const k of RATE)if(k in x)x[k]=Math.log1p(x[k]);
      ['break','rc','ln'].forEach((family,j)=>x[['wasBreak','wasRC','wasLN'][j]]=Number(old.family===family));
      inputs.push(x);frames.push({...old,start,end,parent});parents.push(parent);
    }
  }
  if(model.config.context){
    const keys=['npsPerKey','rowRate','tailRate','occupancy'],means=[0,0,0,0];let duration=0;
    frames.forEach((f,i)=>{const dt=f.end-f.start;duration+=dt;keys.forEach((k,j)=>means[j]+=dt*inputs[i][k]);});
    const names=['relativeLogNps','relativeLogRowRate','relativeLogTailRate','relativeOccupancy'];
    inputs.forEach(x=>keys.forEach((k,j)=>x[names[j]]=x[k]-means[j]/duration));
  }
  for(const x of inputs)for(const key of model.features)if(!Number.isFinite(x[key]))throw Error('NonfiniteActivityFeature: '+key);
  return {frames,inputs,parents};
}
function apply(model,map,source,candidates,classes,context){
  if(model.kind!=='local-activity-routing'||model.stepMs!==500||!Number.isFinite(model.windowMs))throw Error('Unsupported activity model');
  if(source.length!==candidates.length)throw Error('Activity expert inventory mismatch');
  const m=measure(map,source,model,context),result=routing(m.frames,m.inputs,model,m.parents.map(p=>candidates[p]),classes);
  return {...result,inputs:m.inputs,parents:m.parents};
}
module.exports={measure,apply};
