'use strict';
// Actual JS consumer used to cross-check the vectorized research evaluator.
const {CLASSES}=require('./native.cjs');
const {aggregateEvidence,coalesceForDisplay,supportMetrics}=require('./evidence.cjs');
function selectNativeSegments(value,config){
 const source=value.partitions[config.partition].cases[config.head];
 const rows=source[config.smoothing==='legacy'?'legacy':'raw'];
 const extraGate=config.extraGate??'all';
 if(!['all','ln_protected','off'].includes(extraGate))throw Error('Unknown extra relative-burden gate treatment');
 if(!config.burdenGate||extraGate==='off')return rows;
 const active=rows.filter(s=>s.burden>0).map(s=>[s.burden,s.end-s.start]).sort((a,b)=>a[0]-b[0]);
 const total=active.reduce((s,x)=>s+x[1],0);let time=0,reference=1;
 for(const [burden,duration]of active){time+=duration;reference=burden;if(time>=.75*total)break;}
 return rows.map(s=>s.family!=='break'&&!(extraGate==='ln_protected'&&s.family==='ln')&&s.burden/reference<config.burdenGate?
  {...s,tag:'break',family:'break',confidence:1,probability:CLASSES.map(k=>k==='break'?1:0)}:s);
}
function evaluateCachedNative(value,config){
 const rows=selectNativeSegments(value,config),options={...config,lnLow:config.familyThresholds[0],lnHigh:config.familyThresholds[1]};
 const overall=aggregateEvidence(rows,options);
 const segments=config.smoothing==='coherent'&&config.mode==='soft'?coalesceForDisplay(rows,options):rows.map(s=>({...s,metrics:supportMetrics(s.probability)}));
 return {segments,overall};
}
module.exports={selectNativeSegments,evaluateCachedNative};
