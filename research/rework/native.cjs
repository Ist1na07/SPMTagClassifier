'use strict';
// Research consumer of the complete native segmenter + expert/gating pipeline.
// This is not another flat classifier and does not alter the archived engine.
const {loadLab,context,inspectPrepared}=require('./legacy-lab.cjs');
const patches=require('./patches.cjs');
const CLASSES=['chordjack','minijack','dense chordstream','fast chordstream','speed','tech','vibro','anchor',
 'coordination','density','inverse','release','technical','wildcard','ln wildcard','break'];
const BOUNDARIES=[1500,3000,6000].flatMap(minSegLenMs=>[.025,.05,.1].map(segProminenceAbs=>({minSegLenMs,segProminenceAbs})));
const HEADS=[.35,.45,.55].flatMap(lnDominantRatio=>['model','blend'].map(scorer=>({lnDominantRatio,scorer})));
const logical=patches.logical,neutral=logical.concat(patches.neutralHands,patches.coverage);
const TREATMENTS={original:[],area:[patches.area],adjacency:[patches.adjacency],heldOrder:[patches.heldOrder],
 phase:[patches.phase],sustaining:[patches.sustaining],neutralHands:[patches.neutralHands],coverage:[patches.coverage],
 transformedScale:[patches.transformedScale],logical,logical_neutral:neutral};
for(const ms of [20,40,80]){
 TREATMENTS['release_'+ms]=[patches.releaseAt(ms)];
 TREATMENTS['logical_release_'+ms]=neutral.concat(patches.releaseAt(ms));
 TREATMENTS['full_'+ms]=neutral.concat(patches.releaseAt(ms),patches.transformedScale);
}
const SPEC={id:'native-label-free-r1',classes:CLASSES,boundaries:BOUNDARIES,heads:HEADS,
 fixed:{winMs:1000,winStepMs:250,segProminenceRel:.35},
 scoreSemantics:'normalized within-family decision support, not calibrated correctness; rejected wildcard mass is conditioned away',
 treatmentPatches:Object.fromEntries(Object.entries(TREATMENTS).map(([k,ps])=>[k,ps.map(p=>p.id)]))};

function decisionSupport(segment){
 if(segment.family==='break')return CLASSES.map(k=>k==='break'?1:0);
 const values=CLASSES.map(k=>Math.max(0,segment.scores?.[k]||0));
 let top=values.indexOf(Math.max(...values));
 // The native decision rejects a leading wildcard if a concrete class clears
 // its demotion threshold. Its support is conditional on that explicit gate.
 if(['wildcard','ln wildcard'].includes(CLASSES[top])&&!['wildcard','ln wildcard'].includes(segment.tag)){
  values[CLASSES.indexOf('wildcard')]=0;values[CLASSES.indexOf('ln wildcard')]=0;
 }
 const sum=values.reduce((a,b)=>a+b,0);
 if(!(sum>0))throw Error('Native decision has no finite class support');
 const p=values.map(v=>v/sum);top=p.indexOf(Math.max(...p));
 if(CLASSES[top]!==segment.tag)throw Error('Decision/support mismatch: '+segment.tag+' versus '+CLASSES[top]);
 return p;
}
function compact(segment,raw,burdens){
 let sum=0,duration=0;
 raw.forEach((s,i)=>{const d=Math.max(0,Math.min(s.end,segment.end)-Math.max(s.start,segment.start));sum+=d*burdens[i];duration+=d;});
 const probability=decisionSupport(segment);
 return {start:segment.start,end:segment.end,tag:segment.tag,family:segment.family,confidence:segment.confidence,
  probability,burden:duration?sum/duration:0};
}
function extractNative(api,prepared){
 const partitions=[];
 for(const boundary of BOUNDARIES){
  Object.assign(api.lab.CONFIG,boundary);
  const view=inspectPrepared(api,prepared),burdens=view.segments.map(s=>s.f.nps*(.5+s.f.chordMean/3)*(.6+s.f.jackSpeedBpm/150));
  const cases=HEADS.map(head=>{
   const classified=api.lab.classifySegments(view.cache,head);
   const smoothed=api.lab.smoothSegments(structuredClone(classified));
   return {head,raw:classified.map(s=>compact(s,classified,burdens)),legacy:smoothed.map(s=>compact(s,classified,burdens))};
  });
  partitions.push({boundary,cases});
 }
 return partitions;
}
function nativeApi(treatment){
 if(!Object.hasOwn(TREATMENTS,treatment))throw Error('Unknown native treatment '+treatment);
 return loadLab({patches:TREATMENTS[treatment]});
}
module.exports={CLASSES,BOUNDARIES,HEADS,TREATMENTS,SPEC,decisionSupport,extractNative,nativeApi};
