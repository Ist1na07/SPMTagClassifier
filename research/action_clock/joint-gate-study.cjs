'use strict';
// Bounded exact native gate-grid replay; windows and identical decisions are
// reused per map. No new scorer, local head, or alternate aggregation model.
const fs=require('node:fs'),vm=require('node:vm'),zlib=require('node:zlib');
const api=require('../../src/index.cjs'),{compileOverlayEngine}=require('../../scripts/build-overlay.cjs'),{replaceOnce}=require('../rework/legacy-lab.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{musicalSegments}=require('../../src/musical-segmentation.cjs'),{refine}=require('../../src/segment-readout.cjs');
const perception=require('../../src/perception.cjs'),{createPressureGuard}=require('../../src/break-pressure.cjs');
const profiles=require('../../src/models/perception7k.json').profiles,breakProfile=require('../../src/models/break7k.json'),post=require('../../src/models/overlay7k.json').aggregation;
let source=compileOverlayEngine({segmentReadout:true});source=replaceOnce(source,'  return __toCommonJS(index_exports);','  Object.assign(index_exports,{study:{classifySegments,smoothSegments,aggregateOverall,medianWindowFeatures,computeMapMedians,equivBpmFor,model_generated_default,parseOsu}});\n  return __toCommonJS(index_exports);');
const box={module:{exports:{}}};vm.runInNewContext(source,box,{codeGeneration:{strings:false,wasm:false}});const engine=box.module.exports,L=engine.study;
const manifest=JSON.parse(fs.readFileSync('.cache/local-feedback-activity-r6b/manifest.json','utf8')),manual=new Set(manifest.manualUids);
const real=new Set();for(const rnd of ['r4','r5','r6'])for(const a of Object.values(require('../frontier/annotations-'+rnd+'.json').cases))real.add(a.sourceUid);for(const a of Object.values(require('../frontier/annotations-r10.json').originalLocalRevision.cases))real.add(a.sourceUid);
const repairFlag=process.argv.indexOf('--repair-file'),repairFile=repairFlag>=0?process.argv[repairFlag+1]:process.argv.includes('--repair-ranges')?'research/action_clock/results/range-repair-uids.json':null;
const extendContrast=process.argv.includes('--extend-contrast');
const ordering=extendContrast?JSON.parse(fs.readFileSync('research/action_clock/results/music-ordering-audit.json','utf8')):null;
const repair=extendContrast?{reason:'Complete contrast-order extension after all411 cut audit, reuse exactly identical native cuts.',byRadius:Object.fromEntries([.5,1,2].map(r=>[r===.5?'.5':String(r),ordering.differences.filter(d=>d.radius===r).map(d=>d.uid)]))}:repairFile?JSON.parse(fs.readFileSync(repairFile,'utf8')):null;
const original=repair?JSON.parse(zlib.gunzipSync(fs.readFileSync('research/action_clock/results/joint-grid-native.json.gz'))):null;
const resources=fs.readFileSync('.cache/perception-final-resource.jsonl','utf8').trim().split('\n').map(JSON.parse);
const configs=[null,.5,1,2].flatMap(radius=>[.35,.5,.65].flatMap(confidenceFloor=>[.75,1,1.25].map(lowNpsScale=>({radius,confidenceFloor,lowNpsScale}))));
const protocol={scope:'Retrospective 7K development study; historically exposed W/b and labels. No new blind claim.',configs,consumers:['actions','detail','compact'],
 amendment:'Fixed-gate full music study showed local gains and some whole-macro-F1 costs. Before this joint replay, expand every radius INCLUDING off to independent 3x3 confidenceFloor/lowNpsScale tuning. No selective protection of old parameters.',
 selection:'Same multi-objective as previous Break study: 0.30 six-chart per-chart local accuracy +0.15 corrected-denominator Break F1 +0.15 real8 local accuracy +0.40 full349 macro-F1. Select each consumer gate pair within each radius, then one shared radius maximizing mean consumer score. Report full-data and source-scoped selection separately.',
 cached:'Native windows once/map/radius; same tag/secondary decision masks reuse identical readout/integral, never reuse predictions across different masks/consumers/radii.',
 fixed:'Type W/b, type routing, rule attributes, difficulty quantile/fraction, physical integrand, all integral thresholds and original reference segmentation.'};
if(!repair)fs.writeFileSync('research/action_clock/results/joint-protocol.json',JSON.stringify(protocol,null,2));
else {const snapshot='research/action_clock/results/joint-grid-native.'+(extendContrast?'before-contrast':repairFlag>=0?'before-quantization':'r0')+'.json.gz';if(!fs.existsSync(snapshot))fs.copyFileSync('research/action_clock/results/joint-grid-native.json.gz',snapshot);}
const outputs=original?.outputs??configs.map((config,id)=>({id,config,consumers:Object.fromEntries(protocol.consumers.map(n=>[n,[]]))}));
if(extendContrast){if(outputs.some(o=>o.config.mode==='contrast'))throw Error('Contrast extension already present');for(const o of [...outputs])if(o.config.radius!==null)outputs.push({...structuredClone(o),id:outputs.length,config:{...o.config,mode:'contrast'}});}
let checked=0,hits=0,computed=0;
for(let ri=0;ri<resources.length;ri++){
 const r=resources[ri],text=fs.readFileSync(r.source,'utf8'),ctx=prepare(parseOsu(text)),legacyMap=L.parseOsu(text);
 const guards=Object.fromEntries(['detail','compact'].map(n=>[n,createPressureGuard(r.ratings[n],breakProfile.profiles[n],breakProfile.gapMs)]));
 for(const radius of [null,.5,1,2]){
  if(repair&&!(repair.byRadius[radius===.5?'.5':String(radius)]??[]).includes(r.uid))continue;
  const callback=radius===null?undefined:i=>musicalSegments(ctx,i,{maxMoveBeats:radius,mode:extendContrast?'contrast':'nearest'}),cm=engine.buildCachedMap(text,{_musicalSegmenter:callback});
  const reference=L.smoothSegments(L.classifySegments(cm.referenceCm??cm,{scorer:'model',_expertModel:L.model_generated_default}));
  const prior=L.aggregateOverall(reference,post),mapMedians=L.computeMapMedians(cm.windows),median=new Map();
  for(const name of protocol.consumers){const p=profiles[name],old=breakProfile.profiles[name],memo=new Map();
   for(const output of outputs.filter(o=>o.config.radius===radius&&(o.config.mode??'nearest')===(extendContrast?'contrast':'nearest'))){
    const c=output.config,policy={...old,confidenceFloor:c.confidenceFloor,lowNpsScale:c.lowNpsScale,protects:(start,end,stage)=>!!guards[name]&&old.target.split('+').includes(stage)&&guards[name].measure(start,end).protected};
    const raw=L.classifySegments(cm,{scorer:'model',_breakPolicy:policy}),key=raw.map(s=>s.tag+'|'+s.secondary).join(';');let result=memo.get(key);
    if(result)hits++;else{
     computed++;const readout=refine(ctx,raw,p.classes,{supportPower:p.supportPower??1,resolveBpm:(tag,parent)=>{if(!median.has(parent)){const s=cm.rawSegs[parent];median.set(parent,L.medianWindowFeatures(cm.windows,...s.windowRange,mapMedians));}return L.equivBpmFor(tag,median.get(parent),cm.extras[parent]);}});
     const inferred=perception.infer(ctx.map,readout.frames,name==='actions'?undefined:r.ratings[name],p,{context:ctx,baseline:prior,_branchSupport:readout.support});
     const segments=readout.segments.map(({start,end,tag,secondary})=>({start,end,tag,secondary}));if(segments.length){segments[0].start=legacyMap.firstNote;segments.at(-1).end=legacyMap.lastNote;}
     result={tags:inferred.overall.tags,percentages:inferred.overall.percentages,segments};memo.set(key,result);
    }
    const entry={uid:r.uid,tags:result.tags,...(manual.has(r.uid)||real.has(r.uid)?{segments:result.segments}:{})};
    if(repair){const at=output.consumers[name].findIndex(x=>x.uid===r.uid);if(at<0)throw Error('Missing repair input');output.consumers[name][at]=entry;}else output.consumers[name].push(entry);
    if(c.confidenceFloor===.5&&c.lowNpsScale===1&&radius===1&&!extendContrast&&(ri<3||manual.has(r.uid))){
     const actual=api.analyzeMap(text,name==='actions'?{}:{ratingEvidence:r.ratings[name]});
     if(JSON.stringify(actual.overall.tags)!==JSON.stringify(result.tags))throw Error('Native tags mismatch '+r.uid);
     for(const [k,v]of Object.entries(result.percentages))if(Math.abs(actual.overall.percentages[k]-v)>1e-8)throw Error('Native percentage mismatch '+r.uid+'/'+k);
     if(JSON.stringify(actual.segments.map(({start,end,tag,secondary})=>({start,end,tag,secondary})))!==JSON.stringify(result.segments))throw Error('Native local mismatch '+r.uid);
     checked++;
    }
   }
  }
 }
 if((ri+1)%50===0)console.log('joint',ri+1,'/',resources.length,'unique native decisions',computed,'reused',hits,'public checks',checked);
}
fs.writeFileSync('research/action_clock/results/joint-grid-native.json.gz',zlib.gzipSync(JSON.stringify({outputs,...(extendContrast?{orderingExtension:{comparedNativeCuts:ordering.comparisons,differingParentMapRadiusPairs:ordering.differences.length,completeConsumerConfigurations:outputs.length*3}}:original?.orderingExtension?{orderingExtension:original.orderingExtension}:{}),checked:checked+(original?.checked??0),computed:computed+(original?.computed??0),hits:hits+(original?.hits??0),...(original?.rangeRepair?{rangeRepair:original.rangeRepair}:{}),...(repair?{executionRepairs:[...(original?.executionRepairs??[]),{source:extendContrast?'research/action_clock/results/music-ordering-audit.json':repairFile,reason:repair.reason,inputsByRadius:Object.fromEntries(Object.entries(repair.byRadius).map(([k,v])=>[k,v.length])),computed,hits,checked}]}:{})})));
if(extendContrast){const file='research/action_clock/results/joint-protocol.json',p=JSON.parse(fs.readFileSync(file,'utf8'));p.orderingExtension={scope:'Added after full411 native cut audit; initial108 protocol retained.',comparedNativeCuts:ordering.comparisons,differingParentMapRadiusPairs:ordering.differences.length,completeConsumerConfigurations:outputs.length*3,selection:'Share radius AND ordering across consumers; independently select gate pairs; prefer nearest on ties.'};fs.writeFileSync(file,JSON.stringify(p,null,2));}
console.log('complete joint',outputs.length*3,'consumer configurations',checked,'public parity checks',computed,'unique',hits,'reused');
