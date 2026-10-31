'use strict';
// Whole-corpus native-six inference; no difficulty or target tags are read by
// the model. Mirrored raw files also exercise parse/feature/head composition.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {ROOT}=require('./data.cjs'),runtime=require('./coarse-runtime.cjs'),{mirrorText}=require('./mirror.cjs');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const profilePath=process.argv[2],output=process.argv[3];
if(!profilePath||!output)throw Error('Usage: node six_runtime_audit.cjs <profile.json> <new-output-directory>');
const profile=read(profilePath),model=profile.model||profile,options={physicalSupport:profile.postprocess?.physicalSupport??'none'};
const cache=path.join(ROOT,'.cache/rework-six-physical-r2'),inventory=read(path.join(cache,'inventory.json')),schema=read(path.join(cache,'spec.json'));
const directory=path.resolve(output);fs.mkdirSync(directory,{recursive:true});
const protocol={id:'actual-six-coarse-runtime-r3',modelProfile:profilePath,model,postprocess:options,
 scope:'native six-key arithmetic, coverage, held activity, mirror and feature-domain diagnostics; no human pattern truth',
 sourceSnapshot:Object.fromEntries(['six_runtime_audit.cjs','coarse-runtime.cjs','physical-support.cjs','physical.cjs','anchor.cjs','release-geometry.cjs','mirror.cjs','composition.cjs'].map(n=>[n,fs.readFileSync(path.join(__dirname,n),'utf8')])),
 cacheSpecification:schema,mirrorSelection:'32 note-count quantiles; no predicted labels or difficulty used for selection',
 denominator:{requested:inventory.requested,valid:inventory.valid,rejected:inventory.errors}};
const manifest=path.join(directory,'protocol.json');
if(fs.existsSync(manifest))assert.deepEqual(read(manifest),protocol,'Audit candidate changed; use a new output directory');
else fs.writeFileSync(manifest,JSON.stringify(protocol,null,2)+'\n');
function cachedWindows(value){
 return value.windows[model.config.windowMs].map(w=>({...w,structurallyEmpty:w.empty,
  metrics:{...Object.fromEntries(schema.baseKeys.map((k,j)=>[k,w.base[j]])),...Object.fromEntries(schema.keys.map((k,j)=>[k,w.x[j]]))}}));
}
function inspect(result,first,last){
 assert.equal(result.windows[0].start,first);assert.equal(result.windows.at(-1).end,last+1);
 assert.equal(result.segments[0].start,first);assert.equal(result.segments.at(-1).end,last+1);
 for(const rows of[result.windows,result.segments])for(let i=0;i<rows.length;i++){
  assert.ok(rows[i].end>rows[i].start);if(i)assert.equal(rows[i-1].end,rows[i].start);
 }
 const weights=result.windows.map(w=>w.aggregationWeight);assert.ok(weights.every(w=>Number.isFinite(w)&&w>=0));
 for(const w of result.windows){
  if(w.metrics.occupancy>0||w.metrics['release:tailRatePerKey']>0)assert.notEqual(w.tag,'break');
  if(w.structurallyEmpty){assert.equal(w.tag,'break');assert.equal(w.aggregationWeight,0);}
 }
 assert.ok(Object.values(result.coarse.composition).every(p=>Number.isFinite(p)&&p>=0));
 const merged=runtime.aggregateSegments(result.segments,[...model.classes,'break']);
 const comp=Object.values(result.coarse.composition);
 assert.ok(comp.every((p,i)=>Math.abs(p-merged.distribution[i])<1e-9));
 assert.ok(Math.abs(result.coarse.evidence.diversity-merged.diversity)<1e-9);
}
const rows=[],dominant={},windowTotals={count:0,empty:0,heldOnly:0,clippedDuration:0,duration:0};
for(const r of inventory.records){
 const value=read(path.join(cache,r.uid.replaceAll('/','__')+'.json')),windows=cachedWindows(value);
 const result=runtime.analyzeWindows(windows,model,6,options);inspect(result,value.first,value.last);
 const duration=value.last+1-value.first;dominant[result.coarse.dominant]=(dominant[result.coarse.dominant]||0)+1;
 for(const w of result.windows){windowTotals.count++;windowTotals.empty+=Number(w.structurallyEmpty);
  windowTotals.heldOnly+=Number(w.notes===0&&w.metrics.occupancy>0);
  windowTotals.clippedDuration+=(w.end-w.start)*w.clippedFeatureFraction;windowTotals.duration+=w.end-w.start;}
 rows.push({uid:r.uid,dominant:result.coarse.dominant,composition:result.coarse.composition,durationMs:duration,
  clippedFeatureFraction:result.modelInfo.clippedFeatureFraction,maxStandardizedMagnitude:Math.max(...result.windows.map(w=>w.maxStandardizedMagnitude)),
  evidence:result.coarse.evidence});
 if(rows.length%250===0)console.log('actual-six model inference',rows.length);
}
fs.writeFileSync(path.join(directory,'predictions.json'),JSON.stringify(rows,null,2)+'\n');
const sorted=[...inventory.records].sort((a,b)=>a.notes-b.notes||a.uid.localeCompare(b.uid));
const fixtures=[...new Set(Array.from({length:32},(_,i)=>sorted[Math.round(i*(sorted.length-1)/31)]))];
const mirror=[];
for(const r of fixtures){
 const text=fs.readFileSync(path.join(ROOT,r.source),'utf8'),a=runtime.classifyCoarse(text,profile,options),b=runtime.classifyCoarse(mirrorText(text),profile,options);
 inspect(a,a.map.firstNote,a.map.lastNote);inspect(b,b.map.firstNote,b.map.lastNote);
 assert.equal(a.windows.length,b.windows.length);let featureError=0,probabilityError=0,weightError=0;
 for(let i=0;i<a.windows.length;i++){
  const x=a.windows[i],y=b.windows[i];assert.equal(x.start,y.start);assert.equal(x.end,y.end);assert.equal(x.tag,y.tag);
  for(const k of model.keys)featureError=Math.max(featureError,Math.abs(x.metrics[k]-y.metrics[k]));
  x.probability.forEach((p,j)=>probabilityError=Math.max(probabilityError,Math.abs(p-y.probability[j])));
  weightError=Math.max(weightError,Math.abs(x.aggregationWeight-y.aggregationWeight));
 }
 const p=Object.values(a.coarse.composition),q=Object.values(b.coarse.composition),compositionError=Math.max(...p.map((v,j)=>Math.abs(v-q[j])));
 assert.ok(featureError<1e-9&&probabilityError<1e-9&&weightError<1e-8&&compositionError<1e-9,
  JSON.stringify({uid:r.uid,featureError,probabilityError,weightError,compositionError}));
 assert.deepEqual(a.segments.map(s=>[s.start,s.end,s.tag]),b.segments.map(s=>[s.start,s.end,s.tag]));
 assert.equal(a.coarse.dominant,b.coarse.dominant);
 mirror.push({uid:r.uid,notes:r.notes,featureError,probabilityError,weightError,compositionError});
}
const quantile=(v,p)=>[...v].sort((a,b)=>a-b)[Math.floor(p*(v.length-1))];
const summary={scope:protocol.scope,profile:profilePath,requested:inventory.requested,valid:rows.length,rejected:inventory.errors,
 windowTotals,dominantCounts:dominant,mirrorCases:mirror.length,
 clipping:{durationWeighted:windowTotals.clippedDuration/windowTotals.duration,
  chartMedian:quantile(rows.map(r=>r.clippedFeatureFraction),.5),chartQ95:quantile(rows.map(r=>r.clippedFeatureFraction),.95)},
 note:'Dominant counts and support concentration describe model behavior, not human six-key accuracy; no six-key labels selected model parameters.'};
for(const [file,value]of[['summary.json',summary],['predictions.json',rows],['mirror.json',mirror]])fs.writeFileSync(path.join(directory,file),JSON.stringify(value,null,2)+'\n');
console.log(JSON.stringify(summary,null,2));
