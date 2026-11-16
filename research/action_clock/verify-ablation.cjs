'use strict';
const fs=require('node:fs'),path=require('node:path'),cp=require('node:child_process'),vm=require('node:vm'),zlib=require('node:zlib');
const {bundle}=require('../../scripts/bundle.cjs');
const BASE='b311c408ef02d6705d30b51a4ea789d3e86d2976',ROOT=path.resolve(__dirname,'../..'),cache=path.join(ROOT,'.cache/action-clock-baseline-src');
function show(file){return cp.execFileSync('git',['show',BASE+':'+file],{cwd:ROOT,maxBuffer:20*1024*1024});}
if(!fs.existsSync(cache)){
 const files=cp.execFileSync('git',['ls-tree','-r','--name-only',BASE,'src','package.json'],{cwd:ROOT,encoding:'utf8'}).trim().split('\n');
 for(const file of files){if(!file.startsWith('src/')&&file!=='package.json')throw Error('Unexpected baseline path');const dest=path.join(cache,file);fs.mkdirSync(path.dirname(dest),{recursive:true});fs.writeFileSync(dest,show(file));}
 fs.writeFileSync(path.join(cache,'BASELINE.json'),JSON.stringify({commit:BASE}));
}else if(JSON.parse(fs.readFileSync(path.join(cache,'BASELINE.json'))).commit!==BASE)throw Error('Preserve other baseline directory');
const before=require(path.join(cache,'src/index.cjs')),combined=require('../../src/index.cjs');
const refreshCadence=process.argv.includes('--refresh-cadence'),musicArg=process.argv.indexOf('--refresh-music'),refreshMusic=musicArg>=0,refresh=refreshCadence||refreshMusic;
const repairFile=refreshMusic?process.argv[musicArg+1]:'research/action_clock/results/range-repair-uids.json';
const changedRanges=new Set(JSON.parse(fs.readFileSync(repairFile,'utf8')).byRadius['1']);
const retained=refresh?Object.fromEntries(['before','action-only','music-only','combined'].map(n=>[n,new Map(zlib.gunzipSync(fs.readFileSync(`research/action_clock/results/ablation-${n}.jsonl.gz`)).toString().trim().split('\n').map(s=>{const r=JSON.parse(s);return[r.uid,r];}))])):null;
const previous=refresh?JSON.parse(fs.readFileSync('research/action_clock/results/ablation-native.json','utf8')):null;
if(refresh)for(const n of refreshMusic?['music-only','combined']:['action-only','music-only','combined']){const f=`research/action_clock/results/ablation-${n}.jsonl.gz`,snapshot=f.replace('.jsonl.gz',refreshMusic?'.before-quantization.jsonl.gz':'.r0.jsonl.gz');if(!fs.existsSync(snapshot))fs.copyFileSync(f,snapshot);}
const overrides=new Map(['src/cadence.cjs','src/segment-readout.cjs'].map(f=>[path.join(ROOT,f),show(f).toString('utf8')]));
const compiled=bundle(path.join(ROOT,'src/index.cjs'),{overrides}),box={};vm.runInNewContext('var api='+compiled.code,box,{codeGeneration:{strings:false,wasm:false}});const musicOnly=box.api;
const records=fs.readFileSync('.cache/perception-final-resource.jsonl','utf8').trim().split('\n').map(JSON.parse),arms=['before','action-only','music-only','combined'],out=Object.fromEntries(arms.map(n=>[n,[]]));
const reference=new Map(zlib.gunzipSync(fs.readFileSync('research/action_clock/results/music-nearest-1.jsonl.gz')).toString().trim().split('\n').map(s=>{const r=JSON.parse(s);return [r.uid,r]}));
const joint=JSON.parse(fs.readFileSync('research/action_clock/results/joint-full-selection.json','utf8')).selected;
if(joint.radius!==1||joint.mode!=='nearest'||Object.values(joint.consumers).some(c=>c.config.confidenceFloor!==.5||c.config.lowNpsScale!==1))throw Error('Update adoption and reference for selected joint configuration');
const times={before:[],combined:[]};let error=0;
function labels(frames){const out=[];for(const s of frames){const prev=out.at(-1);if(prev&&prev.end===s.start&&prev.tag===s.tag&&prev.secondary===s.secondary)prev.end=s.end;else out.push({start:s.start,end:s.end,tag:s.tag,secondary:s.secondary});}return out;}
for(let ri=0;ri<records.length;ri++){
 const r=records[ri],text=fs.readFileSync(r.source,'utf8'),rows=Object.fromEntries(arms.map(n=>[n,{uid:r.uid,consumers:{}}]));
 for(const name of ['actions','detail','compact']){
  const opts=name==='actions'?{}:{ratingEvidence:r.ratings[name]},results={};
  for(const arm of arms){
   if(refresh&&(arm==='before'||refreshMusic&&(arm==='action-only'||!changedRanges.has(r.uid))||refreshCadence&&arm==='music-only'&&!changedRanges.has(r.uid))){results[arm]=retained[arm].get(r.uid).consumers[name];rows[arm].consumers[name]=results[arm];continue;}
   const api=arm==='before'?before:arm==='music-only'?musicOnly:combined;const t=performance.now(),a=api.analyzeMap(text,{...opts,...(arm==='action-only'?{_musicalSegmenter:false}:{})});if(times[arm])times[arm].push(performance.now()-t);
   const segments=a.segments.map(({start,end,tag,secondary,equivBpm,evidence})=>({start,end,tag,secondary,equivBpm,...(evidence?.cadence?{cadence:evidence.cadence}:{}),...(evidence?.cadenceUnresolved?{cadenceUnresolved:evidence.cadenceUnresolved}:{})}));
   results[arm]={tags:a.overall.tags,percentages:a.overall.percentages,segments};rows[arm].consumers[name]=results[arm];
  }
  for(const [a,b]of [['before','action-only'],['music-only','combined']]){
   if(JSON.stringify(results[a].tags)!==JSON.stringify(results[b].tags)||JSON.stringify(results[a].percentages)!==JSON.stringify(results[b].percentages)||JSON.stringify(labels(results[a].segments))!==JSON.stringify(labels(results[b].segments)))throw Error('Cadence-only arm changed classification/integral '+r.uid+'/'+name+'/'+a);
  }
  const expected=reference.get(r.uid).consumers[name];if(JSON.stringify(results.combined.tags)!==JSON.stringify(expected.tags)||JSON.stringify(labels(results.combined.segments))!==JSON.stringify(labels(expected.segments)))throw Error('Final source differs from selected native music grid '+r.uid+'/'+name);
  for(const[k,v]of Object.entries(expected.percentages))error=Math.max(error,Math.abs(v-results.combined.percentages[k]));
 }
 for(const arm of arms)out[arm].push(rows[arm]);
 if((ri+1)%75===0)console.log('four arms',ri+1);
}
if(error>1e-8)throw Error('Integral mismatch '+error);
for(const arm of arms)if(!(refresh&&(arm==='before'||refreshMusic&&arm==='action-only')))fs.writeFileSync(`research/action_clock/results/ablation-${arm}.jsonl.gz`,zlib.gzipSync(out[arm].map(r=>JSON.stringify(r)).join('\n')+'\n'));
const timing={...(previous?.timing??{})};for(const[n,ts]of Object.entries(times)){if(!ts.length)continue;ts.sort((a,b)=>a-b);if(refreshMusic)continue;timing[n]={medianMs:ts[Math.floor(ts.length*.5)],p95Ms:ts[Math.floor(ts.length*.95)],maxMs:ts.at(-1)};}
fs.writeFileSync('research/action_clock/results/ablation-native.json',JSON.stringify({baseline:BASE,inputs:records.length,consumers:3,arms,maxPercentagePointError:error,classificationAndIntegralInvariantForCadenceOnly:true,timing,...(previous?.refresh?{refresh:previous.refresh}:{}),...(refresh?{executionRepairs:[...(previous?.executionRepairs??[]),{mode:refreshMusic?'quantization-aware music cuts':'cadence and window range',source:repairFile,changedMusicInputs:changedRanges.size,baselineReused:true,actionOnlyReused:refreshMusic,timingScope:refreshMusic?'prior full-corpus timing retained; no new full timing claim':'full-corpus cadence refresh'}]}:{})},null,2));
const file='examplebeatmap/'+fs.readdirSync('examplebeatmap').find(f=>f.endsWith('.osu')),text=fs.readFileSync(file,'utf8'),example=[];
for(const speedRate of [.75,1,1.5])for(const arm of ['before','combined']){
 const a=(arm==='before'?before:combined).analyzeMap(text,{speedRate});example.push({speedRate,arm,segments:a.segments.filter(s=>s.end>131980/speedRate&&s.start<142582/speedRate).map(({start,end,tag,equivBpm,evidence})=>({start,end,tag,equivBpm,cadence:evidence.cadence,musicalStart:evidence.musicalStart,musicalEnd:evidence.musicalEnd})),modelInfo:a.modelInfo});
}
fs.writeFileSync('research/action_clock/results/example-final.json',JSON.stringify({file,targetMs:[131980,142582],example},null,2));console.log('complete four native arms, max percentage error',error,timing);
