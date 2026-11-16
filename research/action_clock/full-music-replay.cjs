'use strict';
const fs=require('node:fs'),zlib=require('node:zlib'),api=require('../../src/index.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{musicalSegments}=require('../../src/musical-segmentation.cjs');
const resources=fs.readFileSync('.cache/perception-final-resource.jsonl','utf8').trim().split('\n').map(JSON.parse);
const extendContrast=process.argv.includes('--extend-contrast');
const configs=extendContrast?[.5,1,2].map(maxMoveBeats=>({id:'contrast-'+maxMoveBeats,maxMoveBeats,mode:'contrast'})):[{id:'off'},...[.5,1,2].map(maxMoveBeats=>({id:'nearest-'+maxMoveBeats,maxMoveBeats,mode:'nearest'}))];
const protocol={scope:'Native 411-input x 3-consumer complete music-parameter replay; historically exposed development labels.',configs,
 amendment:'After six-chart mechanism check and before full-whole-map replay, include whole-tag macro P/R/F1 and exact match for every distinct radius. Contrast/nearest were identical on all six charts because the allowed radii normally contain one bar candidate; keep duplicate-grid evidence but do not duplicate the full replay.',
 selectionScore:'mean manual-chart duration accuracy across all consumers + mean full349 consumer macro-F1; equal dimensionless weights. Report individual metrics and all tradeoffs.',cadence:'current action-clock prototype in all arms; cadence is display-only and cannot change integral scores.'};
if(!process.argv.includes('--repair-ranges')&&!process.argv.includes('--repair-file')&&!extendContrast)fs.writeFileSync('research/action_clock/results/full-protocol.json',JSON.stringify(protocol,null,2));
const repairFlag=process.argv.indexOf('--repair-file'),repairFile=repairFlag>=0?process.argv[repairFlag+1]:process.argv.includes('--repair-ranges')?'research/action_clock/results/range-repair-uids.json':null;
const ordering=extendContrast?JSON.parse(fs.readFileSync('research/action_clock/results/music-ordering-audit.json','utf8')):null;
const repair=extendContrast?{reason:'All411 nearest/contrast candidate-order audit; reuse identical native cuts.',byRadius:Object.fromEntries([.5,1,2].map(r=>[r===.5?'.5':String(r),ordering.differences.filter(d=>d.radius===r).map(d=>d.uid)]))}:repairFile?JSON.parse(fs.readFileSync(repairFile,'utf8')):null;
for(const cfg of configs){
 const affected=repair?.byRadius[cfg.maxMoveBeats===.5?'.5':String(cfg.maxMoveBeats)]??[];if(repair&&!extendContrast&&!affected.length)continue;
 const target=`research/action_clock/results/music-${cfg.id}.jsonl.gz`;
 if(extendContrast){fs.copyFileSync(target.replace('contrast-','nearest-'),target);}
 if(repair&&!extendContrast&&!fs.existsSync(target.replace('.jsonl.gz',repairFlag>=0?'.before-quantization.jsonl.gz':'.r0.jsonl.gz')))fs.copyFileSync(target,target.replace('.jsonl.gz',repairFlag>=0?'.before-quantization.jsonl.gz':'.r0.jsonl.gz'));
 const out=repair?zlib.gunzipSync(fs.readFileSync(target)).toString().trim().split('\n').map(JSON.parse):[],times=[];
 for(const r of resources){if(repair&&!affected.includes(r.uid))continue;const text=fs.readFileSync(r.source,'utf8'),ctx=prepare(parseOsu(text)),row={uid:r.uid,consumers:{}};
  for(const name of ['actions','detail','compact']){
   const opts=name==='actions'?{}:{ratingEvidence:r.ratings[name]};opts._musicalSegmenter=false;if(cfg.id!=='off')opts._musicalSegmenter=input=>musicalSegments(ctx,input,cfg);
   const t=performance.now(),a=api.analyzeMap(text,opts);times.push(performance.now()-t);
   row.consumers[name]={tags:a.overall.tags,percentages:a.overall.percentages,segments:a.segments.map(({start,end,tag,secondary,equivBpm})=>({start,end,tag,secondary,equivBpm}))};
  }if(repair)out[out.findIndex(x=>x.uid===r.uid)]=row;else out.push(row);if(out.length%100===0)console.log(cfg.id,out.length);
 }
 const file=`research/action_clock/results/music-${cfg.id}.jsonl.gz`;fs.writeFileSync(file,zlib.gzipSync(out.map(r=>JSON.stringify(r)).join('\n')+'\n'));
 times.sort((a,b)=>a-b);const receiptPath=`research/action_clock/results/music-${cfg.id}-receipt.json`;
 const receipt=extendContrast?{inputs:out.length,consumers:3,sourceReuse:target.replace('contrast-','nearest-'),actualReplayedInputs:affected.length,unchangedNativeCutsReused:out.length-affected.length,timingScope:'No fresh full-corpus performance claim',output:file}:repair?{...JSON.parse(fs.readFileSync(receiptPath,'utf8')),executionRepairs:[...(JSON.parse(fs.readFileSync(receiptPath,'utf8')).executionRepairs??[]),{inputs:affected.length,reason:repair.reason,source:repairFile}]}:{inputs:out.length,consumers:3,medianMs:times[Math.floor(times.length*.5)],p95Ms:times[Math.floor(times.length*.95)],output:file};
 fs.writeFileSync(receiptPath,JSON.stringify(receipt,null,2));
 console.log('complete',cfg.id);
}

if(extendContrast){const file='research/action_clock/results/full-protocol.json',p=JSON.parse(fs.readFileSync(file,'utf8'));p.orderingExtension={comparedNativeCuts:ordering.comparisons,differingMapRadiusPairs:ordering.differences.length,completeSettings:7,reuse:'Only exactly equal cut geometry reuses predictions; differing inputs are replayed through the public API.'};fs.writeFileSync(file,JSON.stringify(p,null,2));}
