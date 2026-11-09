'use strict';
const fs = require('node:fs'), assert = require('node:assert/strict');
const { loadLab, context, inspectPrepared } = require('../rework/legacy-lab.cjs');
const { records7, scoreIntervals } = require('../rework/data.cjs');
const { GRID, gridPatch } = require('./grid-input.cjs');
const root = 'research/overlay-quality/grid-input-r1';
const boundaries = [1500, 3000, 6000].flatMap(minSegLenMs => [.025, .05, .1].map(segProminenceAbs => ({ minSegLenMs, segProminenceAbs })));
const configs = GRID.flatMap((grid, gridId) => boundaries.map((boundary, partition) => ({ gridId, grid, partition, boundary })));
fs.mkdirSync(root, { recursive: true });
const protocol = { configs, clock: 'Converted milliseconds; row-merging resolution stays10ms. Fixed display smoothing1800ms in all arms.',
  change: 'Only fitGrid coverage tolerance and candidate-spacing floor; all original classifier coefficients,other features/rules/thresholds unchanged.',
  scope: '2x2 matched algorithm ablation. Relative-tolerance ON retunes [.1,.15,.2,.25], OFF keeps30ms; both retune9partitions and245post independently using original6x3groups.',
  budget: 'Final bounded cadence-input block, no new classifier fit. Stop if no worthwhile practical improvement.',
  sourceSnapshots: Object.fromEntries(['grid-input.cjs','extract-grid-input.cjs'].map(f => [f, fs.readFileSync(__dirname+'/'+f,'utf8')])) };
const protocolFile = root+'/extraction-protocol.json';
if (fs.existsSync(protocolFile)) assert.deepEqual(JSON.parse(fs.readFileSync(protocolFile,'utf8')),protocol);
else fs.writeFileSync(protocolFile,JSON.stringify(protocol,null,2)+'\n');
const previous = new Map(fs.readFileSync('.cache/overlay-bilateral-partitions-r1/original-4.jsonl','utf8').trim().split('\n').map(s => {const r=JSON.parse(s);return[r.uid,r];}));
const records=records7(), texts=new Map();
for(const r of records)if(!texts.has(r.path))texts.set(r.path,fs.readFileSync(r.path,'utf8'));
const frames = a => a.map(s => ({...s, secondary:s.secondary??null, scores:s.scores??{[s.tag]:1}}));
for(let gi=0;gi<GRID.length;gi++) {
  const file=root+`/grid-${gi}.jsonl`;
  if(fs.existsSync(file+'.complete.json'))continue;
  const api=loadLab({patches:[gridPatch(GRID[gi])]});const rows=[];let parity=0;
  for(const r of records) {
    const prepared=context(api.lab,texts.get(r.path));const values=[];
    for(let pi=0;pi<boundaries.length;pi++) {
      Object.assign(api.lab.CONFIG,boundaries[pi]);
      const view=inspectPrepared(api,prepared), raw=frames(view.classified);
      if(gi===0&&pi===4){assert.deepEqual(JSON.parse(JSON.stringify(raw)),previous.get(r.uid).frames);parity++;}
      values.push({frames:raw,local:r.dataset==='manual'?scoreIntervals(r.annotations,raw):null});
    }
    rows.push({uid:r.uid,partitions:values});
  }
  fs.writeFileSync(file+'.tmp',rows.map(r=>JSON.stringify(r)).join('\n')+'\n');fs.renameSync(file+'.tmp',file);
  fs.writeFileSync(file+'.complete.json',JSON.stringify({records:rows.length,partitions:9,baselineParity:parity})+'\n');
  console.log('Grid input bank',gi,GRID[gi],rows.length);
}
const player=[];
for(const [pack,names,target]of[
  ['r2',['C16','C24','C32','C48'],[1000,9000]],['r7',['H01','H02','H03','H04'],[9000,17000]],['r5',['F01','F02'],[9000,16500]]]) {
  for(const name of names){
    const text=fs.readFileSync(`artifacts/frontier-annotation-${pack}/${name}.osu`,'utf8'),values=[];
    for(let gi=0;gi<GRID.length;gi++) {
      const api=loadLab({patches:[gridPatch(GRID[gi])]});const prepared=context(api.lab,text);const parts=[];
      for(const boundary of boundaries){Object.assign(api.lab.CONFIG,boundary);parts.push(frames(inspectPrepared(api,prepared).classified));}
      values.push(parts);
    }
    player.push({uid:name,targetMs:target,frames:values,truthScope:'Existing queried local/structural regressions only; not used for selection.'});
  }
}
fs.writeFileSync(root+'/player.json',JSON.stringify(player)+'\n');
fs.writeFileSync(root+'/complete.json',JSON.stringify({records:records.length,configs,baselineParity:JSON.parse(fs.readFileSync(root+'/grid-0.jsonl.complete.json','utf8')).baselineParity,
  classifierFits:0,ratingPredictorCalls:0})+'\n');
console.log('Complete cadence-input bank',configs.length);
