'use strict';
// Existing native segmenter, with two explicit physical change signals.
// The classifier/prior is unchanged. Auxiliary labels retain exact intervals.
const fs = require('node:fs'), path = require('node:path'), assert = require('node:assert/strict');
const { loadLab, context, inspectPrepared } = require('../rework/legacy-lab.cjs');
const { records7 } = require('../rework/data.cjs');
const { parseOsu, prepare } = require('../../src/parser.cjs');
const { features } = require('../../src/features.cjs');
const ROOT = path.resolve(__dirname, '../..'), output = path.join(ROOT, '.cache/overlay-bilateral-partitions-r1');
const source = '.cache/frontier-rc-type-inputs-r1.jsonl';
const old = fs.readFileSync(source, 'utf8').trim().split('\n').map(JSON.parse);
const oldLookup = new Map(old.map(r => [r.uid, r]));
const records = records7();
const recordUids = new Set(records.map(r=>r.uid));
const boundaries = [1500, 3000, 6000].flatMap(minSegLenMs => [.025, .05, .1].map(segProminenceAbs => ({minSegLenMs, segProminenceAbs})));
function bilateral(ctx, a, b) {
  let count = 0, notes = 0, bilateralRows = 0, bilateralNotes = 0;
  for (const row of ctx.rows) {
    if (row.t < a) continue;
    if (row.t >= b) break;
    const n = row.mask.toString(2).replaceAll('0', '').length;
    count++; notes += n;
    if ((row.mask & 0b0000111) && (row.mask & 0b1110000)) {bilateralRows++; bilateralNotes += n;}
  }
  return [count ? bilateralRows / count : 0, notes ? bilateralNotes / notes : 0];
}
const physical = {id:'row-rate-and-bilateral-changes', apply(s, replace) {
  s = replace(s, '  var SEG_FEATURES = [', '  var SEG_FEATURES = [\n    "physicalRowRate",\n    "bilateralRowShare",');
  return replace(s, '      nps: notes / durS,', `      nps: notes / durS,
      physicalRowRate: rowCount / durS,
      bilateralRowShare: (()=>{let both=0;for(let j=loIdx;j<hiIdx;j++){const row=rows[j];if(row.t>=t0&&row.t<t1&&row.hits>0&&(row.mask&7)&&(row.mask&112))both++;}return rowCount?both/rowCount:0;})(),`);
}};
const schema = {boundaries, variants:['original','physical'], modelFeatures:['bilateralRowShare','bilateralNoteShare'],
  originalCache:source, classification:'unchanged archived classifier, scorer=model, original LN threshold',
  auxiliaryIntervals:'Exact existing auxiliary sources/intervals/frames retained; new model features only.',
  sourceSnapshots:Object.fromEntries(['extract-bilateral-partitions.cjs'].map(f=>[f,fs.readFileSync(path.join(__dirname,f),'utf8')]))};
fs.mkdirSync(output, {recursive:true});
const schemaFile = path.join(output, 'schema.json');
if (fs.existsSync(schemaFile)) assert.deepEqual(JSON.parse(fs.readFileSync(schemaFile,'utf8')), schema);
else fs.writeFileSync(schemaFile, JSON.stringify(schema,null,2)+'\n');
const prepared = new Map(), modern = new Map(), texts = new Map();
for (const row of old) if (!texts.has(row.source)) {
  const text = fs.readFileSync(path.join(ROOT,row.source),'utf8'); texts.set(row.source,text);
  modern.set(row.source,prepare(parseOsu(text)));
}
let baselineParity = 0;
for (const variant of schema.variants) {
  const api = loadLab({patches:variant==='physical'?[physical]:[]});
  for (let partition=0; partition<boundaries.length; partition++) {
    const file = path.join(output,`${variant}-${partition}.jsonl`);
    if (fs.existsSync(file+'.complete.json')) continue;
    Object.assign(api.lab.CONFIG,boundaries[partition]);
    const rows = [];
    for (const record of records) {
      const previous = oldLookup.get(record.uid), src=previous.source;
      // Feature windows do not depend on the changed segment feature list.
      // Preparing once per physical source/variant avoids repeated extraction.
      const key=variant+':'+src;
      if (!prepared.has(key)) prepared.set(key,context(api.lab,texts.get(src)));
      const view=inspectPrepared(api,prepared.get(key)), frames=view.classified.map(s=>({...s,secondary:s.secondary??null,scores:s.scores??{[s.tag]:1}}));
      const intervals=frames.map(s=>[s.start,s.end]);
      if(variant==='original'&&partition===4){
        assert.deepEqual(intervals,previous.intervals,record.uid+' original interval parity');
        for(let j=0;j<frames.length;j++)for(const field of ['tag','family','confidence','scores'])
          assert.deepEqual(frames[j][field],previous.frames[j][field],record.uid+' '+field);
        baselineParity++;
      }
      rows.push({uid:record.uid,source:src,sourceUid:null,intervals,frames,
        bilateral:intervals.map(([a,b])=>bilateral(modern.get(src),a,b)),
        effort:intervals.map(([a,b])=>features(modern.get(src),a,Math.max(a+1,b)).metrics.effort)});
    }
    for(const previous of old.filter(r=>!recordUids.has(r.uid))){
      const {uid,source:src,sourceUid,sourceGroup,intervals,frames}=previous;
      rows.push({uid,source:src,sourceUid,sourceGroup,intervals,frames,
        bilateral:intervals.map(([a,b])=>bilateral(modern.get(src),a,b)),
        effort:intervals.map(([a,b])=>features(modern.get(src),a,Math.max(a+1,b)).metrics.effort)});
    }
    assert.equal(rows.length,old.length);
    fs.writeFileSync(file+'.tmp',rows.map(r=>JSON.stringify(r)).join('\n')+'\n');fs.renameSync(file+'.tmp',file);
    fs.writeFileSync(file+'.complete.json',JSON.stringify({records:rows.length,windows:rows.reduce((n,r)=>n+r.intervals.length,0),originalBaselineParity:variant==='original'&&partition===4?baselineParity:0})+'\n');
    console.log('bilateral partition',variant,partition,'windows',rows.reduce((n,r)=>n+r.intervals.length,0));
  }
}
const acceptedParity=JSON.parse(fs.readFileSync(path.join(output,'original-4.jsonl.complete.json'),'utf8')).originalBaselineParity;
fs.writeFileSync(path.join(output,'complete.json'),JSON.stringify({partitions:18,originalBaselineParity:acceptedParity,
  source:'Existing source files; no rating Predictor calls; archived classifier unchanged'})+'\n');
