'use strict';
const fs=require('node:fs'),assert=require('node:assert/strict');
const {loadLab,context,inspectPrepared}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('./break-regularity.cjs'),profile=require('../../src/models/overlay7k.json');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{features}=require('../../src/features.cjs');
const {KEYS,sequenceEvidence}=require('./sequence-evidence.cjs');
const output='.cache/overlay-targeted-sequence-r1.jsonl';
if(fs.existsSync(output)||fs.existsSync(output+'.complete.json'))throw Error('Preserve completed or partial input; use a fresh protocol');
const source=fs.readFileSync('.cache/frontier-rc-type-inputs-r1.jsonl','utf8').trim().split('\n').map(JSON.parse);
const api=loadLab({patches:[actionRatePatch(profile.gate)]});
const schema={keys:KEYS,gate:profile.gate,localContextMs:2000,minOutputCellMs:500,
  mutableClasses:['dense chordstream','fast chordstream','tech'],
  notes:'Split only mutable RC parents into equal >=500ms cells; compute centered2s context clipped to parent. Native control preserves original parents. No annotation boundaries or labels enter extraction.',
  sourceSnapshots:Object.fromEntries(['sequence-evidence.cjs','extract-targeted-sequence.cjs'].map(n=>[n,fs.readFileSync(__dirname+'/'+n,'utf8')]))};
fs.writeFileSync(output+'.schema.json',JSON.stringify(schema,null,2)+'\n');
let count=0,cells=0;
for(const old of source){
  const text=fs.readFileSync(old.source,'utf8'),p=context(api.lab,text),ctx=prepare(parseOsu(text));
  const aux=!old.uid.startsWith('spm_dataset-')&&!old.uid.startsWith('manual/');
  const result=inspectPrepared(api,p,aux?{intervals:old.intervals.map(([start,end])=>({start,end}))}:{});
  const parents=result.classified.map(s=>({...s,secondary:s.secondary??null,scores:s.scores??{[s.tag]:1}}));
  assert.deepEqual(parents.map(s=>[s.start,s.end]),old.intervals,old.uid+' boundaries');
  const native=parents.map(s=>sequenceEvidence(ctx,s.start,s.end));
  const effort=parents.map(s=>features(ctx,s.start,Math.max(s.start+1,s.end)).metrics.effort);
  const frames=[],values=[],efforts=[],parentIds=[];
  parents.forEach((s,j)=>{
    const mutable=s.family==='rc'&&schema.mutableClasses.includes(s.tag);
    const n=mutable?Math.max(1,Math.floor((s.end-s.start)/500)):1;
    for(let k=0;k<n;k++){
      const start=s.start+(s.end-s.start)*k/n,end=s.start+(s.end-s.start)*(k+1)/n,mid=(start+end)/2;
      frames.push({...s,start,end});parentIds.push(j);efforts.push(effort[j]);
      values.push(sequenceEvidence(ctx,Math.max(s.start,mid-1000),Math.min(s.end,mid+1000)));
    }
  });
  fs.appendFileSync(output,JSON.stringify({uid:old.uid,source:old.source,sourceUid:old.sourceUid??null,
    native:{frames,values,effort:efforts,parentIds},parents,originalValues:native,originalEffort:effort})+'\n');
  count++;cells+=frames.length;if(count%50===0)console.log('targeted sequence extraction',count,cells);
}
fs.writeFileSync(output+'.complete.json',JSON.stringify({records:count,cells})+'\n');console.log('complete',count,cells);
