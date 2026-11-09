'use strict';
// The historical local/Fxx caches contain original-song coordinates, not the
// standalone excerpt clock. Extract full natural excerpts independently.
const fs=require('node:fs');
const {loadLab,context,inspectPrepared}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('./break-regularity.cjs'),profile=require('../../src/models/overlay7k.json');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{sequenceEvidence}=require('./sequence-evidence.cjs');
const output='.cache/targeted-natural-queries-r1.json';
if(fs.existsSync(output))throw Error('Preserve completed independent query inputs');
const api=loadLab({patches:[actionRatePatch(profile.gate)]}),rows=[];
for(const name of ['F01','F02']){
  const source=`artifacts/frontier-annotation-r5/${name}.osu`,text=fs.readFileSync(source,'utf8');
  const ctx=prepare(parseOsu(text)),parents=inspectPrepared(api,context(api.lab,text)).classified.map(s=>({...s,secondary:s.secondary??null,scores:s.scores??{[s.tag]:1}}));
  const frames=[],values=[],parentIds=[];
  parents.forEach((p,j)=>{
    const n=p.family==='rc'&&['tech','dense chordstream','fast chordstream'].includes(p.tag)?Math.max(1,Math.floor((p.end-p.start)/500)):1;
    for(let k=0;k<n;k++){
      const start=p.start+(p.end-p.start)*k/n,end=p.start+(p.end-p.start)*(k+1)/n,mid=(start+end)/2;
      frames.push({...p,start,end});parentIds.push(j);values.push(sequenceEvidence(ctx,Math.max(p.start,mid-1000),Math.min(p.end,mid+1000)));
    }
  });
  rows.push({uid:'natural/'+name,name,source,parents,originalValues:parents.map(s=>sequenceEvidence(ctx,s.start,s.end)),local:{frames,values,parentIds}});
}
fs.writeFileSync(output,JSON.stringify({sourceSnapshot:fs.readFileSync(__filename,'utf8'),rows})+'\n');console.log('Natural full-excerpt query inputs',rows.length);
