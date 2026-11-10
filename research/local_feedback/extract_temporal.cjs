'use strict';
// Recompute proposals, not parent logits, on short centered physical windows.
const fs=require('node:fs');
const {loadLab,context}=require('../rework/legacy-lab.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{regularRowCadence}=require('../../src/cadence.cjs');
const api=loadLab(),TYPES=['dense chordstream','fast chordstream','tech'];
const WIDTHS=[1000,2000,4000],variants=WIDTHS.flatMap(width=>[false,true].map(cadence=>({width,cadence})));
const output='.cache/local-feedback-temporal-r4b.jsonl';if(fs.existsSync(output))throw Error('Preserve existing temporal extraction');
let count=0,children=0,proposals=0;
for(const line of fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n')){
  const r=JSON.parse(line),text=fs.readFileSync(r.source,'utf8'),old=context(api.lab,text),native=prepare(parseOsu(text));
  const active=old.windows.map(w=>w.cutDemand).filter(x=>x>0).sort((a,b)=>a-b),ref=active[Math.floor(active.length*.8)]??1,frames=[],odds=[],margins=[];
  for(let parent=0;parent<r.frames.length;parent++){
    const f=r.frames[parent],refine=f.family==='rc'&&['dense chordstream','fast chordstream'].includes(f.tag);
    if(!refine){frames.push({...f,parent});odds.push(variants.map(()=>0));margins.push(variants.map(()=>-1000));continue;}
    for(let start=f.start;start<f.end;start+=250){
      const end=Math.min(start+250,f.end),center=(start+end)/2,values=[],globalMargins=[];
      for(const width of WIDTHS){
        const a=Math.max(old.map.firstNote,center-width/2),b=Math.min(old.map.lastNote,center+width/2);
        const feature=api.lab.computeWindowFeatures(old.ctx,a,b),extra=api.lab.computeSegmentExtras(old.ctx,a,b);
        const measured=regularRowCadence(native,a,b);
        for(const cadence of [false,true]){
          const ff=cadence&&measured?{...feature,equivBpm:measured.equivBpm}:feature;
          const scores=api.lab.modelScores('rc',ff,extra,ref),a0=Math.max(scores.tech??0,1e-300),b0=Math.max((scores['dense chordstream']??0)+(scores['fast chordstream']??0),1e-300);
          values.push(Math.log(a0)-Math.log(b0));globalMargins.push(Math.log(a0)-Math.log(Math.max(1e-300,...Object.entries(scores).filter(([c])=>c!=='tech').map(([,v])=>v))));proposals++;
        }
      }
      frames.push({...f,start,end,parent});odds.push(values);margins.push(globalMargins);
    }
  }
  fs.appendFileSync(output,JSON.stringify({uid:r.uid,frames,localLogOdds:odds,localGlobalMargins:margins})+'\n');children+=frames.length;
  if(++count%50===0)console.log('temporal proposals',count,children,proposals);
}
fs.writeFileSync(output+'.schema.json',JSON.stringify({records:count,children,proposals,variants,stepMs:250,
  scope:'Only inherited RC Dense/Fast CS parents can be refined; all other raw intervals preserved. Native local expert inference, not a new difficulty scorer call.',source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
console.log('complete',count,children,proposals);
