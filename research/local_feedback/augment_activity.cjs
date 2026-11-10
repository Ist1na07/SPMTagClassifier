'use strict';
// Obligations belong to an LN's body as well as its final instantaneous tail.
// Otherwise locally identical held prefixes of simultaneous/staggered releases
// cannot be distinguished even though the complete playable already specifies
// the future release geometry. No difficulty scorer is invoked.
const fs=require('node:fs'),{prepare,lowerBound:lb}=require('../../src/parser.cjs');
const output='.cache/local-feedback-activity-r6b.jsonl';if(fs.existsSync(output))throw Error('Preserve completed obligation features');
const schema=JSON.parse(fs.readFileSync('.cache/local-feedback-activity-r6.jsonl.schema.json','utf8'));
const physical=fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n');
const old=fs.readFileSync('.cache/local-feedback-activity-r6.jsonl','utf8').trim().split('\n');
const KEYS=['continuousReleasePairRate','independentReleasePairRate','independentTailRate','heldReleaseObligation','independentHeldObligation'];
const {evidence}=require('./release_obligations.cjs');
let count=0;
for(let i=0;i<physical.length;i++){
  const src=JSON.parse(physical[i]),row=JSON.parse(old[i]);if(src.uid!==row.uid)throw Error('Obligation feature inventory mismatch');
  const ctx=prepare(src.map),measure=evidence(ctx);
  row.values=row.values.map((values,w)=>values.map((v,j)=>{const f=row.frames[j],mid=(f.start+f.end)/2,a=Math.max(src.map.firstNote,mid-schema.widths[w]/2),b=Math.min(src.map.lastNote,mid+schema.widths[w]/2);return[...v.slice(0,36),...measure(a,Math.max(a+1,b))];}));
  fs.appendFileSync(output,JSON.stringify(row)+'\n');if(++count%100===0)console.log('continuous obligations',count);
}
fs.writeFileSync(output+'.schema.json',JSON.stringify({...schema,featureKeys:[...schema.featureKeys.slice(0,36),...KEYS],
  obligations:'Continuous retained-pair cost at the future tail, on both the tail clock and the associated union-hold body. Scale is the existing OD-derived SPM release precision, not a fitted arbitrary millisecond cutoff. Own/future-head guidance follows the same SPM v1.2.0 factor. Body fields are structural obligations, not instantaneous difficulty or an isolated Release label.',
  source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');console.log('complete continuous obligations',count);
