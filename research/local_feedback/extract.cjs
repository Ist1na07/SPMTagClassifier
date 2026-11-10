'use strict';
// Actual natural segment intervals; no annotation endpoints enter extraction.
const fs=require('node:fs');
const {prepare}=require('../../src/parser.cjs');
const {prepareHandSequence,handSequence,KEYS:SEQ}=require('../frontier/hand-sequence.cjs');
const {handTransitions,KEYS:TRANS}=require('../frontier/hand-transitions.cjs');
const output='.cache/local-feedback-r2-inputs.jsonl';
if(fs.existsSync(output))throw Error('Preserve existing feature extraction');
let count=0,frames=0;
for(const line of fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n')){
  const r=JSON.parse(line),ctx=prepare(r.map),shared=prepareHandSequence(ctx);
  const values=r.frames.map(f=>[...handSequence(ctx,f.start,f.end,shared).values,...handTransitions(ctx,f.start,f.end,shared).values].map(x=>x??0));
  fs.appendFileSync(output,JSON.stringify({uid:r.uid,geometry:r.geometry,hand:values})+'\n');
  frames+=r.frames.length;if(++count%100===0)console.log('local feedback geometry',count,frames);
}
fs.writeFileSync(output+'.schema.json',JSON.stringify({handKeys:[...SEQ,...TRANS],records:count,frames,source:__filename},null,2)+'\n');
console.log('complete',count,frames);
