'use strict';
// Counterfactual experts on the SAME natural intervals, including Break.
// No annotation boundaries, copied whole labels, or new difficulty calls.
const fs=require('node:fs');
const {loadLab,inspect}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const api=loadLab({patches:[actionRatePatch(profile.gate)]});
const output='.cache/local-feedback-routing-r3.jsonl';
if(fs.existsSync(output))throw Error('Preserve existing extraction');
let count=0,frames=0;
for(const line of fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n')){
  const row=JSON.parse(line),r=inspect(api,fs.readFileSync(row.source,'utf8'));
  if(r.classified.length!==row.frames.length)throw Error('Natural interval inventory changed: '+row.uid);
  const candidates=r.segments.map((s,j)=>{
    const old=row.frames[j],now=r.classified[j];
    if(old.start!==s.start||old.end!==s.end||old.tag!==now.tag||old.family!==now.family)throw Error('Natural segment changed: '+row.uid+'/'+j);
    const rc=api.lab.blendScores(s.modelRc,s.ruleRc,'model'),ln=api.lab.blendScores(s.modelLn,s.ruleLn,'model');
    const active=old.family==='rc'?rc:old.family==='ln'?ln:null;
    if(active)for(const [c,v]of Object.entries(old.scores))if(Math.abs(v-(active[c]??0))>1e-12)throw Error('Expert changed: '+c);
    const lnDom=s.f.lnWeightRatio>=api.lab.CONFIG.lnDominantRatio||(s.f.lockRatio>=.5&&s.f.heldCoverage>=.15&&s.f.lnWeightRatio>=.2);
    return {rc,ln,underlyingFamily:lnDom?'ln':'rc',legacyLnWeightRatio:s.f.lnWeightRatio};
  });
  fs.appendFileSync(output,JSON.stringify({uid:row.uid,candidates})+'\n');frames+=candidates.length;
  if(++count%75===0)console.log('counterfactual routing experts',count,frames);
}
fs.writeFileSync(output+'.schema.json',JSON.stringify({records:count,frames,scorerCalls:0,source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
console.log('complete routing experts',count,frames);
