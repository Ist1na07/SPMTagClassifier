'use strict';
const fs=require('node:fs'),zlib=require('node:zlib');
const {loadLab}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const rows=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r1.jsonl.gz')).toString().trim().split('\n').map(JSON.parse);
const api=loadLab({patches:[actionRatePatch(profile.gate)]}),lab=api.lab;
const native=rows.map(r=>lab.classifySegments(r.cm,{scorer:'model'}));
const smooth=x=>lab.smoothSegments(structuredClone(x));
const signature=x=>JSON.stringify(x.map(s=>[s.start,s.end,s.family,s.tag,s.secondary??null]));
const baselineSmooth=native.map(x=>signature(smooth(x)));
const out=[];
for(const family of ['rc','ln']){
 const defs=family==='rc'?lab.RC_DEFAULTS:lab.LN_DEFAULTS;
 for(const[key,base]of Object.entries(defs)){
  // Endpoints can cross: ss explicitly defines a valid hard threshold then.
  // These are diagnostic probes, not claimed tuned candidates.
  const probes=[];
  for(const value of [base*.5,base*1.5]){
   defs[key]=value;let changedRule=0,changedBlend=0,changedPrimary=0,changedSecondary=0,changedSmoothMaps=0,changedScoreMs=0;
   for(let i=0;i<rows.length;i++){
    const r=rows[i],now=lab.classifySegments(r.cm,{scorer:'model'});
    changedSmoothMaps+=signature(smooth(now))!==baselineSmooth[i];
    for(let j=0;j<r.parents.length;j++){
     const s=r.parents[j],rule=family==='rc'?lab.scoreRc({f:s.f,extras:s.extra,cutDemandRef:r.cutDemandRef}):lab.scoreLn({f:s.f,extras:s.extra});
     const oldRule=s[family==='rc'?'ruleRc':'ruleLn'];
     changedRule+=Object.keys(rule).some(k=>Math.abs(rule[k]-oldRule[k])>1e-12);
     const a=native[i][j],b=now[j];
     const changed=Object.keys(a.scores??{}).some(k=>Math.abs(a.scores[k]-(b.scores??{})[k])>1e-12);
     changedBlend+=changed;changedScoreMs+=changed*(b.end-b.start);
     changedPrimary+=a.tag!==b.tag||a.family!==b.family;
     changedSecondary+=(a.secondary??null)!==(b.secondary??null);
    }
   }
   probes.push({value,changedRule,changedBlend,changedPrimary,changedSecondary,changedSmoothMaps,changedScoreMs});
   defs[key]=base;
  }
  const rule=probes.some(p=>p.changedRule),blend=probes.some(p=>p.changedBlend),decision=probes.some(p=>p.changedPrimary||p.changedSecondary||p.changedSmoothMaps);
  const status=!rule?'no-rule-change-in-probes':!blend?'rule-changes-discarded-by-model-path':!decision?'support-changes-without-decision-change':'decision-active';
  out.push({family,key,base,status,probes});console.log(family,key,status);
 }
}
fs.mkdirSync('research/core_retune/results',{recursive:true});
fs.writeFileSync('research/core_retune/results/rule-parameter-audit.json',JSON.stringify({scope:'All existing natural parents; current model path, physical readout excluded (no certified exclusions in this input inventory). Perturbation sensitivity is not validation.',records:rows.length,parents:rows.reduce((n,r)=>n+r.parents.length,0),summary:out.reduce((s,r)=>(s[r.status]=(s[r.status]??0)+1,s),{}),parameters:out,source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
console.log('audit complete');
