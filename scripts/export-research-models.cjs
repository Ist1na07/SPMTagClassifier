'use strict';
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const arguments_=process.argv.slice(2),index=arguments_.indexOf('--root');let root;
if(index>=0){root=arguments_[index+1];if(!root)throw Error('--root requires a reproduced result directory');arguments_.splice(index,2);}
const source=root?{4:path.join(root,'coarse-final/4k/profile.json'),6:path.join(root,'coarse-final/6k/profile.json'),7:path.join(root,'ln-final/profile.json')}:
 {4:'research/rework/results/coarse-final-r1/4k/profile.json',6:'research/rework/results/coarse-final-r1/6k/profile.json',7:'research/rework/results/ln-final-r1/profile.json'};
for(const k of arguments_.length?arguments_:['4','6','7']){
 if(!Object.hasOwn(source,k))throw Error('Expected native key count 4, 6 or 7');
 const p=JSON.parse(fs.readFileSync(source[k],'utf8'));
 assert.ok(['small_gradient','small_projected_gradient','train_prediction_stable','frozen_archived_prior'].includes(p.fit.status),'Unaccepted all-data fit');
 const model=k==='7'?Object.fromEntries(['variant','keys','mean','scale','delta','clip'].map(n=>[n,p.model[n]])):p.model;
 const profile={model,...(k==='7'?{policy:p.policy,extraGate:p.extraGate}:{postprocess:p.postprocess}),
  evidence:p.evidence,provenance:{source:source[k],selection:'all-data choice from repeated inner held-out predictions; not an outer winner',fitStatus:p.fit.status}};
 fs.writeFileSync(`src/models/native${k}k.json`,JSON.stringify(profile)+'\n');console.log('Exported native'+k+'k from',source[k]);
}
