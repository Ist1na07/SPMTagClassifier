'use strict';
// Common profiles were selected on pooled INNER scores, before OUTER replay.
const fs=require('node:fs');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8'));
const {weights,classes}=read('research/perception/recipes.json');
const {EXPECTED_RATING,COMPACT_RATING}=require('../src/perception.cjs');
const profiles={};
for(const [name,model,arm,identity]of [
 ['actions','detail','actions',null],
 ['detail','detail','combined',EXPECTED_RATING],
 ['compact','compact','combined',COMPACT_RATING]]){
 const study='linear-support-r3',root='research/perception/results/'+study+'/'+model,selection=read(root+'/selection.json');
 if(selection.usesOuterTruth!==false)throw Error('INNER-only selection required');
 const c=selection.common[arm];
 profiles[name]={schemaVersion:1,kind:'anchored',classes,components:false,supportPower:c.representation.power,
  recipe:weights[c.representation.weightId],family:c.family,policy:c.policy,modelIdentity:identity,
  provenance:{study,model,arm,selection:'pooled INNER; first-degree affinity restriction; retrospective source-song groups',baseline:'v1.2.2'}};
}
fs.writeFileSync('src/models/perception7k.json',JSON.stringify({schemaVersion:1,profiles},null,2)+'\n');
console.log('Exported action-only, Detailed and Compact perception profiles');
