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
 let c=selection.common[arm],recipe=weights[c.representation.weightId],selectedStudy=study;
 if(name==='detail'){
  const review=read('research/perception/results/selection-review-r4/selected-detail.json');
  c=review.profile;recipe=review.recipe;selectedStudy='anchored-r1';
 }
 profiles[name]={schemaVersion:1,kind:'anchored',classes,components:false,supportPower:c.representation.power,
  recipe,family:c.family,policy:c.policy,modelIdentity:identity,
  provenance:{study:selectedStudy,model,arm,parameterSelection:'pooled INNER; retrospective source-song groups',
    familySelection:name==='detail'?'R1 restored after retrospective R1/R3 review':'first-degree affinity restriction',
    researchBaseline:'v1.1.0',frozenIntermediatePrior:'v1.2.2'}};
}
fs.writeFileSync('src/models/perception7k.json',JSON.stringify({schemaVersion:1,profiles},null,2)+'\n');
console.log('Exported action-only, Detailed and Compact perception profiles');
