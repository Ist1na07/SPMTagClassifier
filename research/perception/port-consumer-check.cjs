'use strict';
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),api=require('../../src/index.cjs');
const {patchRating}=require('../../integration/patch-overlay.cjs'),profiles=require('../../src/models/perception7k.json').profiles;
const reference=JSON.parse(fs.readFileSync('.cache/perception-overlay-port-reference.json','utf8'));
const box={};vm.runInNewContext(patchRating(fs.readFileSync('.upstream/spm_algorithm.js','utf8')),box);
const result=[];
for(const c of reference.cases)for(const model of ['detail','compact']){
 const text=fs.readFileSync(c.source,'utf8'),port=box.SPMEngine.processMap(text,'full',1,{model,includeTagEvidence:true}),r=c.models[model];
 const expectedEvidence={schemaVersion:1,keyCount:7,speedRate:1,clock:'post-rate-ms',modelIdentity:profiles[model].modelIdentity,query:r.query,demand:r.demand};
 const a=api.analyzeMap(text,{ratingEvidence:port.tagEvidence}),b=api.analyzeMap(text,{ratingEvidence:expectedEvidence});
 assert.deepEqual([...a.overall.tags].sort(),[...b.overall.tags].sort(),c.uid+'/'+model);
 let max=0;for(const k of Object.keys(a.overall.percentages))max=Math.max(max,Math.abs(a.overall.percentages[k]-b.overall.percentages[k]));
 assert(max<1e-4,c.uid+'/'+model+' percentage difference '+max);result.push({uid:c.uid,model,tags:a.overall.tags,maxPercentagePointError:max});
}
fs.writeFileSync('research/perception/results/final-runtime/port-consumer.json',JSON.stringify({cases:result.length,source:'actual patched upstream scorer vs cached released Python raw curves; no scoring-formula edits',results:result},null,2)+'\n');
console.log('Actual scorer → final tag consumer:',result.length,'cases');
