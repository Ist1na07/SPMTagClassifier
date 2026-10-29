const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {analyzeMap}=require('../src/index.cjs');const checked=[];let maxError=0;
for(const k of [4,7]) {
 const profile=JSON.parse(fs.readFileSync(`src/models/${k}k.json`,'utf8'));
 const predictions=JSON.parse(fs.readFileSync(`research/results/${k}k-test-predictions.json`,'utf8')).filter(x=>x.variant===profile.variant);
 const counts={};
 for(const r of predictions){const tag=r.truth[0];if((counts[tag]??0)>=2)continue;counts[tag]=(counts[tag]??0)+1;
  const [folder,id]=r.uid.split('/'),file=path.join('dataset',folder,'beatmaps',id+'.osu');
  const a=analyzeMap(fs.readFileSync(file,'utf8'),{algorithm:'modern'});
  assert.deepEqual(a.overall.tags.sort(),r.predicted.sort(),r.uid);
  for(const t of Object.keys(r.scores)){const e=Math.abs(a.overall.scores[t]-r.scores[t]);maxError=Math.max(maxError,e);assert.ok(e<1e-4,`${r.uid}: ${t} ${e}`);}
  checked.push(r.uid);
 }
}
fs.writeFileSync('research/results/runtime-parity.json',JSON.stringify({checked,maxAbsoluteSupportError:maxError,tolerance:1e-4},null,2));
console.log('Python-to-JS fitted-model parity',checked.length,'charts; max error',maxError);
