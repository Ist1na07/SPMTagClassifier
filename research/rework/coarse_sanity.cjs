'use strict';
// Definition-level synthetic counterexamples complement weak-bucket scores.
// These are not a human-annotated accuracy dataset and do not tune parameters.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const runtime=require('./coarse-runtime.cjs');
function text(K,objects){return `[General]\nMode:3\n[Difficulty]\nCircleSize:${K}\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n${objects.join('\n')}`;}
function note(K,c,t,tail=null){return `${Math.floor((c+.5)*512/K)},192,${t},${tail===null?1:128},0,${tail===null?'0:0:0:0:':tail+':0:0:0:0:'}`;}
function fixtures(){
 const cases=[];
 for(const K of[4,6,7]){
  for(const bpm of[60,120,180])for(let width=1;width<=K;width++){
   const layouts=[[...Array(width)].map((_,c)=>c),[...Array(width)].map((_,c)=>Math.floor((c+.5)*K/width))];
   for(const columns of [...new Map(layouts.map(cs=>[cs.join(','),cs])).values()]){
    const objects=[];for(let i=0;i<97;i++)for(const c of columns)objects.push(note(K,c,i*15000/bpm));
    cases.push({id:`K${K}-repeat-${columns.join('')}-${bpm}`,K,bpm,columns,expected:'jack',kind:'strictly periodic repeated identical RC chord',text:text(K,objects)});
   }
  }
  for(const width of[1,K]){
   const columns=Array.from({length:width},(_,c)=>c),objects=columns.map(c=>note(K,c,0,10000));
   cases.push({id:`K${K}-held-${width}`,K,columns,expected:'ln',kind:'only long holds, including headless held windows',text:text(K,objects)});
  }
 }
 return cases;
}
function audit(profile,options={}){
 const rows=[],full={};
 for(const f of fixtures()){
  const result=runtime.classifyCoarse(f.text,profile,options);assert.equal(result.modelInfo.nativeKeyCount,f.K);
  assert.ok(Object.values(result.coarse.composition).every(Number.isFinite));
  rows.push({id:f.id,keys:f.K,columns:f.columns,bpm:f.bpm,kind:f.kind,expectedByConstruction:f.expected,
   predicted:result.coarse.dominant,matches:result.coarse.dominant===f.expected,composition:result.coarse.composition,
   clippedFeatureFraction:result.modelInfo.clippedFeatureFraction});
  if(f.expected==='jack'&&f.columns.length===f.K&&f.bpm===120)full[f.K]=result;
 }
 const reference=full[4],model=profile.model||profile;let featureError=0,probabilityError=0;
 for(const K of[6,7]){
  assert.equal(reference.windows.length,full[K].windows.length);
  for(let i=0;i<reference.windows.length;i++){
   for(const key of model.keys)featureError=Math.max(featureError,Math.abs(reference.windows[i].metrics[key]-full[K].windows[i].metrics[key]));
   reference.windows[i].probability.forEach((p,j)=>probabilityError=Math.max(probabilityError,Math.abs(p-full[K].windows[i].probability[j])));
  }
 }
 assert.ok(featureError<1e-9&&probabilityError<1e-9,'Uniform full-chord native-key normalized features must not gain virtual columns');
 return {scope:'synthetic definition-level counterexamples and native-key equivalence; not human accuracy, no parameter selection',
  profile:model,physicalSupport:options.physicalSupport??profile.postprocess?.physicalSupport??'none',tests:rows.length,counterexamples:rows.filter(r=>!r.matches),rows,
  fullChordCrossKey:{featureError,probabilityError,kind:'same per-finger periodic full-width Jack at each actual key count'}};
}
if(require.main===module){
 const [profilePath,out,mode='none']=process.argv.slice(2);if(!profilePath||!out)throw Error('Usage: node coarse_sanity.cjs <profile.json> <new-output.json> [none|jack|ln|both]');
 const value=audit(JSON.parse(fs.readFileSync(profilePath,'utf8')),{physicalSupport:mode});
 fs.mkdirSync(path.dirname(out),{recursive:true});fs.writeFileSync(out,JSON.stringify({...value,profilePath,source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
 console.log(profilePath,'synthetic cases',value.tests,'counterexamples',value.counterexamples.length,'native-key equivalence',value.fullChordCrossKey);
}
module.exports={fixtures,audit};
