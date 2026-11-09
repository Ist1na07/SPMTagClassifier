'use strict';
const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const {records7}=require('../rework/data.cjs'),{compileRestEngine}=require('./compile-rest-engine.cjs');
const api=require('../..'),fixture=JSON.parse(fs.readFileSync('.cache/overlay-native-expected-r1.json','utf8'));
const output=__dirname+'/overlay-native-audit-r1.json';
if(fs.existsSync(output))throw Error('Preserve completed native audit');
const files=new Map(records7().map(r=>[r.uid,r.path])),texts=new Map(),engines=new Map();
const root=path.resolve('.cache/rest-native-candidates-r1');fs.mkdirSync(root,{recursive:true});
const b={};vm.runInNewContext(fs.readFileSync('dist/tag-engine.js','utf8'),b);
function text(uid){if(!texts.has(uid))texts.set(uid,fs.readFileSync(files.get(uid),'utf8'));return texts.get(uid);}
function core(config,policy){
  const key=JSON.stringify([config,policy]);if(engines.has(key))return engines.get(key);
  const file=path.join(root,'profile-'+engines.size+'.cjs');fs.writeFileSync(file,compileRestEngine(config,policy));
  const value=require(file);engines.set(key,value);return value;
}
function close(a,b,where){assert.ok(Math.abs(a-b)<1e-9,where+': '+a+' != '+b);}
function compare(actual,expected,{publicEnvelope=false,map}={}){
  assert.equal(actual.length,expected.length);
  for(let j=0;j<actual.length;j++){
    const a=actual[j],e=expected[j];
    assert.equal(a.start,publicEnvelope&&j===0?map.firstNote:e.start);
    assert.equal(a.end,publicEnvelope&&j===actual.length-1?map.lastNote:e.end);
    for(const k of ['tag','family'])assert.equal(a[k],e[k]);
    assert.equal(a.secondary??null,e.secondary??null);close(a.confidence,e.confidence,'confidence');
    const aa=a.scores??{[a.tag]:1},ee=e.scores??{[e.tag]:1};assert.deepEqual(Object.keys(aa).sort(),Object.keys(ee).sort());
    for(const k of Object.keys(ee))close(aa[k],ee[k],'score '+k);
  }
}
function overall(actual,expected){
  assert.deepEqual(Array.from(actual.tags),expected.tags);assert.equal(actual.sort,expected.sort);
  assert.deepEqual(Object.keys(actual.weights).sort(),Object.keys(expected.weights).sort());
  for(const k of Object.keys(expected.weights))close(actual.weights[k],expected.weights[k],'weight '+k);
}
let outer=0,final=0,manual=0;
for(const c of fixture.cases){
  const t=text(c.uid);
  if(c.phase==='final'){
    const n=api.analyzeMap(t),s=b.TagClassifier.analyzeMap(t);
    assert.equal(JSON.stringify(n),JSON.stringify(s),c.uid+' public Node/browser');
    compare(n.segments,c.segments,{publicEnvelope:true,map:n.map});overall(n.overall,c.overall);final++;
  }else{
    const n=core(c.config,c.policy).analyzeMap(t);compare(n.segments,c.segments);overall(n.overall,c.overall);
    c.phase==='outer'?outer++:manual++;
  }
}
const queries=[];
for(const q of fixture.queries){
  const t=fs.readFileSync(`artifacts/frontier-annotation-r5/${q.uid}.osu`,'utf8');
  compare(core(q.config,q.policy).analyzeMap(t).segments,q.segments);
  const n=api.analyzeMap(t),s=b.TagClassifier.analyzeMap(t);assert.equal(JSON.stringify(n),JSON.stringify(s));
  const [a,z]=q.targetMs,accuracy=n.segments.reduce((sum,f)=>sum+(f.tag==='speed'?Math.max(0,Math.min(z,f.end)-Math.max(a,f.start)):0),0)/(z-a);
  queries.push({uid:q.uid,sourceHeldOutPrimaryAccuracy:q.expectedPrimaryAccuracy,finalRuntimePrimaryAccuracy:accuracy});
  assert.equal(accuracy,1);
}
fs.writeFileSync(output,JSON.stringify({checked:new Date().toISOString(),outerWholeCharts:outer,outerManualCharts:manual,
  finalPublicNodeAndStaticBrowserCharts:final,queries,matchedPythonDecisionsAndScores:true,
  finalFitIsNotHeldOutAccuracy:true,liveGameOrTosuVerified:false},null,2)+'\n');
console.log({outer,manual,final,queries});
