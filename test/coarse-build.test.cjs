const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const research=require('../research/rework/coarse-runtime.cjs');
const {fixtures}=require('../research/rework/coarse_sanity.cjs');
test('static coarse build runs without Node globals and matches the actual research consumer',()=>{
 const box={module:{exports:{}}};vm.runInNewContext(fs.readFileSync('src/generated/coarse-engine.cjs','utf8'),box);
 const built=box.module.exports;
 const model={keys:['rowRate','occupancy'],classes:['jack','chordstream','speed','tech','ln'],mean:[0,0],scale:[1,1],
  theta:[.1,0,.2,0,.3,0,.4,0,0,1,0,0,0,0,0,.5,.5],clip:6,
  config:{windowMs:4000,burdenGate:.2},spec:{representation:'physical'}};
 const profile={model,postprocess:{physicalSupport:'both'}};
 for(const f of fixtures()){
  const a=research.classifyCoarse(f.text,profile),b=built.classifyCoarse(f.text,profile);
  assert.equal(JSON.stringify(a),JSON.stringify(b),f.id);
  assert.equal(b.coarse.dominant,f.expected,f.id);
 }
});
