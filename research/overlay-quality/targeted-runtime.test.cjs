const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const {refine}=require('./targeted-runtime.cjs'),{compileTargetedEngine}=require('./compile-targeted-engine.cjs');
const {records7}=require('../rework/data.cjs'),{parseOsu,prepare}=require('../../src/parser.cjs');
const fixture=JSON.parse(fs.readFileSync(__dirname+'/targeted-sequence-r1/runtime-resource-fixture.json','utf8'));
const records=new Map(records7().map(r=>[r.uid,r.path]));
function text(uid){return fs.readFileSync(fixture.rows.find(r=>r.uid===uid).source,'utf8');}
function close(a,b){assert.ok(Math.abs(a-b)<1e-8,`${a} != ${b}`);}
test('zero refinement keeps exact native parents, including optional fields',()=>{
  for(const r of fixture.rows){const p={...fixture.profile,model:{...fixture.profile.model,theta:fixture.profile.model.theta.map(()=>0)}};
    assert.equal(refine(r.parents,prepare(parseOsu(text(r.uid))),p),r.parents);}
});
test('actual static consumer matches Python resource candidate and unchanged expert path',()=>{
  const built=compileTargetedEngine(fixture.profile),c={};vm.runInNewContext('var Candidate='+built.code,c);
  assert.equal(/\brequire\s*\(/.test(built.code),false);
  for(const r of fixture.rows){
    const exact=r.scope==='exact-target-interval';
    const result=exact?{segments:refine(r.parents,prepare(parseOsu(text(r.uid))),fixture.profile)}:c.Candidate.analyzeMap(text(r.uid));
    const expected=exact?r.rawExpected:r.expected;assert.equal(result.segments.length,expected.length,r.uid);
    for(let i=0;i<expected.length;i++){
      const a=result.segments[i],b=expected[i];close(a.start,b.start);close(a.end,b.end);assert.equal(a.tag,b.tag,r.uid);assert.equal(a.secondary??null,b.secondary??null,r.uid);close(a.confidence,b.confidence);
      for(const[k,v]of Object.entries(b.scores??{[b.tag]:1}))close((a.scores??{[a.tag]:1})[k],v);
    }
    if(!exact){assert.equal(JSON.stringify(result.overall.tags),JSON.stringify(r.overall.tags));
      for(const[k,v]of Object.entries(r.overall.weights))close(result.overall.weights[k],v);}
  }
});
test('full excerpt consumer preserves both real Speed targets, not just their forced intervals',()=>{
  const built=compileTargetedEngine(fixture.profile),c={};vm.runInNewContext('var Candidate='+built.code,c);
  for(const uid of ['local/F01','local/F02']){
    const result=c.Candidate.analyzeMap(fs.readFileSync('artifacts/frontier-annotation-r5/'+uid.split('/')[1]+'.osu','utf8'));
    const correct=result.segments.reduce((sum,s)=>sum+(s.tag==='speed'?Math.max(0,Math.min(16500,s.end)-Math.max(9000,s.start)):0),0);
    close(correct,7500);
  }
});
