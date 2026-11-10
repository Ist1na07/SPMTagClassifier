'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),os=require('node:os'),vm=require('node:vm');
const patch=require('../integration/patch-overlay.cjs');
test('unknown overlay layouts and changed rating parameters are rejected',()=>{
 assert.throws(()=>patch.patchHtml('<html></html>'),/Unsupported/);
 assert.throws(()=>patch.patchRating('not a supported scorer'),/Unsupported/);
});
test('upstream update is dry-run first, backed up, idempotent and preserves difficulty outputs',{skip:!fs.existsSync('.upstream/index.html')},()=>{
 const directory=fs.mkdtempSync(path.join(os.tmpdir(),'spm-perception-test-'));
 // Keep the small owned test directory as a recovery/debug artifact; no recursive deletion.
 for(const f of ['index.html','spm_algorithm.js','tag_engine.js'])fs.copyFileSync('.upstream/'+f,path.join(directory,f));
 const before=fs.readFileSync(path.join(directory,'index.html'),'utf8'),p=patch.plan(directory);
 assert.equal(fs.readFileSync(path.join(directory,'index.html'),'utf8'),before);assert.equal(p.changed.length,3);
 const receipt=patch.apply(p);assert.equal(fs.readFileSync(path.join(receipt.backup,'index.html'),'utf8'),before);
 assert.equal(patch.plan(directory).changed.length,0);
 // The misnumbered v1.3.0 integration upgrades in place without double-patching.
 const oldMarker='SPMTagClassifier perception integration v1.3.0',marker='SPMTagClassifier perception integration schema 1';
 assert.equal(patch.patchHtml(p.updated['index.html'].replace(marker,oldMarker)),p.updated['index.html']);
 assert.equal(patch.patchRating(p.updated['spm_algorithm.js'].replace(marker,oldMarker)),p.updated['spm_algorithm.js']);
 // Already installed schema1 pages also receive the missing-branch fix.
 const oldBranch=`                    const branch = seg.branches ? seg.branches[seg.tag] || 0 : seg.confidence;
                    elSegConf.textContent = Math.round(branch * 100) + '%';
                    elSegConf.title = seg.branches ? '分支贴合度（不是识别正确率）' : '原分类支持';`;
 const currentBranch=p.updated['index.html'].match(/                    const hasBranch =[^\n]+\n[^\n]+\n[^\n]+\n[^\n]+/)[0];
 assert.equal(patch.patchHtml(p.updated['index.html'].replace(currentBranch,oldBranch)),p.updated['index.html']);
 assert.throws(()=>patch.patchRating(p.originals['spm_algorithm.js'].replace('w_jm:934.2448253030639','w_jm:1')),/parameters/);
 const text='[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n'+Array.from({length:100},(_,i)=>`${(i%7+.5)*512/7},192,${1001+i*83},1,0,0:0:0:0:`).join('\n');
 const old={},updated={};vm.runInNewContext(p.originals['spm_algorithm.js'],old);vm.runInNewContext(p.updated['spm_algorithm.js'],updated);
 for(const model of ['detail','compact'])for(const speedRate of [.75,1,1.5]){
  const a=old.SPMEngine.processMap(text,'full',speedRate,{model}),b=updated.SPMEngine.processMap(text,'full',speedRate,{model});
  assert.equal(JSON.stringify(a),JSON.stringify(b));
  const rated=updated.SPMEngine.processMap(text,'full',speedRate,{model,includeTagEvidence:true}),{tagEvidence,...rest}=rated;
  assert.equal(JSON.stringify(rest),JSON.stringify(a));
  const r=require('..').analyzeMap(text,{speedRate,ratingEvidence:tagEvidence});assert.equal(r.modelInfo.perception.profile,model);
 }
 // A stale plan must not overwrite a later human edit.
 const next=patch.plan(directory);next.changed=['index.html'];fs.appendFileSync(path.join(directory,'index.html'),'\n<!-- human edit -->');
 assert.throws(()=>patch.apply(next),/changed since planning/);
});
