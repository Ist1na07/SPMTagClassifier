'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {extract,summarise,CHANNELS}=require('./mechanics-probe.cjs');
test('mechanism summaries integrate duration, not row count',()=>{
 const s=summarise([0,1,1000],[10,1,0],0,1000);
 assert.equal(s[0],1.009);assert.equal(s[1],1);assert.equal(s[2],10);
 assert.equal(summarise([0,1,1000],[-10,-1,0],0,1000)[2],-1);
});
test('release representation is zero for RC and mirror-invariant for the tested LN geometry',
 {skip:!fs.existsSync('.upstream/spm_algorithm.js')},()=>{
 const header='[General]\nMode:3\n[Difficulty]\nCircleSize:7\nOverallDifficulty:6\n[TimingPoints]\n0,500,4,2,0,100,1,0\n[HitObjects]\n';
 const notes=[[0,0,500],[125,2,725],[300,5,900],[600,1],[800,4],[1000,2,1500],[1300,6],[1700,0]];
 const text=(tap,mirror)=>header+notes.map(([t,c,e])=>`${(mirror?6-c:c)*512/7+32},192,${t},${e&&!tap?128:1},0,${e&&!tap?e+':0:0:0:0':'0:0:0:0:'}`).join('\n');
 const intervals=[{start:0,end:1700}],j=CHANNELS.indexOf('Rbar')*3;
 const rice=extract(text(true,false),intervals)[0],ln=extract(text(false,false),intervals)[0],mirror=extract(text(false,true),intervals)[0];
 for(let k=j;k<j+3;k++){assert.equal(rice[k],0);assert.ok(Math.abs(ln[k]-mirror[k])<1e-10);}
 assert.ok(ln[j]>0);
});
