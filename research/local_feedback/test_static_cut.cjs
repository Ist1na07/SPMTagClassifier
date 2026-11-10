'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const guard=require('./static_cut_guard.cjs');
test('real user H pairs separate hand alternation from genuine cut',()=>{
  for(const id of ['H01','H02','H03','H04']){
    const text=fs.readFileSync('artifacts/frontier-annotation-r7/'+id+'.osu','utf8'),ctx=prepare(parseOsu(text));
    assert.equal(guard.certifiedStaticRice(guard.prepare(ctx),9000,17000),id==='H01'||id==='H03',id);
  }
});
test('rolls and actual LN actions are not declared static Rice',()=>{
  for(const [round,id]of [[2,'C16'],[2,'C24'],[3,'LN-H'],[3,'LN-L']]){
    const text=fs.readFileSync('artifacts/frontier-annotation-r'+round+'/'+id+'.osu','utf8'),ctx=prepare(parseOsu(text));
    assert.equal(guard.certifiedStaticRice(guard.prepare(ctx),9000,17000),false,id);
  }
});
test('no certificate with fewer than four observed head rows',()=>{
  const notes=[{t:0,col:0,lnEnd:null},{t:100,col:4,lnEnd:null},{t:200,col:0,lnEnd:null}],map={keys:7,notes,meta:{od:8},firstNote:0,lastNote:200,timingPoints:[],breaks:[]};
  assert.equal(guard.certifiedStaticRice(guard.prepare(prepare(map)),0,201),false);
});
