'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs');
const {createReadoutProbe}=require('./cadence_readout.cjs');
const {apply}=require('./static_cut_runtime.cjs');
const product=require('../../src/index.cjs');
test('native hand certificate removes H01/H03 false CS, leaves H02/H04 cut support untouched',()=>{
  const analyze=createReadoutProbe({},apply),receipts=[];
  for(const id of ['H01','H02','H03','H04']){
    const text=fs.readFileSync('artifacts/frontier-annotation-r7/'+id+'.osu','utf8'),r=analyze(text),old=product.analyzeMap(text),target=r.debug.rawSegments.filter(s=>s.start>=9000&&s.end<=17000),yes=id==='H01'||id==='H03';
    assert(target.length);if(yes)assert(target.every(s=>!['dense chordstream','fast chordstream'].includes(s.tag)),id);
    else assert(r.debug.activity.certificates.every(c=>!c),id);
    receipts.push({id,before:old.segments.map(s=>({start:s.start,end:s.end,tag:s.tag,bpm:s.equivBpm})),after:r.segments.map(s=>({start:s.start,end:s.end,tag:s.tag,bpm:s.equivBpm})),overall:r.overall.tags});
  }
  fs.writeFileSync('research/local_feedback/results/static-cut-r1/native-h.json',JSON.stringify({scope:'Known player-labelled hand motion cases, parameter-free negative correction on fixed v1.2.3; not a random validation score',cases:receipts},null,2));
});
test('cut certificate never inflates any other class support',()=>{
  const classes=['dense chordstream','fast chordstream','speed','tech'],s={start:0,end:1000,family:'rc',tag:'dense chordstream',scores:{'dense chordstream':2,'fast chordstream':.8,speed:.7,tech:.6}};
  const notes=[];for(let t=0;t<=1000;t+=100)for(const col of(t%200?[4,5]:[0,1]))notes.push({t,col,lnEnd:null});
  const m={keys:7,notes,firstNote:0,lastNote:1000,meta:{od:8},timingPoints:[],breaks:[]},r=apply({},m,[s],[],classes);
  assert(r.certificates.every(Boolean));assert(r.support.every(q=>q[0]===0&&q[1]===0&&q[2]===.35&&q[3]===.3));
});
