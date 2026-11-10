'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const api=require('..'),{parseOsu,prepare}=require('../src/parser.cjs');
const {refine,prepareHands,noCut}=require('../src/segment-readout.cjs');
function chart(masks,steps=[[240,100]]){
  const notes=[];let t=1000,k=0;
  for(const [count,step]of steps)for(let i=0;i<count;i++,k++,t+=step)
    for(const col of masks[k%masks.length])notes.push(`${Math.floor((col+.5)*512/7)},192,${Math.round(t)},1,0,0:0:0:0:`);
  return '[General]\nMode:3\n[Difficulty]\nCircleSize:7\nOverallDifficulty:8\n[TimingPoints]\n0,400,4,2,1,50,1,0\n[HitObjects]\n'+notes.join('\n');
}
test('fixed-hand interleaving is not a cut; genuine hand substitutions remain CS',()=>{
  for(const [masks,expected]of [[[[0,1,2],[4,5]],true],[[[0,2,4],[1,5]],false]]){
    const text=chart(masks),ctx=prepare(parseOsu(text));assert.equal(noCut(ctx,prepareHands(ctx),9000,17000),expected);
    const r=api.analyzeMap(text),target=r.segments.filter(s=>s.end>9000&&s.start<17000);
    assert(target.length);assert.equal(target.some(s=>['dense chordstream','fast chordstream'].includes(s.tag)),!expected);
  }
});
test('static-cut exclusion is drop-only and does not promote Vibro to a base class',()=>{
  const ctx=prepare(parseOsu(chart([[0,2,4]]))),raw=[{start:1000,end:24900,family:'rc',tag:'dense chordstream',confidence:1,scores:{'dense chordstream':2,'fast chordstream':.8,speed:.7,tech:.6,wildcard:.18,vibro:1}}];
  const r=refine(ctx,raw,['dense chordstream','fast chordstream','speed','tech','vibro']);
  assert(r.support.every(q=>q[0]===0&&q[1]===0&&q[2]===.35&&q[3]===.3&&q[4]===.5));
  assert(r.frames.every(s=>s.tag!=='vibro'&&s.tag!=='anchor'));assert.equal(raw[0].scores['dense chordstream'],2);
});
test('actual default static engine retains 150→200 and DT225→300 cadence with measured evidence',()=>{
  const text=chart([[0,1,4],[2,5,6]],[[200,100],[267,75]]);
  for(const rate of [1,1.5]){
    const r=api.analyzeMap(text,{speedRate:rate}),at=t=>r.segments.find(s=>s.start<=t/rate&&s.end>t/rate);
    for(const [t,bpm]of [[10000,150*rate],[30000,200*rate]]){
      const s=at(t);assert.equal(s.equivBpm,bpm);assert(s.evidence.cadence);
      assert(s.evidence.cadence.measuredStart>=s.start&&s.evidence.cadence.measuredEnd<=s.end);
    }
  }
});
test('long natural expert boundaries and scores are not erased by finer cadence sampling',()=>{
  const ctx=prepare(parseOsu(chart([[0,2,4],[1,5,6]]))),classes=['speed'];
  const raw=[{start:1000,end:9000,tag:'speed',family:'rc',confidence:.4,scores:{speed:.4}},{start:9000,end:17000,tag:'speed',family:'rc',confidence:.8,scores:{speed:.8}}];
  const r=refine(ctx,raw,classes);assert.deepEqual(r.segments.map(s=>[s.start,s.end,s.confidence]),[[1000,9000,.4],[9000,17000,.8]]);
});
