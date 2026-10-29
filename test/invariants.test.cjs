const test=require('node:test'),assert=require('node:assert/strict');
const {parseOsu,prepare}=require('../src/parser.cjs');
const {features}=require('../src/features.cjs');
function chart(k){return '[General]\nMode:3\n[Difficulty]\nCircleSize:'+k+'\n[TimingPoints]\n0,500,4,2,1,100,1,0\n[HitObjects]\n'+
 Array.from({length:80},(_,i)=>`${(i%k+.5)*512/k},192,${1000+i*125},${i%3===0?128:1},0,${i%3===0?1000+i*125+200+':0:0:0:0:':'0:0:0:0:'}`).join('\n');}
for(const k of [4,6,7]){
 test(k+'K mirror preserves the chosen geometric and rhythm measurements',()=>{
  const m=parseOsu(chart(k)),mirrored={...m,notes:m.notes.map(n=>({...n,col:k-1-n.col})).sort((a,b)=>a.t-b.t||a.col-b.col)};
  const a=features(prepare(m),1000,10000).metrics,b=features(prepare(mirrored),1000,10000).metrics;
  for(const key of Object.keys(a))assert.ok(Math.abs(a[key]-b[key])<1e-10,key+': '+a[key]+' / '+b[key]);
 });
 test(k+'K rate scaling preserves morphology and scales equivalent BPM',()=>{
  const text=chart(k),a=features(prepare(parseOsu(text)),1000,10000).metrics;
  const b=features(prepare(parseOsu(text,{speedRate:1.5})),1000/1.5,10000/1.5).metrics;
  for(const key of ['chordFraction','overlap','cutReturn','occupancy','inverseGap','shortHoldFraction','trillFraction'])
   assert.ok(Math.abs(a[key]-b[key])<1e-10,key);
  assert.ok(Math.abs(b.rowBpm/a.rowBpm-1.5)<1e-10);
 });
}
