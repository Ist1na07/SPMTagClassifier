'use strict';
const fs=require('node:fs'),{parseOsu,prepare}=require('../../src/parser.cjs'),{jackCadence,switchCadence}=require('../../src/action-clock.cjs');
function context(events,rate=1,mirror=false,phase=0){
 const text='[General]\nMode:3\n[Difficulty]\nCircleSize:7\n[TimingPoints]\n0,500,4,1,0,100,1,0\n[HitObjects]\n'+events.flatMap(e=>e.cols.map(c=>`${Math.floor(((mirror?6-c:c)+.5)*512/7)},192,${Math.round(1000+phase+e.t)},1,0,0:0:0:0:`)).join('\n');
 return prepare(parseOsu(text,{speedRate:rate}),10/rate);
}
const series=(step,count,masks,offset=0)=>Array.from({length:count},(_,i)=>({t:offset+i*step,cols:masks[i%masks.length]}));
const cases=[];const add=(id,tag,events,bpm)=>cases.push({id,tag,events,bpm});
for(const spacing of [50,75,100,125,150,200]){
 const bpm=15000/spacing,main=series(spacing,80,[[0,1,4,5],[1,2,5,6]]);
 add('simultaneous-'+spacing,'chordjack',main,bpm);
 add('alternating-hands-'+spacing,'chordjack',series(spacing,80,[[0,1],[5,6]]),bpm/2);
 add('asynchronous-hands-'+spacing,'chordjack',[...series(spacing,80,[[0,1],[1,2]]),...series(spacing,80,[[4,5],[5,6]],spacing/2)].sort((a,b)=>a.t-b.t),bpm);
 add('single-fillers-'+spacing,'chordjack',[...main,...series(spacing,79,[[0],[4],[1],[5],[2],[6]],spacing/2)].sort((a,b)=>a.t-b.t),bpm);
 add('missing-pulses-'+spacing,'chordjack',main.filter((_,i)=>i%7!==3),bpm);
 add('single-finger-'+spacing,'minijack',series(spacing,80,[[0]]),bpm);
 add('cross-hand-interleaving-'+spacing,'minijack',[...series(spacing,80,[[0]]),...series(spacing,80,[[4],[5],[6]],spacing/2)].sort((a,b)=>a.t-b.t),bpm);
 add('short-repeated-bursts-'+spacing,'minijack',Array.from({length:30},(_,i)=>[{t:i*spacing*4,cols:[i%3]},{t:i*spacing*4+spacing,cols:[i%3]}]).flat(),bpm);
}
add('one-hand-roll-not-jack','minijack',series(50,90,[[0],[1],[2]]),null);
add('single-roll-not-coactive-strikes','chordjack',series(50,90,[[0],[1],[2]]),null);
let t=0;add('irregular-no-period','chordjack',[0,83,106,171,229,310,449,157,191,337,277].map(dt=>({t:t+=dt,cols:[0,1]})),null);
add('equal-two-periods-no-point-clock','chordjack',[...series(100,10,[[0,1]]),...series(200,10,[[0,1,2]],1233)],null);
add('common-finger-anchor-plus-chord-backbone','chordjack',series(50,80,[[0,1],[0]]),150);
add('sparse-chord-accents-inside-roll-not-slower-jack','chordjack',Array.from({length:120},(_,i)=>({t:i*50,cols:i%4===0?[0,1]:[i%3]})),null);
add('short-fast-hand-not-sustained-tempo','chordjack',[...series(50,4,[[0,1]]),...series(200,80,[[4,5]])].sort((a,b)=>a.t-b.t),75);
for(const spacing of [50,75,100,125,150,200]){
 add('switch-stream-'+spacing,'dense chordstream',series(spacing,80,[[0,1],[4,5],[1,2],[5,6]]),15000/spacing);
 add('fast-switch-stream-'+spacing,'fast chordstream',series(spacing,80,[[0,1],[4,5],[1,2],[5,6]]),15000/spacing);
}
add('fixed-grips-no-switch-clock','dense chordstream',series(100,80,[[0,1],[4,5]]),null);
const configs=[0,1,2].flatMap(widthPower=>[.4,.5,.6,.7,.8].flatMap(minimumShare=>[0,.25,.5,.75].map(handSupportRatio=>({widthPower,minimumShare,handSupportRatio})))),grid=[];
for(const config of configs){let count=0,correct=0,resolved=0,known=0;const errors=[];
 for(const c of cases)for(const rate of [.75,1,1.5])for(const mirror of [false,true])for(const phase of [0,17]){
  const ctx=context(c.events,rate,mirror,phase),m=c.tag.endsWith('chordstream')?switchCadence(ctx,ctx.map.firstNote,ctx.map.lastNote+1/rate,rate,config):jackCadence(ctx,ctx.map.firstNote,ctx.map.lastNote+1/rate,c.tag,rate,config);
  const expected=c.bpm===null?null:Math.round(c.bpm*rate),got=m?.equivBpm??null;
  // One BPM is a source-integer-timestamp rounding uncertainty, not an octave error.
  const ok=expected===null?got===null:got!==null&&Math.abs(got-expected)<=1;count++;correct+=ok;known+=expected!==null;resolved+=expected!==null&&got!==null;
  if(!ok)errors.push({id:c.id,rate,mirror,phase,expected,got});
 }
 grid.push({config,count,correct,known,resolved,accuracy:correct/count,errors});console.log(config,correct+'/'+count,errors.slice(0,2));
}
fs.writeFileSync('research/action_clock/results/clock-grid.json',JSON.stringify({scope:'Mechanistic synthetic action-clock constraints, NOT player BPM accuracy; timestamps/hand roles define expected observable rates. No exemplar used for parameter selection.',cases:cases.map(({events,...c})=>c),grid},null,2));
module.exports={context,series,cases};
