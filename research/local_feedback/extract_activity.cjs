'use strict';
// Local route evidence is measured locally, not averaged over a long old slot.
const fs=require('node:fs');
const {prepare,lowerBound:lb}=require('../../src/parser.cjs');
const {features}=require('../../src/features.cjs');
const p=require('../../integration/rating-parameters.json').detail;
const {GEOMETRY}=(()=>({GEOMETRY:['npsPerKey','rowRate','chordFraction','multiFraction','lnHeadFraction','overlap','repeatFraction','cutReturn','chordChange','columnEntropy','handImbalance','sameHand','crossCut','logIoiSd','ioiChange','graceFraction','gridResidual','ioiCv','occupancy','minOccupancy','occupancySd','heldAtHead','lockFraction','releaseIndependence','releaseScatter','tailRate','shortHoldFraction','inverseGap','lnHeadOverlap','anchorShare','repeatRun','maskEntropy','trillFraction','rollFraction','burstiness','activeFraction']}))();
const GUIDES=['releasePairRate','guidedReleasePairRate','independentTailRate'];
const WIDTHS=[1000,2000,4000],output='.cache/local-feedback-activity-r6.jsonl';
if(fs.existsSync(output))throw Error('Preserve completed local activity input');
function guideEvents(ctx){
  const od=Number.isFinite(ctx.map.meta.od)?ctx.map.meta.od:6;
  const q=(64.5-Math.ceil(3*od))/500,x=.3*Math.sqrt(Math.max(q,1e-6)),precision=Math.min(x,.6*(x-.09)+.09)**p.od_mult_r;
  const tails=ctx.holds.flatMap((hs,col)=>hs.map(([head,t])=>({head,t,col}))).sort((a,b)=>a.t-b.t||a.col-b.col),rows=[];
  for(let j=0;j<tails.length;){
    const time=tails[j].t,group=[];while(j<tails.length&&tails[j].t===time)group.push(tails[j++]);
    const released=new Set(group.map(g=>g.col));let retained=0;
    for(let c=0;c<ctx.map.keys;c++)if(!released.has(c)){const hs=ctx.holds[c],i=lb(hs,time,h=>h[0])-1;if(i>=0&&hs[i][1]>time)retained++;}
    let guided=0,independent=0;
    for(const t of group){
      const col=ctx.cols[t.col],own=col[lb(col,time+1e-8,n=>n.t)],any=ctx.map.notes[lb(ctx.map.notes,time+1e-8,n=>n.t)],next=own??any;
      const ih=.001*Math.abs(time-t.head-80)/precision,it=next?.t!==undefined?.001*Math.abs(next.t-time-80)/precision:10;
      const I=2/(2+Math.exp(-p.r_I_steep*(ih-p.r_I_off))+Math.exp(-p.r_I_steep*(it-p.r_I_off)));
      independent+=I;guided+=I*retained;
    }
    rows.push({t:time,values:[group.length*retained,guided,independent]});
  }
  const prefix=[GUIDES.map(()=>0)];for(const r of rows)prefix.push(prefix.at(-1).map((v,j)=>v+r.values[j]));
  return{rows,prefix};
}
let count=0,nframes=0;
for(const line of fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n')){
  const row=JSON.parse(line),ctx=prepare(row.map),guide=guideEvents(ctx),frames=[],values=WIDTHS.map(()=>[]);
  for(let parent=0;parent<row.frames.length;parent++){
    const old=row.frames[parent];
    for(let start=old.start;start<old.end;start+=500){
      const end=Math.min(start+500,old.end),center=(start+end)/2;frames.push({...old,start,end,parent});
      WIDTHS.forEach((width,w)=>{
        const a=Math.max(row.map.firstNote,center-width/2),b=Math.min(row.map.lastNote,center+width/2),m=features(ctx,a,Math.max(a+1,b)).metrics;
        const lo=lb(guide.rows,a,r=>r.t),hi=lb(guide.rows,b,r=>r.t),v=GUIDES.map((_,j)=>(guide.prefix[hi][j]-guide.prefix[lo][j])/ctx.map.keys/((b-a)/1000));
        values[w].push([...GEOMETRY.map(k=>m[k]),...v]);
      });
    }
  }
  fs.appendFileSync(output,JSON.stringify({uid:row.uid,frames,values})+'\n');nframes+=frames.length;
  if(++count%75===0)console.log('local activity',count,nframes);
}
fs.writeFileSync(output+'.schema.json',JSON.stringify({records:count,frames:nframes,widths:WIDTHS,featureKeys:[...GEOMETRY,...GUIDES],
 guide:'Union hold episodes. Same SPM v1.2.0 release-independence factor (OD,80ms lag,strictly future own-column head then global fallback), multiplied by retained-finger pairs; raw pair and total physical action features remain available. Not Rbar or an isolated Release probability.',
 source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
console.log('complete local activity',count,nframes);
