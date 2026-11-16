'use strict';
const {lowerBound,popcount}=require('./parser.cjs');
const MODEL=require('./models/action-clock.json');

// Observed articulations, NOT a reconstruction of wrist position. Both
// admissible 7K thumb assignments are retained; the busier hand is not
// confused with the union of two alternating hand clocks.
function handViews(ctx,start,end){
  const rows=ctx.rows.slice(lowerBound(ctx.rows,start,r=>r.t),lowerBound(ctx.rows,end,r=>r.t));
  const K=ctx.map?.keys??7,left=(1<<Math.floor(K/2))-1;
  return [0,1].map(thumb=>[0,1].map(side=>{
    const mask=K===7?(side===0?left|(thumb===0?8:0):112|(thumb===1?8:0)):(side===0?left:((1<<K)-1)^left);
    return rows.filter(r=>r.mask&mask).map(r=>({t:r.t,mask:r.mask&mask,width:popcount(r.mask&mask),coactive:popcount(r.mask)>=2}));
  }));
}
function intervalMode(pairs,resolutionMs,{minimumShare=MODEL.minimumShare,widthPower=MODEL.widthPower}={}){
  if(pairs.length<3)return null;
  const sorted=pairs.filter(p=>p.dt>0).map(p=>({...p,w:p.width**widthPower})).sort((a,b)=>a.dt-b.dt);
  let total=sorted.reduce((s,p)=>s+p.w,0),best=null;
  // A cluster spans no more than two source milliseconds: two rounded
  // adjacent event intervals can differ by at most two source milliseconds.
  let hi=0,mass=0;
  for(let lo=0;lo<sorted.length;lo++){
    if(hi<lo){hi=lo;mass=0;}
    while(hi<sorted.length&&sorted[hi].dt-sorted[lo].dt<=2*resolutionMs+1e-7)mass+=sorted[hi++].w;
    if(!best||mass>best.mass+1e-10)best={lo,hi,mass};
    mass-=sorted[lo].w;
  }
  if(!best||best.hi-best.lo<3||best.mass/total<=minimumShare+1e-10)return null;
  const selected=sorted.slice(best.lo,best.hi);
  // Fit the supported interval cluster, rather than inheriting the bias of
  // its most frequent rounded millisecond (e.g.112/113ms for133BPM).
  const spacingMs=selected.reduce((sum,p)=>sum+p.w*p.dt,0)/best.mass;
  return {spacingMs,pairCount:selected.length,totalPairs:sorted.length,intervalShare:best.mass/total,
    coveredMs:selected.reduce((sum,p)=>sum+p.dt,0),minSpacingMs:selected[0].dt,maxSpacingMs:selected.at(-1).dt};
}
function jackCadence(ctx,start,end,tag,speedRate=1,config={}){
  const views=handViews(ctx,start,end),resolutionMs=1/speedRate,measurements=[];
  for(let thumb=0;thumb<views.length;thumb++){
    const sides=[];
    for(let side=0;side<2;side++){
      const rows=views[thumb][side],events=rows,pairs=[];
      if(tag==='chordjack'){
        let last=null,between=0;
        for(const b of events){
          if(!b.coactive){between|=b.mask;continue;}
          // The coactive backbone may carry extra retriggers of fingers
          // shared by its two grips (an overlapping Anchor/Minijack role).
          // A changing same-hand roll outside those common fingers cannot
          // certify a slower whole-hand rebound from its sparse accents.
          if(last&&!(between&~(last.mask&b.mask)))pairs.push({dt:b.t-last.t,width:Math.min(last.width,b.width)});
          last=b;between=0;
        }
      }else for(let i=1;i<events.length;i++){
        const a=events[i-1],b=events[i],overlap=popcount(a.mask&b.mask);
        if(overlap)pairs.push({dt:b.t-a.t,width:overlap});
      }
      const m=intervalMode(pairs,resolutionMs,config);
      if(m)sides.push({...m,side,eventCount:events.length});
    }
    // A sustained faster hand may determine the jack rate; simultaneous and
    // alternating other-hand events never double this rate.
    const maximumCoverage=Math.max(0,...sides.map(s=>s.coveredMs));
    const sustained=sides.filter(s=>s.coveredMs>=maximumCoverage*(config.handSupportRatio??MODEL.handSupportRatio));
    sustained.sort((a,b)=>a.spacingMs-b.spacingMs);
    measurements.push({thumb,sides,selected:sustained[0]??null});
  }
  const candidates=measurements.map(m=>m.selected).filter(Boolean);
  if(candidates.length!==2||Math.abs(candidates[0].spacingMs-candidates[1].spacingMs)>2*resolutionMs+1e-7)return null;
  const m=candidates[0],spacingMs=(candidates[0].spacingMs+candidates[1].spacingMs)/2;
  return {source:tag==='chordjack'?'hand-strike-interval-mode':'same-finger-rearticulation-mode',
    equivBpm:Math.round(15000/spacingMs),spacingMs,resolutionMs,rowCount:m.eventCount,
    pairCount:m.pairCount,intervalShare:Math.min(...candidates.map(c=>c.intervalShare)),
    handMeasurements:measurements.map(({thumb,sides})=>({thumb,hands:sides.map(({side,spacingMs,intervalShare,eventCount,coveredMs})=>({side,spacingMs,intervalShare,eventCount,coveredMs}))}))};
}
// CS is a stream of finger substitutions across the two hands, not a
// repeated whole-hand rebound. Its stream clock is the union of observed
// substitutions; it must not inherit the Jack per-hand division by two.
function switchCadence(ctx,start,end,speedRate=1,config={}){
  const measurements=handViews(ctx,start,end).map((sides,thumb)=>{
    const times=[];
    for(const rows of sides)for(let i=1;i<rows.length;i++){
      const a=rows[i-1].mask,b=rows[i].mask;
      if((a&~b)&&(b&~a))times.push(rows[i].t);
    }
    const sorted=[...new Set(times)].sort((a,b)=>a-b),pairs=sorted.slice(1).map((t,i)=>({dt:t-sorted[i],width:1}));
    const m=intervalMode(pairs,1/speedRate,config);
    return m?{...m,thumb,eventCount:sorted.length}:null;
  });
  if(measurements.some(m=>!m)||Math.abs(measurements[0].spacingMs-measurements[1].spacingMs)>2/speedRate+1e-7)return null;
  const m=measurements[0],spacingMs=(m.spacingMs+measurements[1].spacingMs)/2;
  return {source:'hand-substitution-stream-mode',equivBpm:Math.round(15000/spacingMs),spacingMs,resolutionMs:1/speedRate,
    rowCount:m.eventCount,pairCount:m.pairCount,intervalShare:Math.min(...measurements.map(m=>m.intervalShare))};
}
function describeActions(ctx,start,end){
  return {meaning:'per-hand head-event geometry, both thumb assignments',
    thumbAssignments:handViews(ctx,start,end).map((sides,thumb)=>({thumb,hands:sides.map((rows,side)=>{
      let coactivation=0,rearticulation=0,substitution=0,unchanged=0;
      for(let i=0;i<rows.length;i++){
        coactivation+=rows[i].width>1;
        if(!i)continue;const a=rows[i-1].mask,b=rows[i].mask;
        rearticulation+=!!(a&b);substitution+=!!(a&~b)&&!!(b&~a);unchanged+=a===b;
      }
      return {side,eventCount:rows.length,coactiveFraction:coactivation/Math.max(1,rows.length),
        rearticulationFraction:rearticulation/Math.max(1,rows.length-1),
        substitutionFraction:substitution/Math.max(1,rows.length-1),fixedGripFraction:unchanged/Math.max(1,rows.length-1)};
    })}))};
}
module.exports={handViews,intervalMode,jackCadence,switchCadence,describeActions};
