'use strict';
// Explicit, independently selectable research treatments. Never edit the archive.
function patch(id,description,apply){return {id,description,apply};}
const area=patch('area','Integrate LN union per lane once, clipped to the window',(s,r)=>{
 const from='        const hs = ctx.lnHeads;\n        const ts = ctx.lnTails;';
 const to='        f.lnAreaRatio = area / ((t1 - t0) * K);';
 const a=s.indexOf(from),b=s.indexOf(to,a);if(a<0||b<0)throw Error('area anchors missing');
 const old=s.slice(a,b+to.length);
 return r(s,old,`        const lanes = Array.from({length:K},()=>[]);
        const aLo = lowerBoundObj(ctx.lnHeads,t0-ctx.lnMaxDur,h=>h.t);
        const aHi = lowerBoundObj(ctx.lnHeads,t1,h=>h.t);
        for(let i=aLo;i<aHi;i++){
          const h=ctx.lnHeads[i],start=Math.max(t0,h.t),end=Math.min(t1,h.t+h.dur);
          if(end>start)lanes[h.col].push([start,end]);
        }
        let area=0;
        for(const intervals of lanes){
          let end=-Infinity;
          for(const [a,b] of intervals){if(b>end){area+=b-Math.max(a,end);end=b;}}
        }
        f.lnAreaRatio = area / ((t1 - t0) * K);`);
});
const adjacency=patch('head_adjacency','Head-to-head row statistics skip release-only rows',(s,r)=>r(s,
 `      if (i > 0 && rows[i - 1].hits > 0) {\n        const prev = rows[i - 1];`,
 `      let previous=i-1;while(previous>=0&&rows[previous].hits===0)previous--;
      if (previous >= 0) {
        const prev = rows[previous];`));
const heldOrder=patch('held_order','Half-open holds: process tails before heads at identical time',(s,r)=>r(s,
 '    events.sort((a, b) => a.t - b.t || a.type - b.type);',
 '    events.sort((a, b) => a.t - b.t || b.type - a.type);'));
const phase=patch('timing_phase','Use each active uninherited timing point as beat phase origin',(s,r)=>r(s,
 `          const q = beatLengthAt(ctx.map.timingPoints, s.t) / 4;\n          let d = (s.t - anchor) % q;`,
 `          const points=ctx.map.timingPoints;
          const pi=Math.max(0,upperBoundObj(points,s.t,p=>p.t)-1);
          const q = points[pi].beatLength / 4;
          let d = (s.t - points[pi].t) % q;`));
// A replacement feature, not a unique correction of the ambiguous old name.
// Every actual head event matters, regardless of lane or input ordering; a
// release >40 ms from its nearest head is temporally independent.
const releaseIsolation=patch('release_isolation','Release independence = fraction >40 ms from nearest head',(s,r)=>r(s,
 `          const ev = ctx.allEvents;
          const ei = lowerBoundObj(ev, tails[i].t, (e) => e.t);
          if (ei < ev.length && ev[ei].t - tails[i].t < 3e3) {
            if (ev[ei].col !== tails[i].col) indep++;
          }`,
 `          const ns=ctx.map.notes,ei=lowerBoundObj(ns,tails[i].t,n=>n.t);
          const before=ei>0?tails[i].t-ns[ei-1].t:Infinity;
          const after=ei<ns.length?ns[ei].t-tails[i].t:Infinity;
          if(Math.min(before,after)>40)indep++;`));
const sustaining=patch('sustaining','Compute hold occupancy and releases in windows without heads',(s,r)=>{
 s=r(s,'    if (rowCount === 0) return f;',
  '    // Sustain-only windows still carry occupancy and release evidence.');
 return r(s,'      f.lockRatio /= rowCount;','      f.lockRatio /= Math.max(1,rowCount);');
});
module.exports={area,adjacency,heldOrder,phase,releaseIsolation,sustaining,
 logical:[area,adjacency,heldOrder,phase,sustaining]};
