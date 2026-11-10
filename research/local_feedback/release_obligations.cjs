'use strict';
// Continuous release obligations on both physical tail and held-body clocks.
const {lowerBound:lb}=require('../../src/parser.cjs');
const p=require('../../integration/rating-parameters.json').detail;
function upper(a,t,key=x=>x.t){let lo=0,hi=a.length;while(lo<hi){const mid=(lo+hi)>>>1;if(key(a[mid])<=t)lo=mid+1;else hi=mid;}return lo;}
function evidence(ctx){
  const od=Number.isFinite(ctx.map.meta.od)?ctx.map.meta.od:6,q=(64.5-Math.ceil(3*od))/500,x=.3*Math.sqrt(Math.max(q,1e-6)),precision=Math.min(x,.6*(x-.09)+.09)**p.od_mult_r,tau=1000*precision;
  const tails=ctx.holds.flatMap((hs,col)=>hs.map(([head,t])=>({head,t,col}))).sort((a,b)=>a.t-b.t||a.col-b.col),events=[],body=[];
  for(let j=0;j<tails.length;){
    const t=tails[j].t,group=[];while(j<tails.length&&tails[j].t===t)group.push(tails[j++]);
    const released=new Set(group.map(v=>v.col));let retained=0;
    for(let c=0;c<ctx.map.keys;c++)if(!released.has(c)){const hs=ctx.holds[c],i=lb(hs,t,h=>h[0])-1;if(i>=0&&hs[i][1]>t)retained+=(-Math.expm1(-(t-hs[i][0])/tau))*(-Math.expm1(-(hs[i][1]-t)/tau));}
    let sumI=0,sumGuided=0;
    for(const tail of group){
      const col=ctx.cols[tail.col],own=col[upper(col,t,n=>n.t)],any=ctx.map.notes[upper(ctx.map.notes,t,n=>n.t)],next=own??any;
      const ih=.001*Math.abs(t-tail.head-80)/precision,it=next?.t!==undefined ? .001*Math.abs(next.t-t-80)/precision :10;
      const I=2/(2+Math.exp(-p.r_I_steep*(ih-p.r_I_off))+Math.exp(-p.r_I_steep*(it-p.r_I_off)));sumI+=I;sumGuided+=I*retained;
      body.push({t:tail.head,v:[retained/ctx.map.keys,I*retained/ctx.map.keys]},{t,v:[-retained/ctx.map.keys,-I*retained/ctx.map.keys]});
    }
    events.push({t,v:[group.length*retained,sumGuided,sumI]});
  }
  const prefix=[[0,0,0]];for(const e of events)prefix.push(prefix.at(-1).map((v,j)=>v+e.v[j]));
  body.sort((a,b)=>a.t-b.t);const integral=[];let at=ctx.map.firstNote,active=[0,0],area=[0,0];
  for(const e of body){area=area.map((v,j)=>v+active[j]*(e.t-at));active=active.map((v,j)=>v+e.v[j]);integral.push({t:e.t,active:[...active],area:[...area]});at=e.t;}
  const areaAt=t=>{const i=upper(integral,t)-1;return i<0?[0,0]:integral[i].area.map((v,j)=>v+integral[i].active[j]*(t-integral[i].t));};
  return(a,b)=>{const lo=lb(events,a,e=>e.t),hi=lb(events,b,e=>e.t),A=areaAt(a),B=areaAt(b);return[...prefix[hi].map((v,j)=>(v-prefix[lo][j])/ctx.map.keys/((b-a)/1000)),...B.map((v,j)=>(v-A[j])/(b-a))];};
}
module.exports={evidence};
