'use strict';
const {lowerBound:lb}=require('./parser.cjs');
const {correctCadenceDisplay}=require('./cadence.cjs');
const CS=new Set(['dense chordstream','fast chordstream']);
const STEP_MS=500,CONTEXT_MS=4000,MIN_DISPLAY_MS=1800;

// Exact hand events, not globally adjacent rows. The flexible centre thumb is
// assigned to either hand; exclusion must hold under BOTH assignments.
function prepareHands(ctx){
  return [0,1].map(thumb=>[0,1].map(side=>{
    const rows=[];
    for(const n of ctx.map.notes){
      if((n.col===3?thumb:n.col<3?0:1)!==side)continue;
      const last=rows.at(-1);if(last&&last.t===n.t)last.mask|=1<<n.col;else rows.push({t:n.t,mask:1<<n.col});
    }
    return rows;
  }));
}
function noCut(ctx,hands,start,end){
  const a=lb(ctx.rows,start,r=>r.t),b=lb(ctx.rows,end,r=>r.t);
  if(b-a<4)return false;
  const t=lb(ctx.tails,start,r=>r.t);if(t<ctx.tails.length&&ctx.tails[t].t<end)return false;
  for(let j=a;j<b;j++)if(ctx.rows[j].lnMask)return false;
  for(const hs of ctx.holds){const j=lb(hs,end,h=>h[0])-1;if(j>=0&&hs[j][1]>start)return false;}
  for(const view of hands){
    let pairs=0;
    for(const rows of view){
      const lo=lb(rows,start,r=>r.t),hi=lb(rows,end,r=>r.t);
      for(let i=Math.max(1,lo);i<hi;i++){pairs++;if(rows[i-1].mask!==rows[i].mask)return false;}
    }
    if(!pairs)return false;
  }
  return true;
}
function excludeCut(s){
  const scores={...s.scores,'dense chordstream':0,'fast chordstream':0};
  if(!CS.has(s.tag))return {...s,scores,secondary:CS.has(s.secondary)?undefined:s.secondary};
  // Anchor/Vibro remain overlapping attributes, not replacement base types.
  const ranked=Object.entries(scores).filter(([c])=>!CS.has(c)&&c!=='anchor'&&c!=='vibro').sort((a,b)=>b[1]-a[1]);
  let [tag,value]=ranked[0]??['wildcard',0];
  if(tag==='wildcard')for(const [c,v]of ranked.slice(1))if(c!=='wildcard'&&v>.3){tag=c;value=v;break;}
  const other=Object.entries(scores).filter(([c])=>c!==tag&&!CS.has(c)).sort((a,b)=>b[1]-a[1])[0];
  const secondary=other&&other[1]>=.75*value&&other[1]>.3?other[0]:undefined;
  return {...s,scores,tag,confidence:Math.min(1,value),secondary};
}

function smoothCadenceFrames(frames,ctx,speedRate=1){
  const same=(a,b)=>a.tag===b.tag&&a.secondary===b.secondary&&a.family===b.family;
  const measured=(s,anchor=s.equivBpm)=>{
    const copy={...s,evidence:{sourceStart:s.start,sourceEnd:s.end}};
    correctCadenceDisplay({segments:[copy],modelInfo:{}},ctx,speedRate);
    const c=copy.evidence.cadence;
    if(c&&Number.isFinite(anchor)){
      const e=c.resolutionMs/(c.rowCount-1),lo=15000/(c.spacingMs+e),hi=c.spacingMs>e?15000/(c.spacingMs-e):Infinity;
      if(lo<=anchor+.5&&hi>=anchor-.5)copy.equivBpm=anchor;
    }
    return c?copy:null;
  };
  const groups=[];
  for(const s of frames){
    const g=groups.at(-1);
    // Keep long original expert boundaries and their distinct scores. Splits
    // introduced only for this readout must not merge those old observations.
    if(g&&same(g,s)&&(g.parent===s.parent||s.parentDuration<MIN_DISPLAY_MS)){
      g.end=s.end;g.children.push(s);
    }else groups.push({...s,children:[s]});
  }
  const labelGroups=[];
  for(let i=0;i<groups.length;i++){
    const g=groups[i],p=labelGroups.at(-1),n=groups[i+1];
    if(g.family==='break'&&g.end-g.start<MIN_DISPLAY_MS&&p&&n&&p.tag===n.tag){p.end=g.end;continue;}
    labelGroups.push(g);
  }
  const out=[];
  for(const g of labelGroups){
    let pending=null;const plateaus=[];
    const finish=()=>{if(pending&&pending.end-pending.start>=MIN_DISPLAY_MS)plateaus.push(pending);pending=null;};
    for(const s of g.children){
      if(pending){const joined=measured({...pending,end:s.end},g.equivBpm);if(joined){pending=joined;continue;}finish();}
      pending=measured(s,g.equivBpm);
    }
    finish();
    const {children,parent,parentDuration,...plain}=g,base={...plain,_archivedEquivBpm:g.equivBpm};
    if(!plateaus.length){out.push(base);continue;}
    const groupStart=out.length;
    const emit=(start,end,bpm,measurement)=>{
      if(end<=start)return;
      const s={...base,start,end,equivBpm:bpm,...(measurement?{evidence:{cadence:{...measurement.evidence.cadence,legacyEquivBpm:g.equivBpm,measuredStart:measurement.start,measuredEnd:measurement.end}}}:{})},previous=out.at(-1);
      if(out.length>groupStart&&previous.end===s.start&&same(previous,s)&&previous.equivBpm===s.equivBpm)previous.end=s.end;else out.push(s);
    };
    let cursor=g.start;
    for(let i=0;i<plateaus.length;i++){
      const p=plateaus[i],gap=p.start-cursor;let start=p.start;
      if(gap>=MIN_DISPLAY_MS)emit(cursor,p.start,g.equivBpm);
      else if(i===0)start=cursor;
      else if(gap>0)out.at(-1).end=p.start;
      emit(start,p.end,p.equivBpm,p);cursor=p.end;
    }
    if(g.end-cursor>=MIN_DISPLAY_MS)emit(cursor,g.end,g.equivBpm);
    else if(g.end>cursor)out.at(-1).end=g.end;
  }
  return out;
}
function refine(ctx,source,classes,{speedRate=1,supportPower=1,resolveBpm}={}){
  const map=ctx.map,hands=prepareHands(ctx),frames=[],support=[];let exclusions=0;
  source.forEach((old,parent)=>{
    const scores=old.scores??{[old.tag]:1},div=Math.max(1,...Object.values(scores));
    const q=classes.map(c=>old.family==='break'?0:Math.min(.999,Math.max(0,scores[c]??0)/div)**supportPower);
    for(let start=old.start;start<old.end;start+=STEP_MS){
      const end=Math.min(old.end,start+STEP_MS),mid=(start+end)/2,a=Math.max(map.firstNote,mid-CONTEXT_MS/2),b=Math.max(a+1,Math.min(map.lastNote,mid+CONTEXT_MS/2));
      const excluded=old.family==='rc'&&noCut(ctx,hands,a,b);let s={...old,start,end,parent,parentDuration:old.end-old.start};
      if(excluded){s=excludeCut(s);exclusions++;}
      if(s.tag!==old.tag&&resolveBpm){const bpm=resolveBpm(s.tag,parent);if(bpm===undefined)delete s.equivBpm;else s.equivBpm=bpm;}
      frames.push(s);support.push(q.map((v,j)=>excluded&&CS.has(classes[j])?0:v));
    }
  });
  return {frames,support,segments:smoothCadenceFrames(frames,ctx,speedRate),exclusions};
}
module.exports={refine,prepareHands,noCut,smoothCadenceFrames};
