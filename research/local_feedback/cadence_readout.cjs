'use strict';
// Research-only consumer ablation. The fitted type probabilities are unchanged.
// Measure existing reliable row cadence before coalescing fine model frames;
// otherwise the very act of smoothing hides the tempo boundary from the meter.
const fs=require('node:fs'),path=require('node:path');
const {createRequire}=require('node:module');
const {prepare,parseOsu}=require('../../src/parser.cjs');
const {correctCadenceDisplay}=require('../../src/cadence.cjs');
function smoothCadenceFramesNaive(frames,ctx,speedRate=1){
  const segments=frames.map(s=>({...s,evidence:{sourceStart:s.start,sourceEnd:s.end}}));
  correctCadenceDisplay({segments,modelInfo:{}},ctx,speedRate);
  const out=[];
  for(const s of segments){
    const p=out.at(-1);
    if(p&&p.tag===s.tag&&p.secondary===s.secondary&&p.family===s.family&&p.equivBpm===s.equivBpm&&s.end-s.start<1800){p.end=s.end;p.evidence.sourceEnd=s.end;}
    else out.push(s);
  }
  const out2=[];
  for(let i=0;i<out.length;i++){
    const s=out[i],p=out2.at(-1),n=out[i+1];
    if(s.family==='break'&&s.end-s.start<1800&&p&&n&&n.tag===p.tag&&n.equivBpm===p.equivBpm){p.end=s.end;p.evidence.sourceEnd=s.end;}
    else out2.push(s);
  }
  return out2;
}
function smoothCadenceFrames(frames,ctx,speedRate=1){
  const minDisplay=1800; // Existing 3000-ms display minimum times 0.6.
  const same=(a,b)=>a.tag===b.tag&&a.secondary===b.secondary&&a.family===b.family;
  const measured=(s,anchor=s.equivBpm)=>{
    const copy={...s,evidence:{sourceStart:s.start,sourceEnd:s.end}};
    correctCadenceDisplay({segments:[copy],modelInfo:{}},ctx,speedRate);
    const c=copy.evidence.cadence;
    if(c&&Number.isFinite(anchor)){
      // Each endpoint has at most one source-ms *difference* uncertainty.
      // If the old rounded BPM cell is compatible with that slope interval,
      // prefer it over flickering across the shared half-integer boundary.
      const e=c.resolutionMs/(c.rowCount-1),lo=15000/(c.spacingMs+e),hi=c.spacingMs>e?15000/(c.spacingMs-e):Infinity;
      if(lo<=anchor+.5&&hi>=anchor-.5)copy.equivBpm=anchor;
    }
    return copy.evidence.cadence?copy:null;
  };
  // Preserve the established label smoothing first. A tempo correction must
  // not produce an extra stream of 500-ms +/-1-BPM labels from timestamp
  // rounding. Refit the SAME reliable line over a growing uniform plateau.
  const groups=[];
  for(const s of frames){const g=groups.at(-1);if(g&&same(g,s)&&s.end-s.start<minDisplay){g.end=s.end;g.children.push(s);}else groups.push({...s,children:[s]});}
  const labelGroups=[];
  for(let i=0;i<groups.length;i++){
    const g=groups[i],p=labelGroups.at(-1),n=groups[i+1];
    if(g.family==='break'&&g.end-g.start<minDisplay&&p&&n&&p.tag===n.tag){p.end=g.end;continue;}
    labelGroups.push(g);
  }
  const out=[];
  for(const g of labelGroups){
    let pending=null;const plateaus=[];
    const finish=()=>{if(pending&&pending.end-pending.start>=minDisplay)plateaus.push(pending);pending=null;};
    for(const s of g.children){
      if(pending){const joined=measured({...pending,end:s.end},g.equivBpm);if(joined){pending=joined;continue;}finish();}
      pending=measured(s,g.equivBpm);
    }
    finish();
    if(!plateaus.length){const {children,...s}=g;out.push(s);continue;}
    const {children,...base}=g;
    const emit=(start,end,bpm)=>{
      if(end<=start)return;
      const s={...base,start,end,equivBpm:bpm};
      const previous=out.at(-1);
      if(previous&&previous.end===s.start&&same(previous,s)&&previous.equivBpm===s.equivBpm)previous.end=s.end;else out.push(s);
    };
    let cursor=g.start;
    for(let i=0;i<plateaus.length;i++){
      const p=plateaus[i],gap=p.start-cursor;
      let start=p.start;
      if(gap>=minDisplay)emit(cursor,p.start,g.equivBpm);
      else if(i===0)start=cursor;
      else if(gap>0)out.at(-1).end=p.start;
      emit(start,p.end,p.equivBpm);cursor=p.end;
    }
    // Do not carry a short reliable cadence across a long unmeasured region.
    // Only a transition shorter than the established display minimum may be
    // absorbed into its neighbour; long irregular regions keep old fallback.
    if(g.end-cursor>=minDisplay)emit(cursor,g.end,g.equivBpm);
    else if(g.end>cursor)out.at(-1).end=g.end;
  }
  return out;
}
function createReadoutProbe(model,adapter){
  const file=path.resolve(__dirname,'activity_overlay_probe.cjs');
  let source=fs.readFileSync(file,'utf8');
  const before='const segments = smoothSegments(visibleRaw ? visibleRaw.map(s=>({...s})) : classifySegments(cm,{scorer:"model"}));';
  if(source.split(before).length!==2)throw Error('Readout hook source drift');
  source=source.replace(before,'const segments = opts._cadenceReadout ? opts._cadenceReadout(visibleRaw) : smoothSegments(visibleRaw ? visibleRaw.map(s=>({...s})) : classifySegments(cm,{scorer:"model"}));');
  const module={exports:{}};new Function('require','module','exports','__dirname',source)(createRequire(file),module,module.exports,__dirname);
  const analyze=module.exports.createProbe(model,adapter);
  return (text,opts={})=>{const ctx=prepare(parseOsu(text,opts));return analyze(text,{...opts,_cadenceReadout:f=>smoothCadenceFrames(f,ctx,opts.speedRate??1)});};
}
module.exports={smoothCadenceFrames,smoothCadenceFramesNaive,createReadoutProbe};
