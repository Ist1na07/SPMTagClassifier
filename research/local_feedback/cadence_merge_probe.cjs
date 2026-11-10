'use strict';
// A readout reproduction, not a classifier accuracy experiment: fixed Speed
// labels expose whether finer frames erase an actually observed tempo change.
const {createProbe}=require('./activity_overlay_probe.cjs');
const fs=require('node:fs');
function fixture(){
  const notes=[];let t=1000,k=0;
  for(const [count,step]of [[200,100],[267,75]])for(let i=0;i<count;i++,k++,t+=step)
    for(const col of(k%2?[2,5,6]:[0,1,4]))notes.push(`${Math.floor((col+.5)*512/7)},192,${t},1,0,0:0:0:0:`);
  return ['osu file format v14','[General]','Mode:3','[Metadata]','Title:Readout tempo regression','Artist:Research','Creator:SPM','Version:7K','[Difficulty]','CircleSize:7','OverallDifficulty:8','[TimingPoints]','0,400,4,2,1,50,1,0','21000,300,4,2,1,50,1,0','[HitObjects]',...notes].join('\n');
}
function adapter(_model,_map,raw,_candidates,classes){
  const frames=[];raw.forEach((s,parent)=>{for(let start=s.start;start<s.end;start+=500)frames.push({...s,start,end:Math.min(s.end,start+500),family:'rc',tag:'speed',secondary:undefined,scores:{speed:1},confidence:1,parent});});
  return {frames,support:frames.map(()=>classes.map(c=>Number(c==='speed')))};
}
if(require.main===module){
  const text=fixture(),result=createProbe({},adapter)(text);
  const receipt={scope:'Fixed labels; actual compiled readout, not a learned model score',segments:result.segments.map(({start,end,tag,equivBpm,evidence})=>({start,end,tag,equivBpm,cadence:evidence.cadence})),rawParents:[...new Set(result.debug.rawSegments.map(s=>s.parent))].length};
  fs.writeFileSync('research/local_feedback/results/cadence-merge-reproduction.json',JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
}
module.exports={fixture,adapter};
