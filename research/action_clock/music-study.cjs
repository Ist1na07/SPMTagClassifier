'use strict';
const fs=require('node:fs'),api=require('../../src/index.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{musicalSegments,measureLines}=require('../../src/musical-segmentation.cjs');
const manifest=JSON.parse(fs.readFileSync('.cache/local-feedback-activity-r6b/manifest.json','utf8'));
const resources=new Map(fs.readFileSync('.cache/perception-final-resource.jsonl','utf8').trim().split('\n').map(s=>{const r=JSON.parse(s);return [r.uid,r]}));
const configs=[{id:'off'},...[.5,1,2].flatMap(maxMoveBeats=>['nearest','contrast'].map(mode=>({id:mode+'-'+maxMoveBeats,maxMoveBeats,mode})))];
const protocol={scope:'Six previously exposed manual charts; development mechanism/parameter comparison, not new blind evidence.',configs,
 choice:'mean duration-weighted allowed-label accuracy across charts and consumers; musical-boundary alignment breaks ties; never optimize exemplar BPM as segmentation target',
 fixed:'W/b, pressure guards, integral weights/profiles, frozen reference segmentation; only musical-boundary candidate selection changes'};
fs.writeFileSync('research/action_clock/results/music-protocol.json',JSON.stringify(protocol,null,2));
function metric(record,segments){let correct=0,total=0,falseBreak=0;
 for(const truth of record.annotations){if(!truth.tags.length)continue;total+=truth.end-truth.start;
 for(const s of segments){const dt=Math.max(0,Math.min(s.end,truth.end)-Math.max(s.start,truth.start));correct+=dt*truth.tags.includes(s.tag);falseBreak+=dt*(s.tag==='break'&&!truth.tags.includes('break'));}}
 return {accuracy:correct/total,totalMs:total,correctMs:correct,falseBreakMs:falseBreak};}
const grid=[];
for(const config of configs){const details=[];
 for(const uid of manifest.manualUids){
  const record=manifest.records.find(r=>r.uid===uid),resource=resources.get(uid),text=fs.readFileSync(record.source,'utf8'),ctx=prepare(parseOsu(text));
  for(const name of ['actions','detail','compact']){
   let cuts=[];const opts=name==='actions'?{}:{ratingEvidence:resource.ratings[name]};opts._musicalSegmenter=false;
   if(config.id!=='off')opts._musicalSegmenter=input=>{const out=musicalSegments(ctx,input,config);cuts=out;return out;};
   const r=api.analyzeMap(text,opts),lines=measureLines(ctx.map);
   const internal=r.segments.slice(0,-1).map(s=>s.end),aligned=internal.filter(t=>lines.some(p=>Math.abs(p.t-t)<=1)).length;
   details.push({uid,consumer:name,...metric(record,r.segments),alignedBoundaries:aligned,boundaries:internal.length,
    cuts:cuts.map(s=>({start:s.start,end:s.end,windowRange:s.windowRange,musicalStart:s.musicalStart})),segments:r.segments.map(({start,end,tag,equivBpm})=>({start,end,tag,equivBpm}))});
  }
 }
 const row={config,accuracy:details.reduce((s,d)=>s+d.accuracy,0)/details.length,falseBreakMs:details.reduce((s,d)=>s+d.falseBreakMs,0),aligned:details.reduce((s,d)=>s+d.alignedBoundaries,0),boundaries:details.reduce((s,d)=>s+d.boundaries,0),details};grid.push(row);
 console.log(config.id,row.accuracy,row.falseBreakMs,row.aligned+'/'+row.boundaries);
 fs.writeFileSync('research/action_clock/results/music-grid.json',JSON.stringify(grid,null,2));
}
