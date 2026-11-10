'use strict';
// These actively annotated constructions are mechanism checks, not a random
// accuracy set. Some were training-dev cases; D/H and remaining rates were not.
const fs=require('node:fs'),path=require('node:path');
const {createProbe}=require('./activity_overlay_probe.cjs');
const {apply}=require('./joint_runtime.cjs');
const product=require('../../src/index.cjs');
const candidate=process.argv[2];if(!candidate)throw Error('Provide the exact candidate JSON');
const model=JSON.parse(fs.readFileSync(candidate,'utf8')).model,analyze=createProbe(model,apply);
function coverage(frames,a,b){const out={};for(const s of frames){const dt=Math.max(0,Math.min(b,s.end)-Math.max(a,s.start));if(dt)out[s.tag]=(out[s.tag]??0)+dt/(b-a);}return out;}
const sources=[...['A01','A02','B01','B02'].map(id=>({round:1,id})),...['C16','C24','C32','C48','CTX-L','CTX-H'].map(id=>({round:2,id})),...['D000','D008','D024','D072','D125','D216','LN-L','LN-H'].map(id=>({round:3,id})),...['H01','H02','H03','H04'].map(id=>({round:7,id}))],out=[];
for(const item of sources){
  const file=path.join('artifacts','frontier-annotation-r'+item.round,item.id+'.osu'),text=fs.readFileSync(file,'utf8'),old=product.analyzeMap(text),r=analyze(text),target=item.id.startsWith('H')||item.id.startsWith('CTX')||item.id.startsWith('LN-')?[9000,17000]:[r.map.firstNote,r.map.lastNote];
  let release=0;const cls=require('../../src/models/perception7k.json').profiles.actions.classes.indexOf('release');r.debug.rawSegments.forEach((s,j)=>release+=Math.max(0,Math.min(target[1],s.end)-Math.max(target[0],s.start))*r.debug.activity.support[j][cls]/(target[1]-target[0]));
  const row={...item,file,target,baseline:coverage(old.segments,...target),candidate:coverage(r.segments,...target),releaseAffinityMean:release,overallTags:r.overall.tags,overallPercentages:r.overall.percentages,displayBpm:r.segments.filter(s=>s.end>target[0]&&s.start<target[1]).map(s=>({tag:s.tag,start:s.start,end:s.end,bpm:s.equivBpm??null})),labelScope:item.id==='D216'?'No user answer; descriptive only':item.round===7?'H01/H03 not CS; H02/H04 genuine cut; no other exact primary supplied':'Use preserved user annotations, including qualitative/mixed answers; no invented hard grade'};out.push(row);console.log(item.id,row.baseline,'=>',row.candidate,'whole',row.overallTags);
}
fs.writeFileSync('research/local_feedback/results/joint-extra-mechanisms-r7.json',JSON.stringify({candidate,scope:'Actual-engine research adapter, Actions fallback. Active constructed mechanism checks; not held-out performance.',cases:out},null,2)+'\n');
