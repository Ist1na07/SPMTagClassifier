'use strict';
// Discriminating probe, not a tuned/final model: do short-time physical inputs
// actually distinguish the Tech prefixes hidden inside long Dense segments?
const fs=require('node:fs');
const {loadLab,context}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const {parseOsu,prepare}=require('../../src/parser.cjs'),{regularRowCadence}=require('../../src/cadence.cjs');
const manifest=JSON.parse(fs.readFileSync('.cache/perception-r1/manifest.json','utf8'));
const targets=manifest.records.filter(r=>manifest.manualUids.includes(r.uid));
for(const id of ['F01','F02']){
  const r=JSON.parse(fs.readFileSync('research/frontier/annotations-r5.json','utf8')).cases[id];
  const original=manifest.records.find(v=>v.uid===r.sourceUid);targets.push({...original,uid:'probe/'+id,annotations:[{start:r.sourceTargetMs[0],end:r.sourceTargetMs[1],tags:['speed']}]});
}
const api=loadLab({patches:[actionRatePatch(profile.gate)]});
const root='research/local_feedback/results/temporal-probe-r4';fs.mkdirSync(root,{recursive:true});
const results=[];
for(const r of targets){
  const source=fs.readFileSync(r.source,'utf8'),old=context(api.lab,source),native=prepare(parseOsu(source));
  const cut=old.windows.map(w=>w.cutDemand).filter(x=>x>0).sort((a,b)=>a-b),cutRef=cut[Math.floor(cut.length*.8)]??1;
  for(const width of [500,1000,2000,4000])for(const cadence of [false,true]){
    const frames=[];let parent=0;
    for(let start=old.map.firstNote;start<old.map.lastNote;start+=250){
      const end=Math.min(start+250,old.map.lastNote),center=(start+end)/2,a=Math.max(old.map.firstNote,center-width/2),b=Math.min(old.map.lastNote,center+width/2);
      while(parent+1<r.frames.length&&r.frames[parent+1].start<=center)parent++;
      const inherited=r.frames[parent];
      if(inherited.family==='break'){frames.push({start,end,tag:'break',family:'break'});continue;}
      const f=api.lab.computeWindowFeatures(old.ctx,a,b),extra=api.lab.computeSegmentExtras(old.ctx,a,b);
      if(cadence){const measured=regularRowCadence(native,a,b);if(measured)f.equivBpm=measured.equivBpm;}
      const family=inherited.family;
      const model=api.lab.modelScores(family,f,extra,cutRef),rule=family==='rc'?api.lab.scoreRc({f,extras:extra,cutDemandRef:cutRef}):api.lab.scoreLn({f,extras:extra});
      const scores=api.lab.blendScores(model,rule,'model');const ranked=Object.entries(scores).sort((a,b)=>b[1]-a[1]);let [tag,value]=ranked[0];
      if(['wildcard','ln wildcard'].includes(tag))for(const [c,v]of ranked.slice(1))if(!['wildcard','ln wildcard'].includes(c)&&v>.3){tag=c;value=v;break;}
      frames.push({start,end,tag,family,scores,confidence:Math.min(1,value)});
    }
    let correct=0,total=0,techDense=0,techCs=0;const actual={};
    for(const ann of r.annotations){if(!ann.tags.length)continue;total+=ann.end-ann.start;
      for(const f of frames){const dt=Math.max(0,Math.min(f.end,ann.end)-Math.max(f.start,ann.start));if(!dt)continue;
        correct+=dt*Number(ann.tags.includes(f.tag));techDense+=dt*Number(ann.tags.includes('tech')&&f.tag==='dense chordstream');
        techCs+=dt*Number((ann.tags.includes('tech')&&['dense chordstream','fast chordstream'].includes(f.tag))||(f.tag==='tech'&&ann.tags.some(t=>['dense chordstream','fast chordstream'].includes(t))));
        const k=ann.tags.join('|')+' -> '+f.tag;actual[k]=(actual[k]??0)+dt;
      }
    }
    const row={uid:r.uid,width,cadence,accuracy:correct/Math.max(1,total),techToDenseMs:techDense,techCsConfusionMs:techCs,transitions:actual};results.push(row);
    fs.writeFileSync(root+'/'+r.uid.replaceAll('/','__')+'-'+width+'-'+Number(cadence)+'.json',JSON.stringify({row,frames})+'\n');
    console.log(r.uid,width,cadence,(row.accuracy*100).toFixed(2),row.techToDenseMs);
  }
}
fs.writeFileSync(root+'/results.json',JSON.stringify({scope:'Untuned retrospective probe on6manual +2previously annotated real Speed intervals. Not full349, not a new OOF score.',stepMs:250,results,source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
