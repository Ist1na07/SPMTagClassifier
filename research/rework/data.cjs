'use strict';
const fs=require('node:fs'),path=require('node:path');
const ROOT=path.resolve(__dirname,'../..');
function annotation(text){
 const ms=s=>{const a=s.split(':').map(Number);return a[0]*60000+a[1]*1000+a[2];};
 let cursor=null;const intervals=[];
 for(const raw of text.split(/\r?\n/)){
  const line=raw.split('#')[0].trim();
  const m=line.match(/^(?:(\d+:\d+:\d+)\s*)?-\s*(\d+:\d+:\d+)\s+(.+)$/);
  if(!m)continue;
  const start=m[1]?ms(m[1]):cursor,end=ms(m[2]);cursor=end;
  const tags=m[3].toLowerCase().split(',').map(s=>s.trim()).filter(s=>s!=='?');
  if(start!==null&&end>start)intervals.push({start,end,tags,comment:raw.includes('#')?raw.split('#').slice(1).join('#').trim():''});
 }
 return intervals;
}
function records7(){
 const folder='spm_dataset-1.1.1',dir=path.join(ROOT,'dataset',folder);
 const records=JSON.parse(fs.readFileSync(path.join(dir,'labels.json'),'utf8').replace(/^\uFEFF/,''))
  .map((r,i)=>({...r,uid:folder+'/'+r.id+'/'+i,dataset:folder,path:path.join(dir,r.file)}));
 const base=path.join(ROOT,'dataset/TagClassifierDataset');
 for(const id of fs.readdirSync(base)){
  const dir=path.join(base,id),file=fs.readdirSync(dir).find(f=>f.endsWith('.osu'));if(!file)continue;
  const labelText=fs.readFileSync(path.join(dir,'PatternLabel.txt'),'utf8');
  records.push({uid:'manual/'+id,id,dataset:'manual',path:path.join(dir,file),tags:[],annotations:annotation(labelText)});
 }
 return records;
}
function scoreIntervals(annotations,segments,{trimMs=0}={}){
 let correct=0,total=0,uncovered=0;const confusion={},details=[];
 for(const a of annotations){
  if(!a.tags.length||a.end-a.start<=2*trimMs)continue;
  const start=a.start+trimMs,end=a.end-trimMs;
  total+=end-start;let covered=0;
  for(const s of segments){
   const d=Math.max(0,Math.min(end,s.end)-Math.max(start,s.start));if(!d)continue;
   covered+=d;const ok=a.tags.includes(s.tag);correct+=d*ok;
   if(!ok){const k=a.tags.join('|')+' -> '+s.tag;confusion[k]=(confusion[k]||0)+d;}
   details.push({start:Math.max(start,s.start),end:Math.min(end,s.end),truth:a.tags,predicted:s.tag,correct:ok});
  }
  if(covered>end-start+.001)throw new Error('Overlapping predicted segments');
  uncovered+=end-start-covered;
 }
 return {accuracy:correct/Math.max(total,1),correctMs:correct,labelledMs:total,uncoveredMs:uncovered,confusion,details};
}
module.exports={ROOT,annotation,records7,scoreIntervals};
