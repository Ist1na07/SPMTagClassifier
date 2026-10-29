const fs=require('node:fs'),path=require('node:path');
const {analyzeMap}=require('../src/index.cjs');
const {parseOsu}=require('../src/parser.cjs');
const {FINE_TO_COARSE}=require('../src/aggregate.cjs');
const norm=s=>s.toLowerCase().replace(/[^\p{L}\p{N}]+/gu,'');
function ids(map){const m=map.meta,out=['song:'+norm((m.Artist||'')+'|'+(m.Title||''))];
 for(const k of ['BeatmapID','BeatmapSetID'])if(Number(m[k])>0)out.push(k+':'+m[k]);return out;}
const known=new Set();
for(const file of fs.readdirSync('dataset/spm_dataset-1.1.1/beatmaps')){
 const map=parseOsu(fs.readFileSync(path.join('dataset/spm_dataset-1.1.1/beatmaps',file),'utf8'));ids(map).forEach(id=>known.add(id));
}
for(const folder of fs.readdirSync('dataset/TagClassifierDataset'))for(const file of fs.readdirSync(path.join('dataset/TagClassifierDataset',folder)))
 if(file.endsWith('.osu'))ids(parseOsu(fs.readFileSync(path.join('dataset/TagClassifierDataset',folder,file),'utf8'))).forEach(id=>known.add(id));
const root='dataset/spm_dataset_jack_apolar-1.0.0',records=JSON.parse(fs.readFileSync(path.join(root,'labels.json'),'utf8'));
const jackTags=new Set(['jack','chordjack','midjack','minijack','speedjack','dense chordjack','fulljack','anchor','anchorjack','longjack','dense cj','anchor jack','full jack','speedjack-','chordjack-']);
const rows=[];let stable=0,modern=0,eligible=0;
for(const r of records){
 const text=fs.readFileSync(path.join(root,r.file),'utf8'),map=parseOsu(text);
 const overlap=ids(map).some(id=>known.has(id)),positive=r.tags.some(t=>jackTags.has(t));
 const row={id:r.id,tags:r.tags,excluded:overlap?'song/ID overlaps primary or manual corpus':!positive?'no unambiguous jack annotation':null};
 if(!row.excluded){
  const a=analyzeMap(text),b=analyzeMap(text,{algorithm:'modern'});
  row.stableTags=a.overall.tags;row.modernTags=b.overall.tags;
  row.stableJack=a.overall.tags.some(t=>FINE_TO_COARSE[t]==='jack');row.modernJack=b.overall.tags.some(t=>FINE_TO_COARSE[t]==='jack');
  stable+=row.stableJack;modern+=row.modernJack;eligible++;
 }
 rows.push(row);
}
const result={total:records.length,eligible,stablePositiveRecall:eligible?stable/eligible:null,modernPositiveRecall:eligible?modern/eligible:null,
 caveat:'positive-only diagnostic, not specificity/accuracy; historical baseline exposure unknown',rows};
fs.writeFileSync('research/results/apolar-positive-diagnostic.json',JSON.stringify(result,null,2));console.log({eligible,stable,modern});
