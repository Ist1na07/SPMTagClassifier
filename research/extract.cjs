'use strict';
// Feature extraction is shared verbatim with inference. No Python reimplementation.
const fs = require('node:fs'), path = require('node:path');
const {parseOsu,prepare} = require('../src/parser.cjs');
const {features,windows,FEATURE_KEYS,GROUPS,quantile} = require('../src/features.cjs');
const legacy = require('../src/legacy/tag-engine.cjs');
const root = process.argv[2] || 'dataset';
fs.mkdirSync('.cache',{recursive:true}); fs.mkdirSync('research/results',{recursive:true});
const output=process.env.FEATURE_OUTPUT||'.cache/features.jsonl';
const merge=Number(process.env.ROW_MERGE_MS??10);
const out = fs.openSync(output,'w'), signatures = new Map(), inventory = [], errors = [];
const round = x => JSON.parse(JSON.stringify(x,(_,v)=>typeof v==='number'?Math.round(v*1e7)/1e7:v));
function save(record,text) {
  try {
    const map = parseOsu(text);
    if(process.env.LABELLED_ONLY && (map.keys===6 || record.dataset.includes('apolar')))return;
    const ctx = prepare(map,merge);
    // Direct canonical event strings, not file hashes. Uniform rate variants share identity.
    const scale = quantile(ctx.rows.slice(1).map((r,i)=>r.t-ctx.rows[i].t).filter(x=>x>0),.5)||1;
    const sig = map.keys+'|'+map.notes.map(n=>[Math.round((n.t-map.firstNote)/scale*10),n.col,n.lnEnd===null?'':Math.round((n.lnEnd-n.t)/scale*10)].join(',')).join(';');
    const duplicateOf = signatures.get(sig) || null; if (!duplicateOf) signatures.set(sig,record.uid);
    const configs = {};
    for (const win of [2000,4000,8000]) configs[win] = windows(ctx,win,2000);
    const global = features(ctx,map.firstNote,Math.max(map.lastNote+1,map.firstNote+1));
    const result = {...record,keys:map.keys,meta:map.meta,first:map.firstNote,last:map.lastNote,
      duplicateOf,global,windows:configs};
    if (map.keys===7 && !process.env.SKIP_LEGACY) {
      const b = legacy.analyzeMap(text); result.legacy={segments:b.segments,overall:b.overall};
    }
    if (record.annotations) result.annotatedFeatures=record.annotations.map(a=>({...a,feature:features(ctx,a.start,a.end)}));
    fs.writeSync(out,JSON.stringify(round(result))+'\n');
    inventory.push({uid:record.uid,keys:map.keys,duplicateOf,notes:map.notes.length,windows:configs[4000].length});
    if (inventory.length%100===0) console.log('extracted',inventory.length);
  } catch(e) { errors.push({uid:record.uid,error:e.message}); }
}
for (const folder of fs.readdirSync(root)) {
  if(process.env.LABELLED_ONLY && (folder.includes('6k')||folder.includes('apolar')))continue;
  const dir = path.join(root,folder), labels = path.join(dir,'labels.json');
  if (!fs.existsSync(labels)) continue;
  const data = JSON.parse(fs.readFileSync(labels,'utf8').replace(/^\uFEFF/,''));
  for (let i=0;i<data.length;i++) {
    const r=data[i], file=path.join(dir,r.file);
    save({...r,uid:folder+'/'+r.id+'/'+i,dataset:folder},fs.readFileSync(file,'utf8'));
  }
}
const ms = s => { const a=s.split(':').map(Number); return a[0]*60000+a[1]*1000+a[2]; };
for (const folder of fs.readdirSync(path.join(root,'TagClassifierDataset'))) {
  const dir=path.join(root,'TagClassifierDataset',folder), files=fs.readdirSync(dir), osu=files.find(f=>f.endsWith('.osu'));
  if (!osu) continue;
  let cursor=null; const annotations=[];
  for (const raw of fs.readFileSync(path.join(dir,'PatternLabel.txt'),'utf8').split(/\r?\n/)) {
    const line=raw.split('#')[0].trim();
    const m=line.match(/^(?:(\d+:\d+:\d+)\s*)?-\s*(\d+:\d+:\d+)\s+(.+)$/);
    if (!m) continue;
    const start=m[1]?ms(m[1]):cursor,end=ms(m[2]);cursor=end;
    const tags=m[3].toLowerCase().split(',').map(s=>s.trim()).filter(s=>s!=='?');
    if (start!==null && end>start && tags.length) annotations.push({start,end,tags});
  }
  save({uid:'manual/'+folder,dataset:'manual',id:folder,tags:[],annotations},fs.readFileSync(path.join(dir,osu),'utf8'));
}
fs.closeSync(out);
fs.writeFileSync('.cache/feature-schema.json',JSON.stringify({FEATURE_KEYS,GROUPS}));
fs.writeFileSync(process.env.INVENTORY_OUTPUT||'research/results/inventory.json',JSON.stringify({rowMergeMs:merge,charts:inventory,errors},null,2));
console.log('Finished',inventory.length,'errors',errors.length); if(errors.length) process.exitCode=1;
