'use strict';
const fs=require('node:fs');
const {records7}=require('./data.cjs');
const {parseOsu,prepare}=require('../../src/parser.cjs');
const {KEYS,releaseContext,releaseGeometry}=require('./release-geometry.cjs');
const paths=new Map(records7().map(r=>[r.uid,r.path]));
const records=fs.readFileSync('.cache/rework7.jsonl','utf8').trim().split('\n').map(JSON.parse);
const output='.cache/rework-release-geometry-r2.jsonl';
const previousPath='.cache/rework-release-geometry.jsonl';
const previous=fs.existsSync(previousPath)?new Map(fs.readFileSync(previousPath,'utf8').trim().split('\n').map(s=>{const r=JSON.parse(s);return[r.uid,r.features];})):null;
const fd=fs.openSync(output,'w');let count=0,changed=0;
for(const r of records){
 const ctx=releaseContext(prepare(parseOsu(fs.readFileSync(paths.get(r.uid),'utf8'))));
 const features=r.segments.map(s=>releaseGeometry(ctx,s.start,s.end).values);
 const before=previous?.get(r.uid);if(previous&&before.length!==features.length)throw Error('Release repair partition differs');
 if(before)for(let i=0;i<features.length;i++)for(let j=0;j<KEYS.length;j++){
  if(KEYS[j]==='sameHandOrder'){if(features[i][j]!==before[i][j])changed++;}
  else if(features[i][j]!==before[i][j])throw Error('Unexpected release coordinate change: '+KEYS[j]);
 }
 fs.writeSync(fd,JSON.stringify({uid:r.uid,features})+'\n');
 if(++count%50===0)console.log('release geometry',count);
}
fs.closeSync(fd);fs.writeFileSync('.cache/rework-release-geometry-r2-schema.json',JSON.stringify({
 features:KEYS,partition:'same fixed original RAW automatic boundaries as rework7.jsonl',
 semantics:'rate in tails/s/key; all other values timing-normalized morphology, not class probabilities',
 neighbourContext:'nearest heads/tails and pre-release holds use the complete chart; release triple assigned to its middle tail',
 scale:'active uninherited beatLength/4; if timing absent, median positive gap of all unique event times',
 keyGeometry:'all denominators use native active K; 7K centre is flexible and excluded from fixed-hand triples; all three timestamp groups must be singletons',
 repair:{parent:previous?previousPath:null,changedSegmentCoordinates:previous?changed:null,allOtherCoordinatesBitwiseEqual:previous?true:null},
 sourceSnapshot:Object.fromEntries(['release-extract.cjs','release-geometry.cjs'].map(n=>[n,fs.readFileSync(require('node:path').join(__dirname,n),'utf8')]))})+'\n');
console.log('release geometry repair complete',count,'changed segments',changed);
