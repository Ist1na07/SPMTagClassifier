'use strict';
// Text-only, version-checked patches. Never execute the target's source.
const fs=require('node:fs'),path=require('node:path');
const {EXPECTED_RATING,COMPACT_RATING}=require('../src/perception.cjs');
const {version}=require('../package.json');
const MARK='SPMTagClassifier perception integration schema 1';
const OLD_MARK='SPMTagClassifier perception integration v1.3.0';
const OLD_BRANCH=`                    const branch = seg.branches ? seg.branches[seg.tag] || 0 : seg.confidence;
                    elSegConf.textContent = Math.round(branch * 100) + '%';
                    elSegConf.title = seg.branches ? '分支贴合度（不是识别正确率）' : '原分类支持';`;
const BRANCH=`                    const hasBranch = seg.branches && Object.prototype.hasOwnProperty.call(seg.branches, seg.tag);
                    const branch = hasBranch ? seg.branches[seg.tag] : seg.confidence;
                    elSegConf.textContent = Math.round(branch * 100) + '%';
                    elSegConf.title = hasBranch ? '分支贴合度（不是识别正确率）' : '原分类支持（不是识别正确率）';`;
function normalizeMarker(s){return s.replace(OLD_MARK,MARK);}
function once(s,a,b){const i=s.indexOf(a);if(i<0||s.indexOf(a,i+a.length)>=0)throw Error('Unsupported overlay layout: '+a.slice(0,90));return s.slice(0,i)+b+s.slice(i+a.length);}
function numericObject(body){
 const result={};
 for(const entry of body.replace(/\/\/[^\n]*/g,'').split(',')){
  if(!entry.trim())continue;
  const m=entry.trim().match(/^([A-Za-z_][\w]*):\s*([-+\d.eE]+)$/);
  if(!m||!Number.isFinite(Number(m[2]))||Object.hasOwn(result,m[1]))throw Error('Unsupported rating parameter syntax');
  result[m[1]]=Number(m[2]);
 }
 return result;
}
function verifyParams(source){
 const d=source.match(/const SPM_V120_PARAMS = \{([\s\S]*?)\n\};/);
 const c=source.match(/const SPM_V120_COMPACT_PARAMS = Object.assign\(\{\}, SPM_V120_PARAMS, \{([\s\S]*?)\n\}\);/);
 if(!d||!c)throw Error('Unsupported rating parameter declarations');
 const detail=numericObject(d[1]),compact={...detail,...numericObject(c[1])};
 const expected=JSON.parse(fs.readFileSync(path.join(__dirname,'rating-parameters.json'),'utf8'));
 for(const [name,value]of Object.entries({detail,compact})){
  const ref=expected[name];
  if(Object.keys(value).length!==Object.keys(ref).length||Object.keys(ref).some(k=>value[k]!==ref[k]))throw Error('Unsupported rating parameters: '+name);
 }
}
function patchRating(text){
 let s=normalizeMarker(text.replace(/\r\n/g,'\n'));verifyParams(s);
 if(s.includes(MARK)){
  if(!s.includes('query: Array.from(res.b.t), demand: Array.from(res.D)')||!s.includes('options.includeTagEvidence'))throw Error('Partial rating integration: restore the three backed-up files first');
  return s;
 }
 s=once(s,'        firstNoteTime: times[0], lastNoteTime: times[times.length - 1],',
 `        // ${MARK}
        ...(options.includeTagEvidence ? {tagEvidence: {
            schemaVersion: 1, keyCount, speedRate, clock: 'post-rate-ms',
            modelIdentity: variant.id === 'compact' ? '${COMPACT_RATING}' : '${EXPECTED_RATING}',
            query: Array.from(res.b.t), demand: Array.from(res.D),
        }} : {}),
        firstNoteTime: times[0], lastNoteTime: times[times.length - 1],`);
 return s;
}
function patchHtml(text){
 let s=normalizeMarker(text.replace(/\r\n/g,'\n'));
 if(s.includes(MARK)){
  for(const token of ['function renderOverallTagsClassic','includeTagEvidence: cachedKeyCount === 7','ratingEvidence: result.tagEvidence','主要键型未定'])if(!s.includes(token))throw Error('Partial overlay integration: restore the three backed-up files first');
  return s.includes(OLD_BRANCH)?once(s,OLD_BRANCH,BRANCH):s;
 }
 const start=s.indexOf('            // v0.5.0: segment-based sort/tag classification (7K only)');
 const end=s.indexOf('            renderRating(speedRate);',start);
 if(start<0||end<0)throw Error('Unsupported overlay classification caller');
 s=s.slice(0,start)+`            // ${MARK}\n            cachedTagResult = null;\n\n`+s.slice(end);
 s=once(s,'classification is model independent and stays cached.','classification reuses the active model\'s raw demand field.');
 s=once(s,'            model: algorithmModel(),\n        });\n        if (result) {',
 `            model: algorithmModel(),
            includeTagEvidence: cachedKeyCount === 7,
        });
        cachedTagResult = null;
        if (result && cachedKeyCount === 7 && typeof TagClassifier !== 'undefined') {
            try {
                cachedTagResult = TagClassifier.analyzeMap(cachedOsuContent, {
                    speedRate, ratingEvidence: result.tagEvidence,
                });
            } catch (e) {
                console.error('SPM Map Analyser: perception classification failed:', e);
            }
        }
        if (result) {`);
 s=once(s,'    function renderOverallTags(tagRes) {','    function renderOverallTagsClassic(tagRes) {');
 s=once(s,"            if (result.classified) result.mapType = String(cachedTagResult.overall.sort || '').toUpperCase() || result.mapType;",
  "            if (result.classified && cachedTagResult.overall.tags.length) result.mapType = String(cachedTagResult.overall.sort || '').toUpperCase() || result.mapType;");
 s=once(s,'    /** Solid tag chips for the timeline legend. */',fs.readFileSync(path.join(__dirname,'render-perception.js'),'utf8')+'\n    /** Solid tag chips for the timeline legend. */');
 s=once(s,"                    elSegConf.textContent = Math.round(seg.confidence * 100) + '%';",
 BRANCH);
 return s;
}
function plan(directory){
 const target=path.resolve(directory),files=['index.html','spm_algorithm.js','tag_engine.js'];
 const originals=Object.fromEntries(files.map(f=>[f,fs.readFileSync(path.join(target,f),'utf8')]));
 const updated={'index.html':patchHtml(originals['index.html']),'spm_algorithm.js':patchRating(originals['spm_algorithm.js']),
  'tag_engine.js':fs.readFileSync(path.join(__dirname,'../dist/tag-engine.js'),'utf8')};
 return {target,originals,updated,changed:files.filter(f=>originals[f]!==updated[f])};
}
function apply(p){
 if(!p.changed.length)return {target:p.target,changed:[],backup:null};
 // Recheck after planning; never overwrite edits made during the dry run.
 for(const f of p.changed)if(fs.readFileSync(path.join(p.target,f),'utf8')!==p.originals[f])throw Error('Overlay changed since planning: '+f);
 const backup=path.join(p.target,'.spm-tag-backups',new Date().toISOString().replace(/[:.]/g,'-'));
 fs.mkdirSync(backup,{recursive:true});
 for(const f of Object.keys(p.originals))fs.writeFileSync(path.join(backup,f),p.originals[f]);
 fs.writeFileSync(path.join(backup,'manifest.json'),JSON.stringify({target:p.target,version,files:Object.keys(p.originals)},null,2)+'\n');
 try{for(const f of p.changed)fs.writeFileSync(path.join(p.target,f),p.updated[f]);}
 catch(e){for(const f of p.changed)fs.writeFileSync(path.join(p.target,f),p.originals[f]);throw e;}
 return {target:p.target,changed:p.changed,backup};
}
module.exports={patchHtml,patchRating,verifyParams,plan,apply};
