'use strict';
// Recover exact pre-late-gate state from cached native cm; no re-extraction.
const fs=require('node:fs'),zlib=require('node:zlib');
const {loadLab}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const OUT='.cache/core-retune-parents-r2.jsonl.gz';
if(fs.existsSync(OUT))throw Error('Preserve prepared input');
const api=loadLab({patches:[actionRatePatch(profile.gate),{apply(s,r){return r(s,'if (easyAmbiguous || shortSparse) {','if (false && (easyAmbiguous || shortSparse)) {');}}]});
const rows=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r1.jsonl.gz')).toString().trim().split('\n').map(JSON.parse);
let late=0;
for(const row of rows){const frames=api.lab.classifySegments(row.cm,{scorer:'model'});frames.forEach((f,j)=>{row.parents[j].preLate=f;late+=f.family!==row.parents[j].baseline.family;});}
fs.writeFileSync(OUT,zlib.gzipSync(rows.map(r=>JSON.stringify(r)).join('\n')+'\n'));
fs.writeFileSync(OUT+'.schema.json',JSON.stringify({records:rows.length,lateBreakParents:late,source:fs.readFileSync(__filename,'utf8')},null,2)+'\n');
console.log('Prepared exact late-gate input; late Break parents',late);
