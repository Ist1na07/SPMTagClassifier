'use strict';
const fs=require('node:fs'),zlib=require('node:zlib'),assert=require('node:assert/strict');
const {loadLab}=require('../rework/legacy-lab.cjs');
const {actionRatePatch}=require('../overlay-quality/break-regularity.cjs');
const profile=require('../../src/models/overlay7k.json');
const expected=JSON.parse(fs.readFileSync('.cache/core-retune-native-expected.json','utf8'));
const rows=zlib.gunzipSync(fs.readFileSync('.cache/core-retune-parents-r1.jsonl.gz')).toString().trim().split('\n').map(JSON.parse);
const api=loadLab({patches:[actionRatePatch(profile.gate)]});Object.assign(api.lab.MODEL,expected.models);
let j=0,error=0;
for(const r of rows)for(const actual of api.lab.classifySegments(r.cm,{scorer:'model'})){
 const e=expected.frames[j++];assert.equal(actual.tag,e.tag);assert.equal(actual.family,e.family);assert.equal(actual.secondary??null,e.secondary??null);
 for(const[k,v]of Object.entries(actual.scores??{})){error=Math.max(error,Math.abs(v-e.scores[k]));assert(Math.abs(v-e.scores[k])<1e-10);}
}
assert.equal(j,expected.frames.length);const receipt={scope:'Nonzero directly refitted original W/b, actual archived classifier with current rest policy versus research complete classification including late gate',parents:j,maxScoreError:error};fs.writeFileSync('research/core_retune/results/native-refit-parity.json',JSON.stringify(receipt,null,2)+'\n');console.log(receipt);
