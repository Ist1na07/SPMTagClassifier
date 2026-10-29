// Requires the local, ignored upstream checkout and dataset. Compares values, not hashes.
const fs = require('node:fs'); const vm = require('node:vm'); const assert = require('node:assert/strict');
const p = require('node:path'); const extracted = require('../src/legacy/tag-engine.cjs');
const box = {}; vm.runInNewContext(fs.readFileSync('.upstream/tag_engine.js','utf8'),box);
const files = fs.readdirSync('dataset/spm_dataset-1.1.1/beatmaps').filter(x=>x.endsWith('.osu'));
let n=0;
for (const file of files) {
 const text = fs.readFileSync(p.join('dataset/spm_dataset-1.1.1/beatmaps',file),'utf8');
 for (const rate of [1,1.5]) {
 const a=extracted.analyzeMap(text,{speedRate:rate}); const b=box.TagClassifier.analyzeMap(text,{speedRate:rate});
 assert.equal(JSON.stringify(a),JSON.stringify(b),file+' @ '+rate); n++;
 }
}
console.log(JSON.stringify({maps:files.length,rates:[1,1.5],comparisons:n,exact:true}));
