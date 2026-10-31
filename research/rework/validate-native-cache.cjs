'use strict';
// Validate recorded producer code and inventory before reusing a native cache.
// Old repaired caches stay usable; a fresh extraction needs no r1 ancestor.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {ROOT}=require('./data.cjs'),physical=require('./physical.cjs'),base=require('../../src/features.cjs');
const directory=path.resolve(process.argv[2]||'.cache/rework-coarse-r2');
const read=p=>JSON.parse(fs.readFileSync(p,'utf8')),spec=read(path.join(directory,'spec.json')),inventory=read(path.join(directory,'inventory.json'));
assert.ok(['coarse-native-physical-r2','six-native-physical-r2'].includes(spec.id),'Expected corrected native representation');
assert.deepEqual(spec.keys,physical.KEYS);assert.deepEqual(spec.baseKeys,base.FEATURE_KEYS);
const normalized=s=>s.replace(/\r\n/g,'\n');
for(const name of['physical.cjs','anchor.cjs','release-geometry.cjs'])assert.equal(normalized(spec.sourceSnapshot[name]),normalized(fs.readFileSync(path.join(__dirname,name),'utf8')),name+' producer changed');
for(const name of['parser.cjs','features.cjs'])assert.equal(normalized(spec.coreSnapshot[name]),normalized(fs.readFileSync(path.join(ROOT,'src',name),'utf8')),name+' producer changed');
assert.equal(new Set(inventory.records.map(r=>r.uid)).size,inventory.records.length,'Repeated cache identity');
for(const r of inventory.records)assert.ok(fs.existsSync(path.join(directory,r.uid.replaceAll('/','__')+'.json')),r.uid+' missing');
if(spec.id.startsWith('coarse-'))assert.equal(inventory.errors.length,0);
else for(const error of inventory.errors)assert.ok(error.error==='InvalidHoldTail'&&error.uid.includes('bm6k_0778'),'Unexpected six-key rejection');
console.log('Corrected native cache producer/inventory valid:',directory,inventory.records.length,'records; no raw files uploaded');
