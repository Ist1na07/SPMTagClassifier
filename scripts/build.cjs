'use strict';
// Small deterministic CommonJS-to-IIFE packer; no runtime/build dependencies.
const fs=require('node:fs'),path=require('node:path');const modules=[],ids=new Map();
function add(file) {
 file=path.resolve(file);if(ids.has(file))return ids.get(file);
 const id=modules.length;ids.set(file,id);modules.push(null);
 let source=fs.readFileSync(file,'utf8').replace(/\r\n/g,'\n');
 if(file.endsWith('.json'))source='module.exports='+source+';';
 else source=source.replace(/require\(['"](\.[^'"]*)['"]\)/g,(_,name)=>`load(${add(path.resolve(path.dirname(file),name))})`);
 modules[id]=`function(module,exports,load){\n${source}\n}`;return id;
}
const entry=add('src/index.cjs');fs.mkdirSync('dist',{recursive:true});
fs.writeFileSync('dist/tag-engine.js',`/* SPMTagClassifier v1.2.0 | MIT | Ist1na */\nvar TagClassifier=(()=>{\nconst modules=[${modules.join(',\n')}],cache={};\nfunction load(id){if(!cache[id]){const m=cache[id]={exports:{}};modules[id](m,m.exports,load);}return cache[id].exports;}\nreturn load(${entry});})();\nif(typeof module!=='undefined')module.exports=TagClassifier;\n`);
console.log('Built dist/tag-engine.js:',modules.length,'modules');
