'use strict';
// Static relative CommonJS -> IIFE compilation. Node APIs run at build time.
const fs=require('node:fs'),path=require('node:path');
function bundle(entry,{overrides=new Map()}={}){
 const modules=[],ids=new Map();
 function add(file){
  file=path.resolve(file);if(ids.has(file))return ids.get(file);
  const id=modules.length;ids.set(file,id);modules.push(null);
  let source=(overrides.get(file)??fs.readFileSync(file,'utf8')).replace(/\r\n/g,'\n');
  if(file.endsWith('.json'))source='module.exports='+source+';';
  else source=source.replace(/require\(['"](\.[^'"]*)['"]\)/g,(_,name)=>`load(${add(path.resolve(path.dirname(file),name))})`);
  if(/\brequire\s*\(/.test(source))throw Error('Bundle contains a non-static or non-relative runtime require: '+file);
  modules[id]=`function(module,exports,load){\n${source}\n}`;return id;
 }
 const id=add(entry);
 return {count:modules.length,code:`(()=>{\nconst modules=[${modules.join(',\n')}],cache={};\nfunction load(id){if(!cache[id]){const m=cache[id]={exports:{}};modules[id](m,m.exports,load);}return cache[id].exports;}\nreturn load(${id});})()`};
}
module.exports={bundle};
