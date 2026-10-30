'use strict';
const {parseOsu}=require('../../src/parser.cjs');
// Mirror physical columns, not raw X coordinates: 511-x and 512-x both fail
// at some lane boundaries. Canonical lane centres preserve column identity.
function mirrorText(text){
 const K=parseOsu(text).keys;let section='';
 return text.split(/\r?\n/).map(line=>{
  const trim=line.trim();if(trim.startsWith('[')){section=trim.toLowerCase();return line;}
  if(section!=='[hitobjects]'||!trim||trim.startsWith('//'))return line;
  const p=line.split(',');if(p.length<5)return line;
  const c=Math.min(K-1,Math.max(0,Math.floor(Number(p[0])*K/512)));
  p[0]=String(Math.floor((K-.5-c)*512/K));return p.join(',');
 }).join('\n');
}
module.exports={mirrorText};
