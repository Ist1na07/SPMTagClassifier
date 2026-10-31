'use strict';
const test=require('node:test'),assert=require('node:assert/strict');
const {mirrorText}=require('./mirror.cjs'),{parseOsu}=require('../../src/parser.cjs');
for(const K of [4,6,7])test(K+'K physical mirror handles boundary X and is an event-level involution',()=>{
 const text=`[General]\nMode:3\n[Difficulty]\nCircleSize:${K}\n[HitObjects]\n`+[0,1,73,128,256,511,512].map((x,i)=>`${x},192,${i*250},1,0,0:0:0:0:`).join('\n');
 const a=parseOsu(text),b=parseOsu(mirrorText(text)),c=parseOsu(mirrorText(mirrorText(text)));
 a.notes.forEach((n,i)=>assert.equal(b.notes[i].col,K-1-n.col));assert.deepEqual(a.notes,c.notes);
});
