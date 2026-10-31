'use strict';
// Shrink the segmentation metric, not the labels or expert outputs. The
// transformed-domain normalization changes the implicit weighting of four
// rate signals; its weight must be selected with the final pipeline.
const assert=require('node:assert/strict');
const {loadLab}=require('./legacy-lab.cjs');
const {TREATMENTS}=require('./native.cjs');
const patches=require('./patches.cjs');
const STRENGTHS=[0,.05,.2,1],RELEASES=[null,20,40,80];
function metricApi(releaseMs,strength){
 if(!RELEASES.includes(releaseMs)||!STRENGTHS.includes(strength))throw Error('Metric parameters outside the declared joint grid');
 const treatment=releaseMs===null?'logical_neutral':'logical_release_'+releaseMs;
 const api=loadLab({patches:TREATMENTS[treatment]});
 if(strength===0)return api;
 const corrected=loadLab({patches:TREATMENTS[treatment].concat(patches.transformedScale)});
 const old=api.lab.normalizeWindows,modern=corrected.lab.normalizeWindows;
 // inspectPrepared explicitly consumes this exported normalizer. The
 // archived analyzeMap closure is untouched and is NOT this research entry.
 api.lab.normalizeWindows=windows=>{
  const b=modern(windows);if(strength===1)return b;
  const a=old(windows);assert.deepEqual(a.featureNames,b.featureNames);
  return {featureNames:a.featureNames,values:a.values.map((row,i)=>Float64Array.from(row,(v,j)=>(1-strength)*v+strength*b.values[i][j]))};
 };
 return api;
}
module.exports={STRENGTHS,RELEASES,metricApi};
