'use strict';
/** Research-only access to the COMPLETE archived pipeline. No archived edits. */
const fs=require('node:fs'),path=require('node:path');
const SOURCE=path.resolve(__dirname,'../../src/legacy/tag-engine.cjs');
const EXPOSE=['CONFIG','RC_DEFAULTS','LN_DEFAULTS','VEC_KEYS','MEDIAN_KEYS','MODEL',
 'parseOsu','rescale','buildRows','attachTails','buildColumnData','computeHeld','buildFeatureContext',
 'computeWindowFeatures','computeAllWindows','normalizeWindows','segmentMap','markBreaks',
 'computeSegmentExtras','computeMapMedians','medianWindowFeatures','buildVector','modelScores',
 'scoreRc','scoreLn','blendScores','classifySegments','smoothSegments','aggregateOverall'];
function replaceOnce(source,old,replacement){
 const i=source.indexOf(old);if(i<0||source.indexOf(old,i+old.length)>=0)throw new Error('Patch anchor not unique: '+old.slice(0,70));
 return source.slice(0,i)+replacement+source.slice(i+old.length);
}
function loadLab({patches=[],config={}}={}){
 let source=fs.readFileSync(SOURCE,'utf8').replace(/\r\n/g,'\n');
 for(const p of patches)source=p.apply(source,replaceOnce);
 const anchor='  return __toCommonJS(index_exports);';
 source=replaceOnce(source,anchor,`  Object.assign(index_exports,{lab:{${EXPOSE.join(',')}}});\n${anchor}`);
 const module={exports:{}};
 new Function('module','exports',source)(module,module.exports);
 const api=module.exports;Object.assign(api.lab.CONFIG,config);return api;
}
function context(lab,text,rate=1){
 let map=lab.parseOsu(text);if(rate!==1)map=lab.rescale(map,rate);
 const rows=lab.buildRows(map);lab.attachTails(map,rows);
 const cols=lab.buildColumnData(map),held=lab.computeHeld(map,rows),ctx=lab.buildFeatureContext(map,rows,held,cols);
 const windows=lab.computeAllWindows(ctx);return {map,ctx,rows,windows};
}
function inspect(api,text,{intervals,speedRate=1}={}){
 const lab=api.lab,{map,ctx,rows,windows}=context(lab,text,speedRate);
 const norm=lab.normalizeWindows(windows);
 const activeCut=windows.map(w=>w.cutDemand).filter(v=>v>0).sort((a,b)=>a-b);
 const cutDemandRef=activeCut.length?activeCut[Math.floor(activeCut.length*.8)]:1;
 const raw=lab.segmentMap({map,rows,windows,norm});
 const autoMarks=lab.markBreaks(map,rows,raw);
 const rawSegs=raw.map((s,i)=>({...s,family0:autoMarks[i].family}));
 const cm={windows,cutDemandRef,rawSegs,extras:rawSegs.map(s=>lab.computeSegmentExtras(ctx,s.start,s.end))};
 const segments=intervals?intervals.map(s=>({start:s.start,end:s.end,
  windowRange:s.windowRange??[Math.max(0,Math.floor((s.start-map.firstNote)/lab.CONFIG.winStepMs)),
    Math.min(windows.length,Math.max(1,Math.ceil((s.end-map.firstNote)/lab.CONFIG.winStepMs)))]})):cm.rawSegs;
 const marks=lab.markBreaks(map,rows,segments),mapMedians=lab.computeMapMedians(windows);
 const rowsOut=segments.map((s,i)=>{
  const extra=intervals?lab.computeSegmentExtras(ctx,s.start,s.end):cm.extras[i];
  const f=lab.medianWindowFeatures(windows,s.windowRange[0],s.windowRange[1],mapMedians);
  const vector=lab.buildVector(f,extra,cm.cutDemandRef);
  return {...s,f,extra,vector:lab.VEC_KEYS.map(k=>vector[k]),
   modelRc:lab.modelScores('rc',f,extra,cm.cutDemandRef),modelLn:lab.modelScores('ln',f,extra,cm.cutDemandRef),
   ruleRc:lab.scoreRc({f,extras:extra,cutDemandRef:cm.cutDemandRef}),ruleLn:lab.scoreLn({f,extras:extra}),
   family0:marks[i].family};
 });
 const cache={...cm,windows,rawSegs:segments.map((s,i)=>({...s,family0:marks[i].family})),extras:rowsOut.map(r=>r.extra)};
 const classified=lab.classifySegments(cache,{scorer:'model'});
 return {map,ctx,cutDemandRef:cm.cutDemandRef,windows,segments:rowsOut,classified,
   smoothed:lab.smoothSegments(structuredClone(classified)),featureKeys:lab.VEC_KEYS};
}
module.exports={loadLab,context,inspect,replaceOnce};
