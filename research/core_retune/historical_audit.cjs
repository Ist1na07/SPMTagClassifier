'use strict';
// Read-only audit of the historical research, with isolated toy reproductions.
// Never imports its entry points: trainModel/trainGbdt write files on import.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {loadLab}=require('../rework/legacy-lab.cjs');
const root=process.argv[2]??'E:/Files/Developing/SPM/TagClassifier_Eta/tagclassifier';
const read=p=>fs.readFileSync(path.join(root,p),'utf8');
const train=read('scripts/trainModel.ts'),tune=read('scripts/tune.ts'),evalSeg=read('scripts/evalSeg.ts');
assert(train.includes('const epochs = 500;')&&train.includes('const lr = 0.08;')&&train.includes('const l2 = 5e-4;'));
assert(tune.includes('const cur = (rcP as Record<string, number>)[k];'));
assert(tune.includes('cur === undefined ? delete (rcP as Record<string, number>)[k]'));
assert(evalSeg.includes('for (const g of [...gtBreak, ...gtIgnore])'));
// The original closure captures cur once per coordinate, not once per trial.
// First trial improves, second fails: parameter resets to the pre-search value
// while best.score continues to describe the accepted first trial.
const params={x:0};let best=0;const cur=params.x;
for(const [v,score]of [[1,1],[2,-1]]){params.x=v;if(score>best+1e-5)best=score;else params.x=cur;}
assert.equal(params.x,0);assert.equal(best,1);
// Old denominator visits only GT Break/unknown intervals. The second second
// is a false Break inside scored pattern time and is therefore never counted.
const breakToy={correctBreakMs:1000,falseBreakOnScoredMs:1000,unknownMs:0};
const oldPrecision=breakToy.correctBreakMs/(breakToy.correctBreakMs+breakToy.unknownMs);
const correctPrecision=breakToy.correctBreakMs/(breakToy.correctBreakMs+breakToy.falseBreakOnScoredMs);
assert.equal(oldPrecision,1);assert.equal(correctPrecision,.5);
const current=loadLab().lab.MODEL,historical=JSON.parse(read('src/classify/model.generated.json')),models={};
for(const f of ['rc','ln']){
 models[f]={};for(const k of ['keys','classes'])models[f][k+'Match']=JSON.stringify(current[f][k])===JSON.stringify(historical[f][k]);
 for(const k of ['mean','std','W','b'])models[f][k]={count:current[f][k].length,maxAbsoluteDifference:Math.max(...current[f][k].map((v,i)=>Math.abs(v-historical[f][k][i])))};
}
const result={historicalRoot:root,mode:'read only; no historical training/import/build performed',
 training:{epochs:500,learningRate:.08,l2:.0005,hyperparameterSearchInTrainer:false,manualDuplication:[3,5],manualMajorityThreshold:.6,
   labels:'Single-tag whole-chart labels copied to every non-empty non-Break natural segment, plus six charts primary labels; standardization and class weights on those same training samples.',
   validation:'Trainer prints in-sample accuracy only; tuneSeg uses all six annotated charts for both knob selection and its reported objective.'},
 defects:{coordinateRollback:{parameterAfterTrials:params.x,recordedBestScore:best,parameterThatEarnedBestScore:1,source:'scripts/tune.ts'},
   breakPrecision:{toy:breakToy,reported:oldPrecision,correct:correctPrecision,source:'scripts/evalSeg.ts',effect:'Scored-time false Breaks omitted from precision denominator, so reported 100% is not a protection guarantee.'}},
 comparisonToArchivedModel:models,identity:'Same 56 keys and family classes, different scaler/weights; this is not a recovered exact training batch.',
 readmeEvidence:{source:'README.md',manualTrainAndTune:true,reportedCacheMismatch:true,oldOverallMetric:'Composite-aware three-slot partial credit, not the current literal multi-label metric.'}};
fs.mkdirSync('research/core_retune/results/historical-audit',{recursive:true});
fs.writeFileSync('research/core_retune/results/historical-audit/source-audit.json',JSON.stringify(result,null,2)+'\n');
console.log(JSON.stringify(result,null,2));
