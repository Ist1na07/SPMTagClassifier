'use strict';
const {classifyCoarse}=require('./generated/coarse-engine.cjs');
const {version}=require('../package.json');
// Keep the established public source-bucket name. The internal morphology
// class is chordstream, not a claim about generic endurance or stamina.
const external=tag=>tag==='chordstream'?'stamina':tag;
const rename=object=>Object.fromEntries(Object.entries(object).map(([k,v])=>[external(k),v]));

function coarseResult(text,profile,options={}){
 const r=classifyCoarse(text,profile,options),dominant=r.coarse.dominant;
 if(![4,6].includes(r.map.keys))throw Error('Native coarse product path requires 4K or 6K');
 const composition=rename(r.coarse.composition);
 const segments=r.segments.map(s=>({...s,tag:external(s.tag),scores:rename(s.distribution),distribution:rename(s.distribution)}));
 const overall={tags:dominant?[external(dominant)]:[],sort:dominant===null?'unknown':dominant==='ln'?'ln':'rc',
  scores:composition,weights:composition,composition,familyMass:r.coarse.familyMass,
  scoreKind:r.coarse.scoreKind,compositionKind:'same local support weighted by duration × relative physical effort^a × concentration^b',
  taxonomy:'five morphology source classes; stamina is the compatibility name for chordstream',evidence:r.coarse.evidence};
 const descriptors=r.windows.map(w=>({start:w.start,end:w.end,anchor:w.metrics.regularAnchor,trill:w.metrics.trillFraction,
  roll:w.metrics.rollFraction,repetition:1-w.metrics.maskEntropy,occupancy:w.metrics.occupancy,
  releaseIndependence:w.metrics['release:headIsolation'],releaseHeldInteraction:w.metrics['release:independentHeldRelease']}));
 return {map:r.map,segments,overall,coarse:r.coarse,debug:{windows:r.windows,norm:undefined,ctx:undefined},descriptors,
  modelInfo:{...r.modelInfo,version,keyCount:r.map.keys,classes:[...profile.model.classes.map(external),'break'],featureKeys:profile.model.keys,variant:r.map.keys===4?'native-4k-coarse':'native-shared-coarse',
   segmentEvidence:r.map.keys===4?'weak 4K source buckets plus sufficient structural templates':'4K/7K shared coarse extrapolation; no human 6K pattern accuracy',
   temporalResolutionMs:2000,localScoreKind:'within-head softmax support or deterministic structural support, not calibrated correctness',
   overallDecision:'dominant weighted morphology; no independent whole-map head',sortMeaning:'family of dominant coarse class',
   supervision:r.map.keys===4?'4K-only weak supervision and selection':'4K/7K weak labels plus available 7K interval labels'}};
}
module.exports={coarseResult};
