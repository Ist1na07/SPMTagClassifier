'use strict';
const path=require('node:path'),{bundle}=require('../../scripts/bundle.cjs');
const {replaceOnce}=require('../rework/legacy-lab.cjs'),{compileRestEngine}=require('./compile-rest-engine.cjs');
function compileTargetedEngine(profile){
  let source=compileRestEngine(require('../../src/models/overlay7k.json').gate,profile.policy);
  source=replaceOnce(source,'  return __toCommonJS(index_exports);',
    '  Object.assign(index_exports,{native:{classifySegments,smoothSegments,aggregateOverall,parseOsu,rescale}});\n  return __toCommonJS(index_exports);');
  const entry=path.resolve(__dirname,'targeted-product-entry.cjs'),factory=path.resolve(__dirname,'targeted-archive.cjs');
  const code=`const {createConsumer}=require('./targeted-runtime.cjs');\nconst archive=require('./targeted-archive.cjs');\nmodule.exports=createConsumer(archive,${JSON.stringify(profile)});`;
  return bundle(entry,{overrides:new Map([[entry,code],[factory,source]])});
}
module.exports={compileTargetedEngine};
