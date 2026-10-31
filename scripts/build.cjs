'use strict';
// Dependency-free static browser build of the same actual library entry.
const fs=require('node:fs'),{bundle}=require('./bundle.cjs');
require('./build-coarse.cjs');
require('./build-ln.cjs');
const {version}=require('../package.json'),result=bundle('src/index.cjs');
fs.mkdirSync('dist',{recursive:true});
fs.writeFileSync('dist/tag-engine.js',`/* SPMTagClassifier v${version} | MIT | Ist1na */\nvar TagClassifier=${result.code};\nif(typeof module!=='undefined')module.exports=TagClassifier;\n`);
console.log('Built dist/tag-engine.js:',result.count,'modules');
