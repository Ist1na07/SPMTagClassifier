#!/usr/bin/env node
'use strict';
const {plan,apply}=require('../integration/patch-overlay.cjs');
const args=process.argv.slice(2),directory=args.find(x=>!x.startsWith('--'));
if(!directory||args.some(x=>x.startsWith('--')&&x!=='--apply')){
 console.error('Usage: node bin/update-overlay.cjs <plugin-directory> [--apply]\nDefault: dry run. --apply backs up all three files first.');process.exit(2);
}
try{const p=plan(directory);console.log(JSON.stringify(args.includes('--apply')?apply(p):{dryRun:true,target:p.target,changed:p.changed},null,2));}
catch(e){console.error(e.message);process.exit(1);}
