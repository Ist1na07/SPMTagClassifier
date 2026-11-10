'use strict';
const fs=require('node:fs');
const {prepare}=require('../../src/parser.cjs');
const guard=require('./static_cut_guard.cjs');
const output='.cache/local-feedback-static-cut-r1.jsonl',widths=[1000,2000,4000];
if(fs.existsSync(output))throw Error('Preserve completed no-cut extraction');
let maps=0,frames=0;
for(const line of fs.readFileSync('.cache/perception-inputs-r1.jsonl','utf8').trim().split('\n')){
  const input=JSON.parse(line),shared=guard.prepare(prepare(input.map)),children=[],masks=widths.map(()=>[]),alternation=widths.map(()=>[]);
  input.frames.forEach((f,parent)=>{
    for(let start=f.start;start<f.end;start+=500){const end=Math.min(start+500,f.end),mid=(start+end)/2;children.push({start,end,parent});widths.forEach((w,k)=>{const a=Math.max(input.map.firstNote,mid-w/2),b=Math.max(a+1,Math.min(input.map.lastNote,mid+w/2)),staticGrip=guard.certifiedStaticRice(shared,a,b);masks[k].push(staticGrip);alternation[k].push(staticGrip&&guard.alternatingHands(shared,a,b));});}
  });
  fs.appendFileSync(output,JSON.stringify({uid:input.uid,frames:children,masks,alternation})+'\n');maps++;frames+=children.length;
}
fs.writeFileSync(output+'.schema.json',JSON.stringify({widths,maps,frames,scorerCalls:0,meaning:'Exact absence of active-hand mask changes under both center-thumb assignments, with >=4head rows and no LN action/body in the window; a negative CS certificate only.',source:fs.readFileSync(__filename,'utf8'),predicate:fs.readFileSync(require.resolve('./static_cut_guard.cjs'),'utf8')},null,2)+'\n');console.log('static no-cut masks',maps,frames);
