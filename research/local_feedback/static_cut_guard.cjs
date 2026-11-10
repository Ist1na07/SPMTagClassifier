'use strict';
// A deliberately narrow negative certificate, not a new positive type model.
// Interleaving empty other-hand rows must not masquerade as finger changes.
const {lowerBound:lb}=require('../../src/parser.cjs');
const {prepareHandSequence}=require('../frontier/hand-sequence.cjs');
const {handTransitions}=require('../frontier/hand-transitions.cjs');
function prepare(ctx){return {ctx,sequence:prepareHandSequence(ctx)};}
function certifiedStaticRice(shared,start,end){
  const {ctx,sequence}=shared;if(ctx.map.keys!==7)return false;
  const a=lb(ctx.rows,start,r=>r.t),b=lb(ctx.rows,end,r=>r.t);
  // Reuse the established four-row minimum observation support. Do not
  // classify absence of movement from an empty/single-hit excerpt.
  if(b-a<4)return false;
  const t=lb(ctx.tails,start,r=>r.t);if(t<ctx.tails.length&&ctx.tails[t].t<end)return false;
  for(let j=a;j<b;j++)if(ctx.rows[j].lnMask)return false;
  for(const hs of ctx.holds){const j=lb(hs,end,h=>h[0])-1;if(j>=0&&hs[j][1]>start)return false;}
  const r=handTransitions(ctx,start,end,sequence);
  // Both fixed 7K center-thumb assignments must show exactly zero hand-mask
  // change. A subset/superset change is movement too; zero substitutions
  // alone would be an over-broad exclusion of possible chordstreams.
  return r.profiles.every(p=>p[1]===0);
}
function alternatingHands(shared,start,end){
  const {ctx}=shared,a=lb(ctx.rows,start,r=>r.t),b=lb(ctx.rows,end,r=>r.t);let previous=-1;
  if(b-a<4)return false;
  for(let j=a;j<b;j++){
    const mask=ctx.rows[j].mask;if(mask&(1<<3))return false;
    const l=mask&7,r=mask&112;if((!l&&!r)||(l&&r))return false;
    const side=l?0:1;if(side===previous)return false;previous=side;
  }
  return true;
}
module.exports={prepare,certifiedStaticRice,alternatingHands};
