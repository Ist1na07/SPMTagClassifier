'use strict';
const anchor=require('./anchor.cjs'),release=require('./release-geometry.cjs');
const {lowerBound}=require('../../src/parser.cjs');
function describeInterval(ctx,start,end){
 const rows=ctx.rows.slice(lowerBound(ctx.rows,start,r=>r.t),lowerBound(ctx.rows,end,r=>r.t));
 const values=anchor.describe(rows,ctx.map.keys);
 return {start,end,anchor:Object.fromEntries(anchor.KEYS.map((k,i)=>[k,values[i]])),
  release:release.releaseGeometry(ctx,start,end).metrics};
}
module.exports={prepare:release.releaseContext,describeInterval};
