'use strict';
// Pure JS: reuses an existing curve; never invokes a difficulty scorer.
const {prepare}=require('./parser.cjs');
const {features}=require('./features.cjs');
const EXPECTED_RATING='5b0ddc61be53f5107f3b4dfa26a8397aa515e316c16b244878d1fc7039d0ba8f';
const COMPACT_RATING='55216a3abcf6653c4c78b2082c730d64a0ed4bc476d3e477c4789a39f309fb4e';
const PARTS=['Rbar','Pbar','Cbar','Jm'];
const RC=new Set(['anchor','chordjack','minijack','dense chordstream','fast chordstream','speed','tech','vibro']);
const SLOTS=['ordinary','rc mix','ln mix','hybrid'];
const sigmoid=x=>1/(1+Math.exp(-Math.max(-700,Math.min(700,x))));
const logit=x=>Math.log(Math.max(x,1e-12))-Math.log(Math.max(1-x,1e-12));
const sum=a=>a.reduce((s,x)=>s+x,0);
function finiteArray(a){return a!=null&&typeof a.length==='number'&&Array.from(a).every(Number.isFinite);}
function validateRating(map,rating,{speedRate=1,components=false,modelIdentity=EXPECTED_RATING}={}){
  if(!rating)throw Error('PerceptionMissingRating: provide an already-computed matching curve');
  if(rating.schemaVersion!==1||rating.clock!=='post-rate-ms'||rating.keyCount!==map.keys||rating.speedRate!==speedRate)
    throw Error('PerceptionRatingClock: schema, native keys and speedRate must match the analyzed map');
  if(rating.modelIdentity!==modelIdentity)throw Error('PerceptionRatingModel: curve does not match the selected SPMRating v1.2.0 profile');
  const q=rating.query,heads=[...new Set(map.notes.map(n=>n.t))];
  if(!finiteArray(q)||!q.length||q.length!==heads.length||Array.from(q).some((t,i)=>t!==heads[i]))
    throw Error('PerceptionRatingClock: curve query must equal this map\'s physical head-row clock');
  const channels=['demand',...(components?PARTS:[])];
  for(const k of channels){const a=k==='demand'?rating.demand:rating.components?.[k];
    if(!finiteArray(a)||a.length!==q.length||Array.from(a).some(x=>x<0))throw Error('PerceptionRatingChannel: finite nonnegative '+k+' on the same query required');}
  return rating;
}
function baseSupport(segments,classes,power=1,hard=false){
  return segments.map(s=>{
    if(s.family==='break')return classes.map(()=>0);
    if(hard)return classes.map(c=>Number(c===s.tag));
    const scores=s.scores??{[s.tag]:1},div=Math.max(1,...Object.values(scores));
    return classes.map(c=>Math.min(.999,Math.max(0,scores[c]??0)/div)**power);
  });
}
function integrate(map,segments,rating,recipe,{speedRate=1,components=false,modelIdentity=EXPECTED_RATING,geometry,context,includeGeometry=true}={}){
  if(map.keys!==7)throw Error('PerceptionUnsupportedKeys: fitted profiles currently require native 7K');
  if(!Array.isArray(segments)||segments.some((s,i)=>!Number.isFinite(s.start)||!Number.isFinite(s.end)||s.end<=s.start||(i&&s.start<segments[i-1].end)))
    throw Error('PerceptionIntervals: finite ordered nonoverlapping raw intervals required');
  if(!Number.isFinite(speedRate)||speedRate<=0)throw Error('PerceptionInvalidRate');
  const {alpha,beta,width,hold}=recipe;
  if(![alpha,beta,width,hold].every(Number.isFinite)||alpha<0||beta<0||width<=0||hold<0)throw Error('PerceptionInvalidRecipe');
  if(alpha>0||components)validateRating(map,rating,{speedRate,components,modelIdentity});
  if(context&&context.map!==map)throw Error('PerceptionContext: reuse the context belonging to this exact map');
  const start=map.firstNote,end=Math.max(start+1,map.lastNote),ctx=context??prepare(map),events=[],points=[start,end];
  function impulse(t){events.push([t-width/2,1000/width/map.keys],[t+width/2,-1000/width/map.keys]);}
  for(const n of map.notes)impulse(n.t);
  for(const hs of ctx.holds)for(const [a,b]of hs){impulse(b);events.push([a,hold/map.keys],[b,-hold/map.keys]);}
  for(const [t]of events)points.push(t);
  for(const s of segments)points.push(s.start,s.end);
  if(rating&&(alpha>0||components))for(const t of rating.query)points.push(t);
  events.sort((a,b)=>a[0]-b[0]);
  const grid=[...new Set(points.map(t=>Math.max(start,Math.min(end,t))))].sort((a,b)=>a-b),atoms=[];
  let event=0,action=0,frame=-1,query=0,meanD=0,meanA=0,totalDuration=0;
  for(let i=0;i<grid.length-1;i++){
    const a=grid[i],b=grid[i+1],dt=b-a;
    while(event<events.length&&events[event][0]<=a)action+=events[event++][1];
    while(frame+1<segments.length&&segments[frame+1].start<=a)frame++;
    let fi=frame>=0&&a<segments[frame].end?frame:-1;
    if(rating&&(alpha>0||components))while(query+1<rating.query.length&&rating.query[query+1]<=a)query++;
    const ld=alpha>0||components?Math.log1p(rating.demand[query]):0,la=Math.log1p(Math.max(0,action));
    const ps=components?PARTS.map(k=>Math.log1p(rating.components[k][query])):[0,0,0,0];
    const x=[ld,la,ps[0]-ps[1],ps[2]-ps[1],ps[3]-ps[1],ld];
    const weight=dt*(alpha?ld**alpha:1)*(beta?la**beta:1);
    atoms.push({start:a,end:b,frame:fi,weight,x});meanD+=dt*ld;meanA+=dt*la;totalDuration+=dt;
  }
  meanD/=Math.max(1,totalDuration);meanA/=Math.max(1,totalDuration);
  const n=segments.length,mass=Array(n).fill(0),x=Array.from({length:n},()=>Array(8).fill(0)),total=sum(atoms.map(a=>a.weight));
  for(const a of atoms){a.x[0]-=meanD;a.x[1]-=meanA;if(a.frame<0)continue;mass[a.frame]+=a.weight;
    for(let j=0;j<6;j++)x[a.frame][j]+=a.weight*a.x[j];}
  geometry??=includeGeometry?segments.map(s=>features(ctx,s.start,Math.max(s.start+1,s.end)).metrics):segments.map(()=>({releaseIndependence:0,crossCut:0}));
  for(let i=0;i<n;i++){
    for(let j=0;j<6;j++)x[i][j]=mass[i]>0?x[i][j]/mass[i]:0;
    x[i][6]=geometry[i].releaseIndependence;x[i][7]=geometry[i].crossCut;mass[i]/=Math.max(total,1e-30);
  }
  return {mass,x,total,atoms,endpointExtensionMs:rating&&(alpha>0||components)?Math.max(0,end-rating.query.at(-1)):null};
}
function standard(raw,scaler){return [1,...raw.map((v,j)=>scaler.keep[j]?(v-scaler.mean[j])/scaler.scale[j]:0)];}
function pooled(segments,classes,mass,support,x,components){
  const share=classes.map(()=>0),moments=classes.map(()=>Array(8).fill(0)),family=[0,0];
  for(let i=0;i<segments.length;i++){
    const f=segments[i].family;if(f==='rc')family[0]+=mass[i];if(f==='ln')family[1]+=mass[i];
    for(let j=0;j<classes.length;j++){const a=mass[i]*support[i][j];share[j]+=a;
      if(components)for(let k=0;k<8;k++)moments[j][k]+=a*x[i][k];}
  }
  const fineRaw=classes.map((_,j)=>[logit(share[j]),...([5,2,3,4].map(k=>components&&share[j]>0?moments[j][k]/share[j]:0)),0,0,0]);
  return {share,family,fineRaw};
}
function mixFeatures(classes,share,family,kind){
  const active=sum(family),out=[active>0?family[1]/active:0,active],available=[true];
  for(const rc of [true,false]){
    const p=share.filter((_,i)=>RC.has(classes[i])===rc),total=sum(p),q=p.map(x=>total>0?x/total:0);
    const entropy=-sum(q.map(v=>v>0?v*Math.log(v):0))/Math.log(p.length);
    if(kind==='distinct'){
      const second=sum(q.map(v=>v*v)),third=sum(q.map(v=>v*v*v));
      out.push(entropy,total>0?1-second:0,total>0?1-3*second+2*third:0);
    }else{const top=[...q].sort((a,b)=>b-a);out.push(entropy,top[1]??0,top[2]??0);}
    available.push(family[rc?0:1]>0&&p.filter(x=>x>0).length>=2);
  }
  available.push(family.every(x=>x>0));return {values:out,available};
}
function infer(map,segments,rating,profile,options={}){
  if(profile.schemaVersion!==1||profile.classes.length!==14)throw Error('PerceptionProfileSchema');
  const classes=profile.classes,needComponents=!!profile.components;
  const integral=integrate(map,segments,rating,profile.recipe,{...options,modelIdentity:profile.modelIdentity??EXPECTED_RATING,components:needComponents,includeGeometry:!!profile.localCoef});
  // A negative physical exclusion must not renormalize and inflate the other
  // branches when it removes the old maximum-scoring class.
  const base=options._branchSupport??baseSupport(segments,classes,profile.supportPower??1,profile.hard);
  if(base.length!==segments.length||base.some(row=>row.length!==classes.length||row.some(v=>!Number.isFinite(v)||v<0||v>1)))throw Error('PerceptionBranchSupport');
  const support=base.map(a=>[...a]);
  if(profile.localCoef){const s=profile.localScaler;
    for(let i=0;i<support.length;i++)for(let j=0;j<classes.length;j++){
      if(base[i][j]<=0)continue;let residual=0;
      for(let k=0;k<8;k++)if(s.keep[j][k])residual+=(integral.x[i][k]-s.mean[j][k])/s.scale[j][k]*profile.localCoef[j][k];
      support[i][j]=sigmoid(logit(base[i][j])+residual);
    }
  }
  const p=pooled(segments,classes,integral.mass,support,integral.x,needComponents);let fine=[],slots=[],tags=[];
  if(profile.kind==='anchored'){
    if(!options.baseline||!Array.isArray(options.baseline.tags))throw Error('PerceptionPrior: matching stable v1.2.2 whole-tag prior required');
    const old=options.baseline.tags,policy=profile.policy,family=profile.family;
    const strong=classes.filter((_,j)=>p.share[j]>=policy.mixOverride),mixed=old.some(c=>SLOTS.includes(c)&&c!=='ordinary');
    if(family.enabled&&p.family[0]>=family.minRc&&p.family[1]>=family.minLn)tags=['hybrid'];
    else if(mixed)tags=strong.length?strong:[...old];
    else tags=classes.filter((c,j)=>(old.includes(c)&&p.share[j]>=policy.keepFloor)||p.share[j]>=policy.addFloor);
  }else{
    for(let j=0;j<classes.length;j++){
      const scaler={mean:profile.fine.scaler.mean[j],scale:profile.fine.scaler.scale[j],keep:profile.fine.scaler.keep[j]},v=standard(p.fineRaw[j],scaler);
      fine.push(p.share[j]>0?sigmoid(sum(v.map((x,k)=>x*profile.fine.coef[j][k]))):0);
    }
    const mf=mixFeatures(classes,p.share,p.family,profile.mixKind),sv=standard(mf.values,profile.slot.scaler);
    let z=[0,...profile.slot.coef.map(row=>sum(row.map((x,j)=>x*sv[j])))].map((v,j)=>mf.available[j]?v:-1e6);
    const peak=Math.max(...z);z=z.map(v=>Math.exp(v-peak));const den=sum(z);slots=z.map(v=>v/den);const slot=slots.indexOf(Math.max(...slots));
    tags=slot?[SLOTS[slot]]:classes.filter((_,j)=>fine[j]>=.5);
    if(!tags.length&&profile.fallback!==false&&Math.max(...fine)>0)tags=[classes[fine.indexOf(Math.max(...fine))]];
  }
  const weights=Object.fromEntries(classes.map((c,j)=>[c,p.share[j]*integral.total/1000]));
  const percentages=Object.fromEntries(classes.map((c,j)=>[c,100*p.share[j]]));
  const labelPercentages=Object.fromEntries(tags.map(c=>[c,c==='hybrid'?100*sum(p.family):c==='rc mix'?100*p.family[0]:c==='ln mix'?100*p.family[1]:percentages[c]]));
  const totalShare=sum(p.share),composition=Object.fromEntries(classes.map((c,j)=>[c,totalShare>0?p.share[j]/totalShare:0]));
  // Rank fine tags by their actual burden share, not alphabetical class order.
  const prior=options.baseline?.tags??[];
  tags.sort((a,b)=>(percentages[b]??0)-(percentages[a]??0)||(prior.includes(a)&&prior.includes(b)?prior.indexOf(a)-prior.indexOf(b):classes.indexOf(a)-classes.indexOf(b)));
  // Display family follows the selected taxonomy, not a second untuned ratio gate.
  const hasRC=tags.some(c=>RC.has(c)||c==='rc mix');
  const hasLN=tags.some(c=>(classes.includes(c)&&!RC.has(c))||c==='ln mix');
  const sort=tags.includes('hybrid')||(hasRC&&hasLN)?'hb':hasRC?'rc':hasLN?'ln':options.baseline?.sort??'hb';
  return {overall:{tags,sort,weights,composition,percentages,labelPercentages,familyMass:{rc:p.family[0],ln:p.family[1]},
    scoreKind:'integrated prototype support times perceptual burden; not correctness probability',
    percentageKind:'overlapping support-weighted burden / all-chart burden; mixture labels cover the associated family union',
    perception:{difficultyAvailable:!!rating&&(profile.recipe.alpha>0||needComponents),componentsAvailable:needComponents,burdenIntegral:integral.total/1000,
      coveredBurdenShare:sum(integral.mass),ratingEndpointExtensionMs:integral.endpointExtensionMs}},
    support,mass:integral.mass,x:integral.x,share:p.share,fine,slots,integral};
}
function decorateSegments(segments,analysis,classes){
  const sums=segments.map(()=>classes.map(()=>0)),weights=segments.map(()=>0);let index=0;
  for(const a of analysis.integral.atoms){
    while(index<segments.length&&segments[index].end<=a.start)index++;
    for(let k=index;k<segments.length&&segments[k].start<a.end;k++){
      const overlap=Math.max(0,Math.min(a.end,segments[k].end)-Math.max(a.start,segments[k].start));
      if(!overlap)continue;const w=a.weight*overlap/(a.end-a.start);weights[k]+=w;
      if(a.frame>=0)for(let j=0;j<classes.length;j++)sums[k][j]+=w*analysis.support[a.frame][j];
    }
  }
  segments.forEach((s,i)=>{
    s.branches=Object.fromEntries(classes.map((c,j)=>[c,weights[i]>0?sums[i][j]/weights[i]:0]));
    s.aggregationWeight=weights[i]/Math.max(analysis.integral.total,1e-30);
  });
  return segments;
}
function ratingEvidenceFromSPM(result,components){
  if(result?.scope!=='whole')throw Error('PerceptionRatingScope: reuse the whole-map include_curve result, not a cropped scalar');
  const evidence={schemaVersion:1,keyCount:result.key_count,speedRate:result.rate,clock:'post-rate-ms',modelIdentity:result.model_identity,
    query:Array.from(result.query??[]),demand:Array.from(result.field??[])};
  if(!finiteArray(evidence.query)||!evidence.query.length||!finiteArray(evidence.demand)||evidence.demand.length!==evidence.query.length)
    throw Error('PerceptionRatingChannel: rate with include_curve=True and retain query/field');
  if(components!==undefined){
    evidence.components={};
    for(const k of PARTS){if(!finiteArray(components[k])||components[k].length!==evidence.query.length)throw Error('PerceptionRatingChannel: provide Rbar/Pbar/Cbar/Jm together on the public query');evidence.components[k]=Array.from(components[k]);}
  }
  return evidence;
}
module.exports={EXPECTED_RATING,COMPACT_RATING,validateRating,baseSupport,integrate,pooled,mixFeatures,infer,ratingEvidenceFromSPM,decorateSegments};
