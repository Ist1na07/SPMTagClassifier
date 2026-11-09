"""Exact original smoothing/aggregation for the restricted local experiment.

The post grid is a Cartesian comparison of three existing coordinates; other
original behavior stays fixed for attribution, not claimed fully re-tuned.
"""
import itertools
import numpy as np
from rc_action_model import redistribute,TRIAD,INDICES,mil,read

SCHEMA=read('.cache/frontier-rc-frames-r1.jsonl.schema.json')
NON_AGG=set(SCHEMA['nonAggregated'])
POST=[{'dominantShare':d,'mixShare':m,'secondRel':r}for d,m,r in itertools.product(
    [.45,.52,.55,.65,.75,.85,.95],[.05,.07,.1,.15,.2,.25,.3],[.4,.6,.75,.9,1.])]


def frames(world,uid,model=None):
    original=world.frames[uid]['frames']
    if model is None:return [dict(f)for f in original]
    p,unchanged=redistribute(world,uid,model);out=[]
    for j,f in enumerate(original):
        if unchanged[j]:out.append(dict(f));continue
        scores=dict(f['scores']);total=sum(scores.values())
        for c,index in zip(TRIAD,INDICES):scores[c]=float(p[j,index]*total)
        ranked=sorted(scores.items(),key=lambda r:-r[1]);tag,confidence=ranked[0]
        if tag in ['wildcard','ln wildcard']:
            for other,value in ranked[1:]:
                if other not in ['wildcard','ln wildcard']and value>.3:tag,confidence=other,value;break
        secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
        out.append({**f,'tag':tag,'secondary':secondary,'confidence':min(1.,confidence),'scores':scores})
    return out


def smooth(source):
    minimum=SCHEMA['minimumSegmentMs']*.6;out=[]
    for f in source:
        seg=dict(f)
        if out and out[-1]['tag']==seg['tag']and out[-1]['secondary']==seg['secondary']and out[-1]['family']==seg['family']and seg['end']-seg['start']<minimum:out[-1]['end']=seg['end']
        else:out.append(seg)
    result=[]
    for j,seg in enumerate(out):
        previous=result[-1]if result else None;next=out[j+1]if j+1<len(out)else None
        if seg['family']=='break'and seg['end']-seg['start']<minimum and previous is not None and next is not None and next['tag']==previous['tag']:previous['end']=seg['end']
        else:result.append(seg)
    return result


def sufficient(source):
    weights={};ln=rc=total=0.;types={'rc':set(),'ln':set()}
    for s in smooth(source):
        family=s['family'];tag=s['tag']
        if family=='break':continue
        duration=max(.5,(s['end']-s['start'])/1000)
        if family=='rc':rc+=duration
        elif family=='ln':ln+=duration
        if tag not in NON_AGG and family in types:types[family].add(tag)
        if tag in NON_AGG:continue
        total+=duration;weights[tag]=weights.get(tag,0.)+duration*max(.4,s['confidence'])
    # Stable insertion-order tie breaking, matching Object.entries().sort().
    ranked=sorted(weights,key=lambda c:-weights[c]);share=np.array([weights[c]/total for c in ranked])if total else np.zeros(0);ratio=ln/max(1.,ln+rc);sort='rc'if ratio<.18 else'ln'if ratio>.68 else'hb'
    return {'weights':weights,'ranked':ranked,'shares':share,'total':total,'sort':sort,'familyCount':sum(len(v)for v in types.values())}


def aggregate(stats,policy):
    entries=stats['ranked'];share=stats['shares'];sort=stats['sort']
    if not entries:tags=[]
    elif np.count_nonzero(share>=policy['mixShare'])>=3 and stats['familyCount']>=3 and share[0]<policy['dominantShare']:tags=[{'rc':'rc mix','ln':'ln mix','hb':'hybrid'}[sort]]
    elif len(entries)>=2 and share[1]>=policy['secondRel']*share[0]and share[0]<.55:tags=entries[:2]
    else:tags=entries[:1]
    return {'tags':tags,'sort':sort,'weights':stats['weights']}


def policy_predictions(stats,policies=POST):
    out=np.zeros((len(policies),len(stats),len(mil.MAP_CLASSES)),bool);d=np.array([p['dominantShare']for p in policies]);m=np.array([p['mixShare']for p in policies]);r=np.array([p['secondRel']for p in policies]);index={c:i for i,c in enumerate(mil.MAP_CLASSES)}
    for j,s in enumerate(stats):
        names=s['ranked'];shares=s['shares']
        if not names:continue
        mixed=(np.sum(shares[None,:]>=m[:,None],axis=1)>=3)&(s['familyCount']>=3)&(shares[0]<d)
        mix=index[{'rc':'rc mix','ln':'ln mix','hb':'hybrid'}[s['sort']]];out[mixed,j,mix]=True;fine=~mixed
        if names[0]in index:out[fine,j,index[names[0]]]=True
        if len(names)>=2 and names[1]in index:
            second=fine&(shares[1]>=r*shares[0])&(shares[0]<.55);out[second,j,index[names[1]]]=True
    return out


def macro_scores(y,p,w):
    tp=np.sum(p*y[None,:,:]*w[None,:,None],axis=1);fp=np.sum(p*(1-y)[None,:,:]*w[None,:,None],axis=1);support=w@y;den=support[None,:]+tp+fp
    f=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(1);recall=np.divide(tp,support[None,:],out=np.zeros_like(tp),where=support[None,:]>0).mean(1)
    return f,recall


def segment_accuracy(world,records,model):
    values=[]
    for r in records:
        f=frames(world,r['uid'],model);correct=total=0.
        for a in r['annotations']:
            if not a['tags']:continue
            total+=a['end']-a['start']
            for s in f:
                d=max(0,min(s['end'],a['end'])-max(s['start'],a['start']))
                correct+=d*(s['tag']in a['tags'])
        values.append(correct/total if total else 0.)
    return values
