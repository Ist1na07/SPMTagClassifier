"""The archived hierarchical expert/gating decision, with replaceable heads.

Unlike the flat 16-way representation comparator, this preserves the original
RC/LN route, rule exceptions and Break heuristics. Cold refits can then compare
the actual original architecture without inheriting frozen pretrained weights.
"""
from mil import *

def load_metadata():
    with Path('.cache/rework-family-meta.jsonl').open(encoding='utf8')as f:
        return {r['uid']:r for r in map(json.loads,f)}

def model_scores(model,x):
    z=(x-np.array(model['mean']))/np.array(model['std'])@np.array(model['W']).reshape(len(model['classes']),-1).T+np.array(model['b'])
    z-=z.max(1,keepdims=True);p=np.exp(z);return p/p.sum(1,keepdims=True)

def classify(record,metadata,models,representation='original',ln_ratio=.45,scorer='model',precomputed=None):
    segments=record['segments'];meta=metadata[representation]
    if scorer!='rules':
        if precomputed is None:
            x=np.array([s[representation]for s in segments]);rc=model_scores(models['rc'],x);ln=model_scores(models['ln'],x)
        else:rc,ln=precomputed['rc'],precomputed['ln']
    features=[m['gates']for m in meta];nps=sorted(g['nps']for g in features if g['rowCount'])
    map_nps=nps[len(nps)//2]if nps else 0.;all_nps=sorted(g['nps']for g in features);median_nps=all_nps[len(all_nps)//2]
    output=[];burdens=[]
    for i,(segment,m,g)in enumerate(zip(segments,meta,features)):
        row={'start':segment['start'],'end':segment['end']};duration=row['end']-row['start']
        burden=g['nps']*(.5+g['chordMean']/3)*(.6+g['jackSpeedBpm']/150);row['burden']=burden
        low=(duration>=2500 and g['nps']<map_nps and
             (g['nps']<18.5 and g['segIvLogStd']>.41 or
              g['chordMean']<1.9 and g['vibroFraction']>.3 and g['segRhythmChangeRate']<.1 or
              g['nps']<22.5 and g['segIvLogStd']>.25 and g['vibroFraction']<.45 and g['lnWeightRatio']<.5))
        if m['family0']=='break'or not g['rowCount']or low:
            row.update(tag='break',family='break',confidence=1.)
        else:
            family='ln'if(g['lnWeightRatio']>=ln_ratio or g['lockRatio']>=.5 and g['heldCoverage']>=.15 and g['lnWeightRatio']>=.2)else'rc'
            rule=m['ruleLn'if family=='ln'else'ruleRc']
            if scorer=='rules':scores=dict(rule)
            else:
                head=models[family];mdl=dict(zip(head['classes'],ln[i]if family=='ln'else rc[i]))
                scores={}
                for tag in dict.fromkeys(list(mdl)+list(rule)):
                    if tag=='vibro':scores[tag]=min(1.,rule.get(tag,0)*2)
                    elif tag in ['wildcard','ln wildcard']:scores[tag]=rule.get(tag,0)*.6
                    elif scorer=='model':scores[tag]=mdl.get(tag,0)*1.4
                    else:scores[tag]=.5*(mdl.get(tag,0)+rule.get(tag,0))
            # Pure rules can have ZERO affinity after the LN lock rescue route:
            # its old lnPresence factor still gates every class at lnWeight=.45.
            # Zero information is not evidence for the dictionary's first tag.
            if sum(scores.values())<=0:
                unknown='ln wildcard'if family=='ln'else'wildcard'
                row.update(tag=unknown,family=family,confidence=0.,scores={unknown:1.},secondary=None,evidenceState='zero-rule-abstention')
                burdens.append(burden);output.append(row);continue
            ranked=sorted(scores.items(),key=lambda pair:-pair[1]);tag,confidence=ranked[0]
            if tag in ['wildcard','ln wildcard']:
                for other,value in ranked[1:]:
                    if other not in ['wildcard','ln wildcard']and value>.3:tag,confidence=other,value;break
            secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=ranked[0][1]*.75 and ranked[1][1]>.3 else None
            row.update(tag=tag,family=family,confidence=min(1.,float(confidence)),scores=scores,secondary=secondary)
        burdens.append(math.inf if row['family']=='break'else burden);output.append(row)
    finite=sorted(v for v in burdens if math.isfinite(v));p15=finite[int(len(finite)*.15)]if finite else math.inf
    for row,g,burden in zip(output,features,burdens):
        if row['family']=='break':continue
        easy=(row['confidence']<.5 and burden<=max(p15,1)and g['nps']<map_nps)
        sparse=(row['end']-row['start']<20000 and g['nps']>0 and g['nps']<median_nps*.4 and g['nps']<4)
        if easy or sparse:row.update(tag='break',family='break',secondary=None)
    for row in output:
        if row['family']=='break':row['probability']=[float(c=='break')for c in CLASSES]
        else:
            scores=row['scores'].copy();top=max(scores,key=scores.get)
            if top in ['wildcard','ln wildcard']and row['tag']not in ['wildcard','ln wildcard']:
                scores['wildcard']=0.;scores['ln wildcard']=0.
            total=sum(scores.values());row['probability']=[float(scores.get(c,0)/total)for c in CLASSES]
    return output

def smooth(rows,min_segment=3000):
    # Exact historical policy, including first-segment evidence inheritance.
    rows=[r.copy()for r in rows];out=[];minimum=.6*min_segment
    for row in rows:
        last=out[-1]if out else None
        if last and all(last.get(k)==row.get(k)for k in ['tag','secondary','family'])and row['end']-row['start']<minimum:last['end']=row['end']
        else:out.append(row)
    result=[]
    for i,row in enumerate(out):
        previous=result[-1]if result else None;following=out[i+1]if i+1<len(out)else None
        if row['family']=='break'and row['end']-row['start']<minimum and previous and following and following['tag']==previous['tag']:previous['end']=row['end']
        else:result.append(row)
    return result

def check_archived_consumer():
    main,manual,data=load();metadata=load_metadata();models=json.loads(Path('.cache/archived-model-reference.json').read_text(encoding='utf8'))
    original={r['uid']:r for r in json.loads(Path('.cache/rework-legacy-full.json').read_text(encoding='utf8'))}
    n=0
    for r in data:
        prediction=classify(r,metadata[r['uid']],models)
        assert [s['tag']for s in prediction]==[s['legacy']for s in r['segments']],r['uid']+' raw tags'
        expected=original[r['uid']]['segments'];actual=smooth(prediction)
        assert len(actual)==len(expected),r['uid']+' smoothing length'
        for a,e in zip(actual,expected):
            for key in ['start','end','tag','family']:assert a[key]==e[key],r['uid']+' '+key
            assert abs(a['confidence']-e['confidence'])<1e-10,r['uid']+' confidence'
        n+=1
    print('hierarchical native consumer parity',n,'complete cached charts; no model refitting',flush=True)

if __name__=='__main__':check_archived_consumer()
