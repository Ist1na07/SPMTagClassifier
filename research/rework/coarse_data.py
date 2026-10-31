"""Shared 4K/7K coarse labels and cross-key song-group identities.

Fine attributes are not inferred from coarse bucket absence. The unlabelled
6K corpus is not used for supervised selection or claimed accuracy.
"""
from mil import canonical,grouped_folds
from pathlib import Path
from collections import Counter,defaultdict
import json

CACHE=Path('.cache/rework-coarse-r2')
CLASSES=['jack','chordstream','speed','tech','ln','break']
FINE_TO_COARSE={
 'chordjack':'jack','minijack':'jack','vibro':'jack','anchor':'jack',
 'dense chordstream':'chordstream','fast chordstream':'chordstream','chordstream':'chordstream',
 'speed':'speed','tech':'tech',
 **{x:'ln'for x in ['coordination','density','inverse','release','technical','ln wildcard','ln mix']},
 'break':'break'}
FOUR={'jack':'jack','stamina':'chordstream','speed':'speed','tech':'tech','ln':'ln'}

def whole_target(r):
    if r['dataset']=='manual':return []
    if r['keys']==4:return [FOUR[r['subset']]]if r.get('subset')in FOUR else []
    tags=[x.lower()for x in r['tags']]
    if not tags or any(x not in FINE_TO_COARSE for x in tags):return []
    labels={FINE_TO_COARSE[x]for x in tags}
    # A single coarse impression is supported even with multiple finer tags.
    # RC Mix/Hybrid do not identify a unique coarse shape and are not forced.
    return sorted(labels)if len(labels)==1 and 'break'not in labels else []

def annotate(r):
    out=[]
    for a in r.get('annotations',[]):
        tags=a['tags']
        # A wildcard alternative is genuinely unknown at this layer; dropping
        # only that alternative would invent a stricter annotation.
        coarse=sorted({FINE_TO_COARSE[t]for t in tags})if tags and all(t in FINE_TO_COARSE for t in tags)else []
        out.append({**a,'fineTags':tags,'tags':coarse})
    return out

def group(records):
    parent={r['uid']:r['uid']for r in records}
    def find(x):
        if parent[x]!=x:parent[x]=find(parent[x])
        return parent[x]
    def join(a,b):parent[find(b)]=find(a)
    known={}
    for r in records:
        if r.get('duplicateOf')in parent:join(r['uid'],r['duplicateOf'])
        m=r['meta'];artist=canonical(m.get('Artist',''));title=canonical(m.get('Title',''))
        identities=[('song',artist,title)]if title else []
        for key in ['BeatmapID','BeatmapSetID']:
            v=str(m.get(key,''))
            if v.isdigit()and int(v)>0:identities.append((key,v))
        base=canonical(r.get('extra',{}).get('base_title',''))
        if base:identities.append(('base',base))
        identities.append(('source',r['source']))
        for identity in identities:
            if identity in known:join(r['uid'],known[identity])
            else:known[identity]=r['uid']
    for r in records:r['group']=find(r['uid'])
    return records

def load_headers():
    inventory=json.loads((CACHE/'inventory.json').read_text(encoding='utf8'))
    if inventory['errors']:raise RuntimeError('Resolve labelled extraction errors before supervised search')
    records=group(inventory['records']);manual=[r for r in records if r['dataset']=='manual']
    mg={r['group']for r in manual};weak=[r for r in records if whole_target(r)and r['group']not in mg]
    copies=Counter(r['source']for r in weak)
    for r in weak:r['coarseTags']=whole_target(r);r['recordWeight']=1/copies[r['source']]
    for r in manual:r['annotations']=annotate(r)
    return weak,manual,records

def plans(weak,manual):
    if len(manual)!=6 or len({r['group']for r in manual})!=6:
        raise RuntimeError('Leave-one-chart protocol requires the six actual manual song groups')
    # Allocate source/class pairs, but share one group namespace across keys.
    strat=[{**r,'tags':[str(r['keys'])+':'+r['coarseTags'][0]]}for r in weak]
    outer=grouped_folds(strat,6);result=[]
    for o,test in enumerate(outer):
        train=[i for i in range(len(weak))if i not in set(test)];sm=[i for i in range(6)if i!=o]
        inner=[]
        for j,va0 in enumerate(grouped_folds([strat[i]for i in train],3)):
            va=[train[i]for i in va0];tr=[i for i in train if i not in set(va)];sv=[i for k,i in enumerate(sm)if k%3==j]
            inner.append({'train':tr,'validation':va,'strongTrain':[i for i in sm if i not in sv],'strongValidation':sv})
        result.append({'train':train,'test':test,'strongTrain':sm,'strongTest':[o],'inner':inner})
    return result

def audit():
    weak,manual,allrows=load_headers();ps=plans(weak,manual)
    groups=defaultdict(list)
    for r in allrows:groups[r['group']].append(r)
    mixed=[{'group':g,'uids':[r['uid']for r in rs],'keys':sorted({r['keys']for r in rs})}for g,rs in groups.items()if len({r['keys']for r in rs})>1]
    support={str(K):{'records':sum(r['keys']==K for r in weak),'groups':len({r['group']for r in weak if r['keys']==K}),
      'classes':dict(Counter(r['coarseTags'][0]for r in weak if r['keys']==K))}for K in [4,7]}
    for p in ps:
        for a,b in [(p['train'],p['test'])]+[(s['train'],s['validation'])for s in p['inner']]:
            if {weak[i]['group']for i in a}&{weak[i]['group']for i in b}:raise AssertionError('Cross-key song leakage')
    result={'id':'coarse-cross-key-data-r1','labelMeaning':'4K stamina is the source jump/handstream bucket; 7K fine tags mapped only when one coarse impression is determined',
      'support':support,'manualCharts':len(manual),'allCrossKeySongGroups':mixed,
      'largestGroups':[{'group':g,'records':len(rs)}for g,rs in sorted(groups.items(),key=lambda x:-len(x[1]))[:12]],
      'excludedManualOverlap':[r['uid']for r in allrows if r['dataset']!='manual'and whole_target(r)and r['group']in {s['group']for s in manual}],
      'records':[{'uid':r['uid'],'group':r['group'],'keys':r['keys'],'tags':r['coarseTags'],'weight':r['recordWeight']}for r in weak],
      'manual':[{'uid':r['uid'],'group':r['group'],'annotations':r['annotations']}for r in manual],'plans':ps}
    out=Path('research/rework/results/coarse-data');out.mkdir(parents=True,exist_ok=True)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps({'support':support,'crossKeyGroups':len(mixed),'excludedManualOverlap':len(result['excludedManualOverlap'])},ensure_ascii=False))

if __name__=='__main__':audit()
