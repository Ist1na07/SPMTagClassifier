"""Prepare exact duration/pressure marginals from already computed curves."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,itertools,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from pressure_data import mil,read
from whole_reference import write
from rating_bridge import load_rating,capture_native,RATING_COMMIT
from integral import atoms,summarize,LOCAL_KEYS,COMPONENTS

ROOT=Path('.cache/perception-r1')
FINE=[c for c in mil.MAP_CLASSES if c not in ['hybrid','rc mix','ln mix']]
WEIGHTS=[]
for alpha,beta in itertools.product([0.,1.,2.],[0.,.5,1.]):
    for width,hold in ([(1000.,1.)]if beta==0 else itertools.product([1000.,2000.],[0.,1.])):
        WEIGHTS.append(dict(alpha=alpha,beta=beta,width=width,hold=hold))


def run():
    if(ROOT/'complete.json').exists():raise ValueError('Completed inputs preserved')
    ROOT.mkdir(parents=True,exist_ok=True);rows=[json.loads(s)for s in Path('.cache/perception-inputs-r1.jsonl').read_text(encoding='utf8').splitlines()]
    main,manual,records=mil.load();lookup={r['uid']:r for r in records};original_order=[r['uid']for r in main];sources={}
    for r in records:
        p=Path('.cache/frontier-rating-r1/detail')/(r['uid'].replace('/','__')+'.json')
        sources.setdefault(r['source'],read(p)['curvePath'])
    rating=engine=model=None;blocks=[];meta=[];offset=0;newcalls=0;start=time.monotonic()
    for i,r in enumerate(rows):
        source=r['source'];key=r['uid'].replace('/','__');curvepath=Path(sources.get(source,ROOT/'curves'/(key+'.npz')))
        if not curvepath.exists():
            if rating is None:rating,engine=load_rating('.cache/spm-rating-v1.2.0');model=rating.load_model('detail')
            notes=engine.parser.parse_file(source,rate=1.);e,receipt=capture_native(notes,model,engine)
            curvepath.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(curvepath,query=e.query,**e.channels);newcalls+=1
        with np.load(curvepath)as z:curve={k:z[k].copy()for k in ['query','demand']+COMPONENTS}
        cache={};masses=[];xs=[];totals=[]
        for cfg in WEIGHTS:
            pair=cfg['width'],cfg['hold']
            if pair not in cache:cache[pair]=atoms(r,curve,*pair)
            m,x,t=summarize(r,cache[pair],cfg['alpha'],cfg['beta']);masses.append(m);xs.append(x);totals.append(t)
        frames=r['frames'];n=len(frames);support=np.array([[max(0.,float(s.get('scores',{}).get(c,0.)))for c in FINE]for s in frames])
        divisor=np.array([max(1.,max(s.get('scores',{}).values(),default=0.))for s in frames]);support/=divisor[:,None]
        for j,s in enumerate(frames):
            if s['family']=='break':support[j]=0.
        hard=np.array([[float(s['tag']==c)for c in FINE]for s in frames]);family=np.array([[s['family']=='rc',s['family']=='ln']for s in frames],float)
        original=lookup.get(r['uid']);derived=read('research/frontier/annotations-r10.json')['derivedWholeCharts']['cases']
        if original:group=original['group'];tags=mil.target(original);weight=original.get('recordWeight',1.);dataset=original['dataset'];annotations=original.get('annotations',[])
        elif r['uid'].startswith('derived/'):
            d=next(d for d in derived.values()if d['uid']==r['uid']);group=d['sourceGroup'];tags=d['tags'];weight=1.;dataset='derived';annotations=[]
        else:group=r['uid'];tags=[];weight=1.;dataset='query';annotations=[]
        meta.append(dict(uid=r['uid'],source=source,group=group,tags=tags,weight=weight,dataset=dataset,annotations=annotations,
            offset=[offset,offset+n],frames=frames,legacyOverall=r['legacyOverall'],curvePath=str(curvepath),
            ratingEndpointExtensionMs=next(iter(cache.values()))['ratingEndpointExtensionMs']))
        blocks.append(dict(mass=np.array(masses),x=np.array(xs),support=support,hard=hard,family=family,total=np.array(totals)))
        offset+=n
        if(i+1)%50==0:print('perception marginals',i+1,'/',len(rows),'newRatingCalls',newcalls,flush=True)
    np.savez_compressed(ROOT/'arrays.npz',mass=np.concatenate([b['mass']for b in blocks],axis=1),x=np.concatenate([b['x']for b in blocks],axis=1),
        support=np.concatenate([b['support']for b in blocks]),hard=np.concatenate([b['hard']for b in blocks]),family=np.concatenate([b['family']for b in blocks]),
        bag=np.concatenate([np.full(len(b['support']),i)for i,b in enumerate(blocks)]),total=np.array([b['total']for b in blocks]).T)
    write(ROOT/'manifest.json',dict(records=meta,mainUids=original_order,manualUids=[r['uid']for r in manual],fineClasses=FINE,mapClasses=mil.MAP_CLASSES,
        weights=WEIGHTS,localKeys=LOCAL_KEYS,ratingCommit=RATING_COMMIT,newRatingCalls=newcalls,
        integral='Exact union-breakpoint right-continuous rating/action/support integration. Endpoint curve extension recorded per chart. Break has zero support but physical burden remains in denominator.',
        sourceSnapshots={k:Path(__file__).with_name(k).read_text(encoding='utf8')for k in ['prepare.py','integral.py']}))
    write(ROOT/'complete.json',dict(records=len(rows),frames=offset,weights=len(WEIGHTS),newRatingCalls=newcalls,seconds=time.monotonic()-start))
    print('complete',read(ROOT/'complete.json'),flush=True)


if __name__=='__main__':run()
