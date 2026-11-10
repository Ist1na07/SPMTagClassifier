"""Use actual held-out source heads for the player's previously labelled cases."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route_consumers import Consumers,Kernel,read,write,PLANS
from competition import RC,LN

ROOT=Path('research/local_feedback/results/route-consumers-r5')


def coverage(frames,start,end):
    out={}
    for f in frames:
        dt=max(0.,min(end,f['end'])-max(start,f['start']))
        if dt:out[f['tag']]=out.get(f['tag'],0.)+dt/(end-start)
    return out


def probability(d,outer,arm):
    choice=read(ROOT/str(outer)/'selection.json')['choices'][arm];ci=choice['configId']
    if ci<0:return np.eye(3)[d.route]
    with np.load(ROOT/str(outer)/'probabilities.npz')as z:return z[str(ci)].copy()


def run():
    d=Consumers();cases=[];heads={}
    for rnd in ['r4','r5','r6']:
        for name,ann in read('research/frontier/annotations-'+rnd+'.json')['cases'].items():
            i=d.lookup[ann['sourceUid']];group=d.records[i]['group'];o=next(o for o,p in enumerate(PLANS)if group in {d.records[j]['group']for j in d.main[p['test']]});plan=PLANS[o]
            if group in{d.records[j]['group']for j in d.main[plan['train']]}:raise ValueError('Actual query song group leaked into fitting')
            for arm in ['all','joint','local_only','no_context','no_teacher']:
                if(o,arm)not in heads:heads[o,arm]=d.routed_scores(probability(d,o,arm))
                a,b=d.records[i]['offset'];start,end=ann['sourceTargetMs'];baseline=coverage([f for _,f in d.frame_records[a:b]],start,end);candidate=coverage(heads[o,arm][a:b],start,end);truth=ann['localCategory']
                cases.append(dict(id=name,arm=arm,sourceUid=ann['sourceUid'],sourceGroup=group,outer=o,actual=truth,additionalComponent=ann.get('additionalLocalComponent'),baseline=baseline,candidate=candidate,primaryCoverageDelta=candidate.get(truth,0)-baseline.get(truth,0)))
    saved=read(ROOT/'candidate-all.json');cfg=saved['model']['config'];k=Kernel(d,d.main,d.manual,cfg);theta=np.array(min(saved['receipt']['starts'],key=lambda s:s['loss'])['theta']);p,_,_,_,_=k.forward(theta,all_rows=True);frames=d.routed_scores(p);constructed=[]
    expected={'CTX-L':('dense chordstream','fast chordstream'),'CTX-H':('break',),'LN-H':('release',),'LN-L':('release',),'B01':('break',),'B02':('release',),'C16':('speed',),'C24':('tech',)}
    for name,tags in expected.items():
        i=d.lookup['query/'+name];a,b=d.records[i]['offset'];start,end=(9000.,17000.)if name.startswith(('CTX','LN-'))else(d.records[i]['frames'][0]['start'],d.records[i]['frames'][-1]['end']);base=coverage([f for _,f in d.frame_records[a:b]],start,end);after=coverage(frames[a:b],start,end)
        constructed.append(dict(id=name,allowedPrimary=tags,baseline=base,candidate=after,allowedCoverageDelta=sum(after.get(t,0)for t in tags)-sum(base.get(t,0)for t in tags)))
    write(ROOT/'human-regressions.json',dict(scope='Original source-song-held-out real target models; constructed cases use pooled-INNER common model. Active queries, not random accuracy.',real=cases,constructed=constructed))
    print('held-out real targets',flush=True)
    for r in cases:
        if r['arm']in ['all','no_context']:print(r['id'],r['arm'],r['actual'],r['baseline'],'=>',r['candidate'],flush=True)
    print('constructed',constructed,flush=True)


if __name__=='__main__':run()
