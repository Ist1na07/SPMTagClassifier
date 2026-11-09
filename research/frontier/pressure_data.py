"""Fixed-boundary, label-independent joins for the soft-pressure model."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rework'))
import mil
import numpy as np
import json


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def attach(records):
    event={r['uid']:r for r in map(json.loads,Path('.cache/frontier-event-features-r1.jsonl').read_text().splitlines())}
    schema=read('.cache/frontier-event-features-r1.jsonl.schema.json')['keys'];archive=read('.cache/archived-model-reference.json')
    active=mil.CLASSES[:-1];base_count=schema.index('after15')
    core=list(range(base_count+8));motion=list(range(base_count+8,len(schema)))
    for r in records:
        intervals=[[s['start'],s['end']]for s in r['segments']];row=event[r['uid']]
        rating=read(Path('.cache/frontier-rating-r1/detail')/(r['uid'].replace('/','__')+'.json'))
        hand=read(Path('.cache/frontier-hand-r1')/(r['uid'].replace('/','__')+'.json'))
        if intervals!=row['intervals']or intervals!=hand['intervals']or intervals!=[[s['start'],s['end']]for s in rating['windows']]:raise ValueError('Joined interval inventory differs')
        x=np.array([s['values']for s in row['windows']]);h=np.array(hand['values'],float)
        prior=np.zeros((len(x),len(active)))
        raw=np.array([s['original']for s in r['segments']])
        for family in ['rc','ln']:
            m=archive[family];z=(raw-np.array(m['mean']))/np.array(m['std'])@np.array(m['W']).reshape(len(m['classes']),-1).T+np.array(m['b'])
            z-=z.mean(1,keepdims=True)
            for i,c in enumerate(m['classes']):prior[:,active.index(c)]=z[:,i]
        duration=np.array([s['end']-s['start']for s in r['segments']],float)
        pressure=np.array([[x[i,schema.index('npsPerKey')],x[i,schema.index('release:tailRatePerKey')],*w['retainedRates'],
            rating['windows'][i]['channels']['demand']['rowMean'],rating['windows'][i]['channels']['demand']['rowP90']]for i,w in enumerate(row['windows'])],float)
        pressure=np.log1p(pressure);relative=pressure.copy()
        for j in range(pressure.shape[1]):
            observed=np.isfinite(pressure[:,j])
            if observed.any():relative[observed,j]-=np.average(pressure[observed,j],weights=duration[observed])
        mask=np.isfinite(pressure[:,-1]).astype(float)[:,None]
        r['_frontier']={'core':x[:,core],'motion':x[:,motion],'hand':np.c_[h,np.asarray(hand['observed'],float)],
            'pressure':np.c_[pressure,relative,mask],'typePrior':prior,'hardActive':np.array([s['legacy']!='break'for s in r['segments']],float),
            'tailRate':x[:,schema.index('release:tailRatePerKey')]}
    return {'core':[schema[i]for i in core],'motion':[schema[i]for i in motion],
        'hand':hand['keys']+['handObserved'],
        'pressure':['absolute:'+c for c in ['headPerKey','tailPerKey','retained15','retained30','retained60','retained120','ratingMean','ratingP90']]+
            ['relative:'+c for c in ['headPerKey','tailPerKey','retained15','retained30','retained60','retained120','ratingMean','ratingP90']]+['ratingObserved']}


def scale_matrix(raw,weights,mean=None,scale=None):
    observed=np.isfinite(raw)
    if mean is None:
        mass=(weights[:,None]*observed).sum(0)
        mean=np.divide((np.where(observed,raw,0)*weights[:,None]).sum(0),mass,out=np.zeros(raw.shape[1]),where=mass>0)
        centered=np.where(observed,raw-mean,0)
        scale=np.sqrt(np.divide((centered**2*weights[:,None]).sum(0),mass,out=np.zeros(raw.shape[1]),where=mass>0))
        scale=np.maximum(scale,1e-6)
    # Missing means unobserved; explicit hand/rating masks remain in the input.
    x=np.clip(np.where(observed,(raw-mean)/scale,0),-6,6)
    return x,mean,scale
