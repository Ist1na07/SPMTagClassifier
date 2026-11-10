"""Nonzero synthetic-residual profile; real inputs test math, not accuracy."""
from pathlib import Path
import sys,json
import numpy as np
from model import Data,read
from joint import Kernel,infer
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def profile(d,model):
    c=model['config'];fi=[d.classes.index(c)for c in d.fine]
    return dict(schemaVersion=1,classes=d.fine,mapClasses=d.classes,components=True,mixKind='distinct',supportPower=c['power'],hard=False,fallback=True,
        recipe=d.manifest['weights'][c['weightId']],localCoef=model['localCoef'],localScaler=model['localScaler'],
        fine=dict(coef=model['fineCoef'],scaler={k:np.asarray(model['fineScaler'][k])[fi].tolist()for k in ['mean','scale','keep']}),
        slot=dict(coef=model['slotCoef'],scaler={k:model['slotScaler'][k][0]for k in ['mean','scale','keep']}))


def run():
    d=Data();p=PLANS[0]['inner'][0];wid=next(i for i,c in enumerate(d.manifest['weights'])if c==dict(alpha=1.,beta=1.,width=1000.,hold=1.));cfg=dict(weightId=wid,power=1.,components=True,hard=False,conditional=True,ridge=.003,balance=.5,localL2=.3,teacherWeight=1.,strongWeight=1.)
    k=Kernel(d,d.main[p['train']],d.manual[p['strongTrain']],cfg);theta=k.initial.copy();theta[:k.sizes[0]]=np.random.default_rng(20261110).normal(0,.02,k.sizes[0]);m=k.model(theta);out=infer(d,m)
    uids=[d.records[i]['uid']for i in d.main[[0,33,90,120,150,190,260,320]]]+['manual/DS','query/LN-H','query/CTX-H','query/B01','query/B02','derived/r10/I01-target-only','derived/r10/I02-target-only']
    rows={}
    for line in Path('.cache/perception-inputs-r1.jsonl').open(encoding='utf8'):
        r=json.loads(line)
        if r['uid']in uids:rows[r['uid']]=r
    cases=[]
    for uid in uids:
        i=d.lookup[uid];r=rows[uid];a,b=d.records[i]['offset']
        with np.load(d.records[i]['curvePath'])as z:
            curve=dict(schemaVersion=1,keyCount=7,speedRate=1.,clock='post-rate-ms',modelIdentity='5b0ddc61be53f5107f3b4dfa26a8397aa515e316c16b244878d1fc7039d0ba8f',
                query=z['query'].tolist(),demand=z['demand'].tolist(),components={k:z[k].tolist()for k in ['Rbar','Pbar','Cbar','Jm']})
        cases.append(dict(uid=uid,map=r['map'],segments=r['frames'],geometry=r['geometry'],rating=curve,
            expected=dict(mass=d.arrays['mass'][wid,a:b].tolist(),x=d.arrays['x'][wid,a:b].tolist(),support=out['support'][a:b].tolist(),share=out['share'][i].tolist(),fine=out['fine'][i,[d.classes.index(c)for c in d.fine]].tolist(),slots=out['slot'][i].tolist())))
    write('.cache/perception-runtime-resource.json',dict(purpose=__doc__,profile=profile(d,m),cases=cases))
    print('runtime resource cases',len(cases),flush=True)


if __name__=='__main__':run()
