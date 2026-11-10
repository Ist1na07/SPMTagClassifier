"""One bounded domain extension at the observed active parameter edges."""
from pathlib import Path
import json,itertools,time
import numpy as np
from model import Data,map_features,read
from integral import atoms

ROOT=Path('.cache/perception-extended-r2')
WEIGHTS=[]
for alpha,beta in itertools.product([0.,1.,2.,3.,4.],[0.,.5,1.,1.5,2.]):
    for width,hold in ([(1000.,1.)]if beta==0 else itertools.product([500.,1000.,2000.],[0.,1.])):
        WEIGHTS.append(dict(alpha=alpha,beta=beta,width=width,hold=hold))


def prepare():
    from whole_reference import write
    if(ROOT/'complete.json').exists():return
    d=Data();lookup={r['uid']:r for r in d.records};masses=[];totals=[];start=time.monotonic()
    for line in Path('.cache/perception-inputs-r1.jsonl').open(encoding='utf8'):
        row=json.loads(line);meta=lookup[row['uid']]
        with np.load(meta['curvePath'])as z:curve={k:z[k].copy()for k in ['query','demand','Rbar','Pbar','Cbar','Jm']}
        cache={};ms=[];ts=[];n=len(row['frames'])
        for c in WEIGHTS:
            pair=c['width'],c['hold']
            if pair not in cache:cache[pair]=atoms(row,curve,*pair)
            a=cache[pair];w=a['dt'].copy()
            if c['alpha']:w*=a['logDemand']**c['alpha']
            if c['beta']:w*=a['logActions']**c['beta']
            valid=a['frame']>=0;total=w.sum();ms.append(np.bincount(a['frame'][valid],weights=w[valid],minlength=n)/max(total,1e-30));ts.append(total)
        masses.append(np.array(ms));totals.append(ts)
        if len(masses)%75==0:print('extended integrals',len(masses),flush=True)
    mass=np.concatenate(masses,axis=1);total=np.array(totals).T;embedded=[]
    for old,c in enumerate(d.manifest['weights']):
        new=WEIGHTS.index(c);error=float(np.abs(mass[new]-d.arrays['mass'][old]).max())
        if error>1e-13:raise ValueError('Embedded original recipe changed')
        embedded.append(dict(originalWeightId=old,extendedWeightId=new,maxAbsoluteError=error))
    ROOT.mkdir(parents=True,exist_ok=True);np.savez_compressed(ROOT/'arrays.npz',mass=mass,total=total)
    write(ROOT/'manifest.json',dict(weights=WEIGHTS,recordUids=[r['uid']for r in d.records],embeddedOriginalRecipes=embedded,
        reason='R1 active optimum at alpha2,beta1 and several gate edges. One full Cartesian extension to alpha4,beta2,500ms; hold0/1 is a binary physical-credit ablation, supportpower1/2 remains the prototype definition/control.',
        newScorerCalls=0,sourceSnapshots={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['extended.py','integral.py']}))
    write(ROOT/'complete.json',dict(records=len(masses),weights=len(WEIGHTS),frames=mass.shape[1],seconds=time.monotonic()-start,newScorerCalls=0))


class ExtendedData(Data):
    def __init__(self):
        super().__init__();read(ROOT/'complete.json');meta=read(ROOT/'manifest.json')
        if meta['recordUids']!=[r['uid']for r in self.records]:raise ValueError('Extended physical inventory mismatch')
        with np.load(ROOT/'arrays.npz')as z:self.arrays['mass']=z['mass'].copy();self.arrays['total']=z['total'].copy()
        self.manifest['weights']=meta['weights'];self.cache={}
    def representation(self,wid,power=1.,hard=False,components=False):
        if components:raise ValueError('This bounded extension only changes the direct non-component readout')
        key=wid,power,hard
        if key not in self.cache:
            s=self.hard if hard else self.soft**power;empty=np.zeros((len(s),8))
            self.cache[key]=map_features(self,self.arrays['mass'][wid],s,empty,False)
        return self.cache[key]


if __name__=='__main__':prepare()
