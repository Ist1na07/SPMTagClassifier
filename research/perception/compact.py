"""Reuse the already rated Compact curves for the same aggregation model."""
from pathlib import Path
import sys,json,time
import numpy as np
from model import Data,read
from extended import WEIGHTS,ExtendedData
from integral import atoms
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write
from rating_bridge import load_rating,capture_native

ROOT=Path('.cache/perception-compact-r2')
IDENTITY='55216a3abcf6653c4c78b2082c730d64a0ed4bc476d3e477c4789a39f309fb4e'


def prepare():
    if(ROOT/'complete.json').exists():return
    d=Data();lookup={r['uid']:r for r in d.records};masses=[];totals=[];curves={};start=time.monotonic();rating=engine=model=None;newcalls=0
    for line in Path('.cache/perception-inputs-r1.jsonl').open(encoding='utf8'):
        row=json.loads(line);uid=row['uid'];old=Path('.cache/frontier-rating-r1/compact')/(uid.replace('/','__')+'.json')
        if old.exists():path=Path(read(old)['curvePath'])
        else:
            path=ROOT/'curves'/(uid.replace('/','__')+'.npz')
            if not path.exists():
                if rating is None:rating,engine=load_rating('.cache/spm-rating-v1.2.0');model=rating.load_model('compact')
                notes=engine.parser.parse_file(row['source'],rate=1.);e,receipt=capture_native(notes,model,engine);path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,query=e.query,**e.channels);newcalls+=1
        curves[uid]=str(path)
        with np.load(path)as z:curve={k:z[k].copy()for k in ['query','demand','Rbar','Pbar','Cbar','Jm']}
        cache={};ms=[];ts=[];n=len(row['frames'])
        for c in WEIGHTS:
            pair=c['width'],c['hold']
            if pair not in cache:cache[pair]=atoms(row,curve,*pair)
            a=cache[pair];w=a['dt'].copy()
            if c['alpha']:w*=a['logDemand']**c['alpha']
            if c['beta']:w*=a['logActions']**c['beta']
            valid=a['frame']>=0;total=w.sum();ms.append(np.bincount(a['frame'][valid],weights=w[valid],minlength=n)/max(total,1e-30));ts.append(total)
        masses.append(np.array(ms));totals.append(ts)
        if len(masses)%75==0:print('compact integral',len(masses),'newScores',newcalls,flush=True)
    ROOT.mkdir(parents=True,exist_ok=True);mass=np.concatenate(masses,axis=1);np.savez_compressed(ROOT/'arrays.npz',mass=mass,total=np.array(totals).T)
    write(ROOT/'manifest.json',dict(recordUids=[r['uid']for r in d.records],weights=WEIGHTS,modelIdentity=IDENTITY,curvePaths=curves,newScorerCalls=newcalls,sourceSnapshot=Path(__file__).read_text(encoding='utf8')))
    write(ROOT/'complete.json',dict(records=len(masses),weights=len(WEIGHTS),newScorerCalls=newcalls,seconds=time.monotonic()-start))


class CompactData(ExtendedData):
    def __init__(self):
        super().__init__();read(ROOT/'complete.json');m=read(ROOT/'manifest.json')
        if m['recordUids']!=[r['uid']for r in self.records]:raise ValueError('Compact inventory mismatch')
        detail=self.arrays['mass']
        with np.load(ROOT/'arrays.npz')as z:self.arrays['mass']=z['mass'].copy();self.arrays['total']=z['total'].copy()
        for wid,w in enumerate(WEIGHTS):
            if w['alpha']==0:np.testing.assert_array_equal(detail[wid],self.arrays['mass'][wid])
        for r in self.records:r['curvePath']=m['curvePaths'][r['uid']]
        self.cache={};self.rating_identity=IDENTITY


if __name__=='__main__':prepare()
