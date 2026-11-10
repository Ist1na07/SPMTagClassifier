"""Exact integral marginals on new child intervals, reusing native curves."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import PerceptionData,read,write
from integral import atoms

ROOT=Path('.cache/local-feedback-temporal-r4b')


def run():
    if(ROOT/'complete.json').exists():return
    d=PerceptionData();profile=read('src/models/perception7k.json')['profiles']['detail'];wid=d.manifest['weights'].index(profile['recipe']);schema=read('.cache/local-feedback-temporal-r4b.jsonl.schema.json')
    source=Path('.cache/perception-inputs-r1.jsonl').open(encoding='utf8');temporal=Path('.cache/local-feedback-temporal-r4b.jsonl').open(encoding='utf8');records=[];blocks=[];offset=0;maximum=0.;begin=time.monotonic()
    for meta,a,b in zip(d.records,source,temporal):
        row=json.loads(a);new=json.loads(b)
        if row['uid']!=meta['uid']or new['uid']!=meta['uid']:raise ValueError('Temporal source order mismatch')
        row['frames']=new['frames'];frames=row['frames'];n=len(frames)
        with np.load(meta['curvePath'])as z:curve={k:z[k].copy()for k in ['query','demand','Rbar','Pbar','Cbar','Jm']}
        atom=atoms(row,curve,1000.,1.);weight=atom['dt']*atom['logDemand']**2*atom['logActions'];valid=atom['frame']>=0;mass=np.bincount(atom['frame'][valid],weights=weight[valid],minlength=n)/max(weight.sum(),1e-30)
        parent=np.array([f['parent']for f in frames]);old_a,old_b=meta['offset'];restored=np.bincount(parent,weights=mass,minlength=old_b-old_a);error=float(np.max(np.abs(restored-d.arrays['mass'][wid,old_a:old_b])));maximum=max(maximum,error)
        if error>1e-12:raise ValueError('Refinement changed physical burden/parent integral')
        div=np.array([max(1.,max(f['scores'].values(),default=0.))for f in frames]);raw=np.array([[f['scores'].get(c,0.)for c in list(d.fine)+['wildcard']]for f in frames]);base=np.minimum(.999,np.maximum(0.,raw)/div[:,None]);base[[j for j,f in enumerate(frames)if f['family']=='break']]=0.
        blocks.append(dict(mass=mass,base=base,raw=raw,div=div,parent=parent,local=np.array(new['localLogOdds']).T,margin=np.array(new['localGlobalMargins']).T,dt=np.array([f['end']-f['start']for f in frames])))
        records.append(dict(meta,offset=[offset,offset+n],frames=frames));offset+=n
        if len(records)%75==0:print('exact temporal marginals',len(records),offset,flush=True)
    if len(records)!=len(d.records):raise ValueError('Temporal inventory incomplete')
    ROOT.mkdir(exist_ok=True);np.savez_compressed(ROOT/'arrays.npz',**{k:np.concatenate([r[k]for r in blocks],axis=1 if k in ['local','margin']else 0)for k in blocks[0]},bag=np.concatenate([np.full(len(b['mass']),i)for i,b in enumerate(blocks)]))
    write(ROOT/'manifest.json',dict(records=records,mainUids=d.manifest['mainUids'],manualUids=d.manifest['manualUids'],fineClasses=d.fine,mapClasses=d.classes,profile=profile,variants=schema['variants'],parentIntegralMaximumError=maximum))
    write(ROOT/'complete.json',dict(records=len(records),frames=offset,newScorerCalls=0,seconds=time.monotonic()-begin,parentIntegralMaximumError=maximum));print('complete exact temporal',offset,maximum,flush=True)


if __name__=='__main__':run()
