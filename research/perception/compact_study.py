"""Same frozen anchored model, separately tuned for the existing Compact UI."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time
import numpy as np
from model import read,metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parent))
from compact import CompactData,prepare,IDENTITY
from anchored import base,predictions
from anchored_extended_study import FINE,FAMILY,NEAR
from anchored_metrics import metrics_under_hybrid
from fast_select import select
from threshold_study import reps,KEYS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write

ROOT=Path('research/perception/results/anchored-compact-r2')
DETAIL=Path('research/perception/results/anchored-r2')


def run():
    prepare();d=CompactData();b=base(d);main=d.main;y=d.y[main];weight=d.weight[main];start=time.monotonic()
    protocol=dict(id='perception-compact-anchored-r2',ratingModelIdentity=IDENTITY,representations=reps(d),finePolicies=FINE,familyPolicies=FAMILY,plans=PLANS,
        purpose='Necessary caller coverage: existing overlay supports both Detailed and Compact. Consume each already-computed raw field with its own jointly selected aggregation profile; never relabel a Compact curve as Detailed or invoke an extra scorer.',
        comparison='Exactly the same anchored R2 rule,parameter Cartesian domain,near-optimal mathematical prior and INNER protection. Only frozen external curve/model changes. Alpha0 recipes have exact identical marginals; their score arrays are reused with proof rather than claimed as new fits.',
        selection='Same lexicographic selector algebra, vectorized ties. Compare actual completed Detailed scope choices before accepting this numerical implementation. No OUTER labels select parameters.',
        scope='Original349sourcegroups,6outer eachpool3INNER, no newly labelled queries fitted. No numerical classifier fit.12newCompactscorer calls only for new playables; original397curves reused.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['compact.py','compact_study.py','fast_select.py','anchored_metrics.py']})
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Frozen Compact study changed')
    if not p.exists():write(p,protocol)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    reused=0
    for ri,r in enumerate(reps(d)):
        p=ROOT/('representation-'+str(ri)+'.npz')
        if p.exists():continue
        if d.manifest['weights'][r['weightId']]['alpha']==0:
            with np.load(DETAIL/('representation-'+str(ri)+'.npz'))as z:rows=z['scores'].copy()
            reused+=1
        else:
            _,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);pre=predictions(d,share[main],family[main],b[main],FINE,[dict(enabled=False,minRc=0.,minLn=0.)])[0]
            trigger=np.array([(family[main,0]>=p['minRc'])&(family[main,1]>=p['minLn'])if p['enabled']else np.zeros(len(main),bool)for p in FAMILY]);rows=[]
            for plan in PLANS:
                w=weight.copy();w[plan['test']]=0.;rows.append(metrics_under_hybrid(y,pre,trigger,w,d.classes.index('hybrid')))
            rows=np.array(rows)
        np.savez_compressed(p,scores=rows)
        if ri%40==0:print('Compact percent gates',ri,'/',len(reps(d)),round(time.monotonic()-start,1),'s',flush=True)
    for o in range(6):
        matrices=[]
        for ri in range(len(reps(d))):
            with np.load(ROOT/('representation-'+str(ri)+'.npz'))as z:matrices.append(z['scores'][o].copy())
        scores=np.array(matrices);folder=ROOT/str(o);folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'inner-scores.npz',scores=scores)
        ids=d.main[PLANS[o]['train']];bm=metrics(d.y[ids],b[ids],d.weight[ids]);ref=np.array([bm[k]for k in KEYS]);choices={}
        for name in ['all','hard_time','branch_time','difficulty','actions','combined']:
            for criterion in ['protected','balanced','macro']:choices[name+'/'+criterion]=select(d,scores,ref,name,criterion)
        for flag in [False,True]:choices['hybrid_'+str(flag)]=select(d,scores,ref,'all','protected',flag)
        write(folder/'selection.json',dict(usesOuterTruth=False,baseline=bm,choices=choices));out={};queries={}
        for name,c in choices.items():
            r=c['representation'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);q=predictions(d,share,family,b,[c['policy']],[c['family']])[0,0]
            pack=lambda i:dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))
            out[name]=[pack(i)for i in d.main[PLANS[o]['test']]];queries[name]=[pack(i)for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
        write(folder/'outer-predictions.json',out);write(folder/'queries.json',queries);write(folder/'complete.json',dict(candidateFunctions=int(np.prod(scores.shape[:-1])),numericalFits=0))
    write(ROOT/'complete.json',dict(scopes=6,candidateFunctionsPerOuter=len(reps(d))*len(FINE)*len(FAMILY),reusedNoRatingRepresentations=reused,numericalFits=0,seconds=time.monotonic()-start))


if __name__=='__main__':run()
