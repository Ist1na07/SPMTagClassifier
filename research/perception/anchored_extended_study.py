"""One complete boundary extension; same model, faster exact metric algebra."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,itertools,time
import numpy as np
from extended import ExtendedData,prepare,WEIGHTS
from model import read,metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parent))
from anchored import base,predictions
from anchored_metrics import metrics_under_hybrid
from threshold_study import reps,KEYS,eligible
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write

ROOT=Path('research/perception/results/anchored-r2')
FINE=[dict(keepFloor=k,addFloor=a,mixOverride=m)for k,a,m in itertools.product([0.,.03,.07,.15,.25,.35,.5],[.25,.35,.4,.5,.65,.8,.95,1.01],[.35,.5,.65,.8,.95,1.01])]
FAMILY=[dict(enabled=False,minRc=0.,minLn=0.)]+[dict(enabled=True,minRc=r,minLn=l)for r,l in itertools.product([.03,.05,.07,.1,.18,.2,.3,.32,.4],[.03,.05,.07,.1,.18,.2,.3,.32,.4])]
NEAR=.001


def select(d,scores,baseline,name='all',criterion='protected',family_enabled=None):
    allowed=np.array([eligible(d,r,name)for r in reps(d)])[:,None,None]
    if family_enabled is not None:allowed=allowed&np.array([p['enabled']==family_enabled for p in FAMILY])[None,:,None]
    if criterion=='protected':
        for j in [0,1,2,3]:allowed=allowed&(scores[...,j]>=baseline[j]-1e-12)
    objective=.5*scores[...,2]+.25*scores[...,3]+.25*scores[...,4]if criterion!='macro'else scores[...,2];value=np.where(allowed,objective,-np.inf)
    at=np.flatnonzero(value.ravel()>=value.max()-(NEAR if criterion=='protected'else 1e-14))
    def simplicity(flat):
        r,f,p=np.unravel_index(flat,value.shape);c=reps(d)[r];w=d.manifest['weights'][c['weightId']];policy=FINE[p];family=FAMILY[f]
        return (c['hard'],(w['alpha']-1)**2+(w['beta']-1)**2+(c['power']-1)**2,w['width']!=1000.,w['hold']!=1.,
            sum([policy['keepFloor']>0,policy['addFloor']<=1.,policy['mixOverride']<=1.,family['enabled']]),-value.ravel()[flat],-scores[...,1].ravel()[flat],int(flat))
    chosen=min(at,key=simplicity)if criterion=='protected'else min(at,key=lambda i:(-scores[...,1].ravel()[i],i))
    r,f,p=map(int,np.unravel_index(chosen,value.shape));return dict(representationId=r,familyId=f,policyId=p,representation=reps(d)[r],family=FAMILY[f],policy=FINE[p],metrics=dict(zip(KEYS,scores[r,f,p].tolist())),
        objectiveDeficitFromMaximum=float(value.max()-value.ravel()[chosen]),nearOptimalTolerance=NEAR if criterion=='protected'else 0.)


def declare(d):
    v=dict(id='perception-anchored-r2-boundary-extension',representations=reps(d),familyPolicies=FAMILY,finePolicies=FINE,plans=PLANS,
        purpose='Same anchored percentage model, one bounded full Cartesian extension at active R1 edges: difficulty exponents to4,actions to2,window down500ms; keepFloor up.5,add/mixOverride down.25/.35,family significance down.03. All combinations jointly compared, not one-coordinate-only tuning.',
        prior='Protected mean score within.1percentagepoint of maximum prefers linear alpha/beta/support degree1,1s action window,explicit hold credit, fewer active correction rules. MacroP,R,F1,micro safeguards against baseline still hold on INNER. This is a stated mathematical stability prior, not a newly identified data well.',
        metricAlgebra='Exact Boolean rule factorization: pre-correction predictions masked by family Hybrid; TP/FP, sample and exact metrics computed with matrix products. Compared to original explicit Boolean implementation max2.22e-16. No changed target, metric, parameter or source inventory.',
        labels='Original349, original6outer source/song/physical groups. No fit: pooled3INNER validation is exactly outerTRAIN. Cropped/context query labels never selected. Model inputs unchanged frozenv1.2.2; no newscorer calls.',
        semantics='mixOverride replaces ANY old composite slot (includingHybrid) when a strong fine prototype exists; absence of a new family trigger alone never removes an oldHybrid. Identity is exact tag-set identity; selected fine tags are ranked by their new physical shares for display.',
        stop='Last grid expansion in this half-iteration. Profile active boundaries and output-vector/percentage instability; report unresolved edges instead of promising a global optimum or extending the budget.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['anchored.py','anchored_metrics.py','anchored_extended_study.py','extended.py']})
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=v:raise ValueError('Frozen extended readout changed')
    if not p.exists():write(p,v)


def run():
    prepare();d=ExtendedData();declare(d);b=base(d);main=d.main;y=d.y[main];weight=d.weight[main];start=time.monotonic();ROOT.mkdir(parents=True,exist_ok=True)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    for ri,r in enumerate(reps(d)):
        p=ROOT/('representation-'+str(ri)+'.npz')
        if p.exists():continue
        _,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);pre=predictions(d,share[main],family[main],b[main],FINE,[dict(enabled=False,minRc=0.,minLn=0.)])[0]
        trigger=np.array([(family[main,0]>=p['minRc'])&(family[main,1]>=p['minLn'])if p['enabled']else np.zeros(len(main),bool)for p in FAMILY]);rows=[]
        for plan in PLANS:
            w=weight.copy();w[plan['test']]=0.;rows.append(metrics_under_hybrid(y,pre,trigger,w,d.classes.index('hybrid')))
        np.savez_compressed(p,scores=np.array(rows))
        if ri%25==0:print('anchored boundary',ri,'/',len(reps(d)),round(time.monotonic()-start,1),'s',flush=True)
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
    write(ROOT/'complete.json',dict(scopes=6,candidateFunctionsPerOuter=len(reps(d))*len(FINE)*len(FAMILY),numericalFits=0,newScorerCalls=0,seconds=time.monotonic()-start))


if __name__=='__main__':run()
