"""One final simple readout: preserve prior, use percentages to correct it."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,argparse
import numpy as np
from model import Data,read,metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parent))
from threshold_study import reps,KEYS,eligible
from threshold import batch_metrics
from anchored import base,predictions,FINE_POLICIES,FAMILY_POLICIES
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write

ROOT=Path('research/perception/results/anchored-r1')
IDENTITY_FINE=FINE_POLICIES.index(dict(keepFloor=0.,addFloor=1.01,mixOverride=1.01))


def choose(d,scores,baseline,name='all',criterion='protected',family_enabled=None):
    allowed=np.array([eligible(d,r,name)for r in reps(d)])[:,None,None]
    if family_enabled is not None:allowed=allowed&np.array([p['enabled']==family_enabled for p in FAMILY_POLICIES])[None,:,None]
    if criterion=='protected':
        for j in [0,1,2,3]:allowed=allowed&(scores[...,j]>=baseline[j]-1e-12)
    objective=.5*scores[...,2]+.25*scores[...,3]+.25*scores[...,4]if criterion!='macro'else scores[...,2];value=np.where(allowed,objective,-np.inf)
    at=np.flatnonzero(value.ravel()>=value.max()-1e-14);recall=scores[...,1].ravel();at=at[recall[at]>=recall[at].max()-1e-14]
    # Numeric predictions can tie while displayed percentages differ. Prefer
    # the physical linear recipe, not a claim of data-identifying that recipe.
    def simplicity(flat):
        r,f,p=np.unravel_index(flat,value.shape);c=reps(d)[r];w=d.manifest['weights'][c['weightId']];policy=FINE_POLICIES[p];family=FAMILY_POLICIES[f]
        return (c['hard'],abs(w['alpha']-1)+abs(w['beta']-1)+abs(c['power']-1),w['width']!=1000.,w['hold']!=1.,sum([policy['keepFloor']>0,policy['addFloor']<=1.,policy['mixOverride']<=1.,family['enabled']]),int(flat))
    r,f,p=map(int,np.unravel_index(min(at,key=simplicity),value.shape));return dict(representationId=r,familyId=f,policyId=p,representation=reps(d)[r],family=FAMILY_POLICIES[f],policy=FINE_POLICIES[p],metrics=dict(zip(KEYS,scores[r,f,p].tolist())))


def run():
    d=Data();baseline=base(d);protocol=dict(id='perception-anchored-readout-r1',representations=reps(d),familyPolicies=FAMILY_POLICIES,finePolicies=FINE_POLICIES,plans=PLANS,
        reason='Unanchored learned and direct-percent readouts lose useful mature fine decisions. Keep them as a frozen prior; new integrated percentage can veto low-burden tags, promote strong branches, replace a Mix slot with a clear prototype, or recognize Hybrid from two families rather than requiring3fine types.',
        semantics='No chart-ID features, no new local predictions or scorer invocation. Fine thresholds shared acrossclasses. If no modification is enabled, exact v1.2.2 is embedded. Original positive Hybrid is retained when no new family trigger; this is conservative extension, not a silent redefinition of all prior decisions.',
        selection='All new recipe/support/fine/family coordinates jointly exhaustively selected. Protected requires INNER macroP,R,F1,microF1>=fixedbaseline, then balanced objective andrecall. Balanced/no-guard and macro secondary. Separate fullcombined and no-ratingactions selections. Final common model uses mean INNER only.',
        folds='Rules have no learned TRAIN coefficients: evaluating all OUTER-TRAIN charts is exactly the pooled3INNER validation inventory. Hyperparameters chosen on that inventory; original6OUTERgroups unseen to eachchoice. Do not inflate this into millions of numerical fits.',
        plateau='Equal objectives prefer a linear physical recipe and fewer active corrections; this is an explicit mathematical prior, not evidence of a unique parameter well. Identity tag outputs can coexist with different descriptive percentages.',
        stop='Final new aggregation family in this budget. No more alternate model families; finish existing full joint study and integrate the worthwhile candidate plus complete evidence.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['anchored.py','anchored_study.py']})
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Frozen anchored study changed')
    if not p.exists():write(p,protocol)
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    main=d.main;baseline_main=baseline[main];ys=d.y[main];ws=d.weight[main];all_scores=[[]for _ in range(6)];start=time.monotonic()
    for ri,r in enumerate(reps(d)):
        cache=ROOT/('representation-'+str(ri)+'.npz')
        if cache.exists():
            with np.load(cache)as z:rows=z['scores'].copy()
        else:
            _,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);q=predictions(d,share[main],family[main],baseline_main);rows=[]
            # Exact identity proof for every recipe, not just one special chart.
            np.testing.assert_array_equal(q[0,IDENTITY_FINE],baseline_main)
            for plan in PLANS:
                weight=ws.copy();weight[plan['test']]=0.;rows.append(batch_metrics(ys,q,weight))
            rows=np.array(rows);cache.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(cache,scores=rows)
        for o in range(6):all_scores[o].append(rows[o])
        if ri%10==0:print('anchored percent gates',ri,'/',len(reps(d)),round(time.monotonic()-start,1),'s',flush=True)
    for o in range(6):
        scores=np.array(all_scores[o]);folder=ROOT/str(o);folder.mkdir(parents=True,exist_ok=True);np.savez_compressed(folder/'inner-scores.npz',scores=scores)
        train=d.main[PLANS[o]['train']];b=metrics(d.y[train],baseline[train],d.weight[train]);reference=np.array([b[k]for k in KEYS]);choices={}
        for name in ['all','hard_time','branch_time','difficulty','actions','combined']:
            for criterion in ['protected','balanced','macro']:choices[name+'/'+criterion]=choose(d,scores,reference,name,criterion)
        for flag in [False,True]:choices['hybrid_'+str(flag)]=choose(d,scores,reference,'all','protected',flag)
        write(folder/'selection.json',dict(usesOuterTruth=False,baseline=b,choices=choices))
        out={};queries={}
        for name,c in choices.items():
            r=c['representation'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);q=predictions(d,share,family,baseline,[c['policy']],[c['family']])[0,0]
            pack=lambda i:dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))
            out[name]=[pack(i)for i in d.main[PLANS[o]['test']]];queries[name]=[pack(i)for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
        write(folder/'outer-predictions.json',out);write(folder/'queries.json',queries);write(folder/'complete.json',dict(candidateFunctions=int(np.prod(scores.shape[:-1])),numericalFits=0))
    write(ROOT/'complete.json',dict(scopes=6,candidateFunctionsPerOuter=int(np.prod(np.array(all_scores[0]).shape[:-1])),numericalFits=0,seconds=time.monotonic()-start))


if __name__=='__main__':run()
