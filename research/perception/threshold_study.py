"""Direct physical percentage gates, all declared parameters jointly selected."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,time,argparse,itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from model import Data,read,metrics
from threshold import POLICIES,SHRINK,train_thresholds,predict,batch_metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write

ROOT=Path('research/perception/results/threshold-r1')
KEYS=['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']
_D=None


def reps(d):return [dict(weightId=0,power=1.,hard=True)]+[dict(weightId=w,power=p,hard=False)for w,p in itertools.product(range(len(d.manifest['weights'])),[1.,2.])]


def init():
    global _D
    _D=Data()
    if os.name=='nt':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)


def eligible(d,r,name):
    w=d.manifest['weights'][r['weightId']]
    if name=='all':return True
    if name=='hard_time':return r['hard']
    if r['hard']:return False
    if name=='branch_time':return w['alpha']==w['beta']==0
    if name=='difficulty':return w['alpha']>0 and w['beta']==0
    if name=='actions':return w['alpha']==0 and w['beta']>0
    if name=='combined':return w['alpha']>0 and w['beta']>0
    return False


def choose(d,score,name='all',criterion='balanced',hybrid=None):
    mask=np.array([eligible(d,r,name)for r in reps(d)])[:,None,None]
    if hybrid is not None:mask=mask&np.array([p['hybrid']==hybrid for p in POLICIES])[None,:,None]
    value=.5*score[...,2]+.25*score[...,3]+.25*score[...,4]if criterion=='balanced'else score[...,2]
    value=np.where(mask,value,-np.inf);best=np.flatnonzero(value.ravel()>=value.max()-1e-14);recall=score[...,1].ravel();best=best[recall[best]>=recall[best].max()-1e-14];at=np.unravel_index(best[0],value.shape);ri,pi,si=map(int,at)
    return dict(representationId=ri,policyId=pi,shrinkId=si,representation=reps(d)[ri],policy=POLICIES[pi],shrink=SHRINK[si],metrics=dict(zip(KEYS,score[at].tolist())))


def job(outer):
    d=_D;folder=ROOT/str(outer)
    if(folder/'complete.json').exists():return outer,'resumed'
    plan=PLANS[outer];ids=[i for s in plan['inner']for i in s['validation']];va=d.main[ids];scores=[];start=time.monotonic()
    for ri,r in enumerate(reps(d)):
        _,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);ps=[]
        for s in plan['inner']:
            tau,iv,obj=train_thresholds(d,share,family,d.main[s['train']]);v=d.main[s['validation']];ps.append(predict(d,share[v],family[v],tau))
        q=np.concatenate(ps,axis=2);m=batch_metrics(d.y[va],q,d.weight[va]);scores.append(m)
        if ri%10==0:print('percentage gates',outer,ri,'/',len(reps(d)),round(time.monotonic()-start,1),'s',flush=True)
    scores=np.array(scores);np.savez_compressed(folder/'inner-scores.npz',scores=scores);choices={}
    for name in ['all','hard_time','branch_time','difficulty','actions','combined']:
        for criterion in ['balanced','macro']:choices[name+'/'+criterion]=choose(d,scores,name,criterion)
    for hybrid in [False,True]:choices['hybrid_'+str(hybrid)]=choose(d,scores,'all','balanced',hybrid)
    write(folder/'selection.json',dict(usesOuterTruth=False,choices=choices));models={};out={};queries={}
    for name,c in choices.items():
        r=c['representation'];ident=c['representationId'],c['policyId'],c['shrinkId'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard'])
        if ident not in models:
            tau,iv,obj=train_thresholds(d,share,family,d.main[plan['train']],[c['policy']],[c['shrink']]);models[ident]=dict(thresholds=tau[0,0].tolist(),intervals=iv[0,0].tolist(),trainUtility=obj[0,0].tolist(),choice=c,trainUids=[d.records[i]['uid']for i in d.main[plan['train']]])
        m=models[ident];q=predict(d,share,family,np.array(m['thresholds'])[None,None,:],[c['policy']])[0,0]
        pack=lambda i:dict(uid=d.records[i]['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))
        out[name]=[pack(i)for i in d.main[plan['test']]];groups={d.records[i]['group']for i in d.main[plan['train']]}
        queries[name]=[dict(pack(i),heldOutSourceGroup=x['group']not in groups)for i,x in enumerate(d.records)if x['dataset']in ['query','derived']]
    write(folder/'outer-models.json',{str(k):v for k,v in models.items()});write(folder/'outer-predictions.json',out);write(folder/'queries.json',queries)
    write(folder/'complete.json',dict(representations=len(scores),candidatePolicies=int(np.prod(scores.shape[:-1])),innerCalibrationBatches=len(reps(d))*3,seconds=time.monotonic()-start))
    return outer,read(folder/'complete.json')


def declare(d):
    protocol=dict(id='perception-direct-percentage-gates-r1',representations=reps(d),policies=POLICIES,shrink=SHRINK,plans=PLANS,
        reason='User requests percentage itself to gate tags, not a second correctness-probability scale. Test the simplest direct gates. Hybrid requires two meaningful families, not three fine categories; primitive mixtures retain3significant fine branches.',
        fit='For each representation/coarse policy/shrink config and class, exhaustive TRAIN share-threshold partitions maximize weighted endpoint F1 minus strength/(1+positiveMass)*log(tau/reference)^2. Apply actual composite-slot mask before fine fitting. Fixed coarse errors retained. No fallback primary coupling; empty evidence can abstain.',
        flatParameters='Each partition is an interval. Choose point closest to physical reference, including strength0; reference still changes unseen-chart decisions on a TRAIN plateau, so strength0 references are NOT falsely canonicalized as identical functions. Report accepted intervals and actual held-out changes.',
        selection='Full representation x coarse x shrink Cartesian INNER OOF: same balanced criterion=.5macroF1+.25microF1+.25sampleF1, recall tie; macro-only secondary. Six original OUTER song groups unchanged, common model selected using mean INNER arrays before OUTER consumption.',
        comparisons='hard time,branch time,difficulty only,actions only,both all independently retuned. Hybrid enabled/disabled also independently retuned. No components or changed local scores here; compare separately with full-component/backprop studies.',
        resource='Vectorized exact threshold search: real inner fold740policiesx12shrinkx14fine in~.36s during6jointworkers. Batch metric identity checked against scalar evaluator. Run1lower-priorityworker, do not replace or duplicate current joint producer.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['threshold.py','threshold_study.py']})
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Frozen threshold protocol changed')
    if not p.exists():write(p,protocol)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=1);a=p.parse_args();d=Data();declare(d)
    for o in range(6):(ROOT/str(o)).mkdir(parents=True,exist_ok=True)
    with ProcessPoolExecutor(max_workers=a.workers,initializer=init)as pool:
        for r in pool.map(job,range(6)):print('complete percentage gates',r,flush=True)
    c=[read(ROOT/str(o)/'complete.json')for o in range(6)];write(ROOT/'complete.json',dict(scopes=6,innerCalibrationBatches=sum(x['innerCalibrationBatches']for x in c),candidatePoliciesPerOuter=c[0]['candidatePolicies']))
