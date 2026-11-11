"""Matched whole-feedback/strength ablation on the actual original expert.

Training uses the validated smooth32 normalization surrogate, while ALL
selection metrics use exact archived classification and native integral gates.
"""
from joint_core import *
from local_study import PLANS,subset_metric,mean_metrics
import itertools,argparse
from concurrent.futures import ProcessPoolExecutor

ROOT=Path('research/core_retune/results/joint-source-r1')
CONFIGS=[dict(wholeWeight=weight,learnGain=free,ridge=r)for weight,free,r in itertools.product([0.,1.],[False,True],[.1,.01,.001])]
D=W=None


def init():
    global D,W
    D=Data();W=Whole(D)


def scope(split,train=True):
    key='train'if train else'validation';strong='strongTrain'if train else'strongValidation';ids=[W.main[i]for i in split[key]];groups={W.records[i]['group']for i in ids};ms=[D.manual[i]for i in split[strong]];extras=[e for e in D.extra if D.records[e['uid']]['group']in groups];return ids,set(ms)|{e['uid']for e in extras},ms,[e['id']for e in extras]


def train(c,split):
    ids,uids,_,_=scope(split);changes={};gains={};models={};receipts={}
    for family in ['rc','ln']:
        k=JointKernel(D,W,family,uids,ids,ridge=c['ridge'],whole_weight=c['wholeWeight'],learn_gain=c['learnGain'],normalization='smooth32');theta,receipts[family]=fit(k)
        if any(not np.isfinite(r['loss'])or r['maximumGradient']>1e-5 for r in receipts[family]):raise ValueError('Unaccepted smooth fit '+str(c)+' '+family+' '+str(receipts[family]))
        changes[family]=k.changes(theta);models[family]=k.export(theta);gains[family]=models[family]['scoreGain']
    return D.frames(changes,gains=gains),models,receipts


def evaluate(frames,split):
    ids,_,ms,es=scope(split,False);return dict(local=subset_metric(D,frames,ms,es),consumers=W.evaluate(frames,ids))


def pooled(rows):
    local=mean_metrics([r['local']for r in rows]);mean={k:float(np.mean([v[k]for r in rows for v in r['consumers'].values()]))for k in ['macroF1','macroPrecision','macroRecall']};return dict(local=local,consumers=mean,score=.5*local['objective']+.25*mean['macroF1']+.125*mean['macroPrecision']+.125*mean['macroRecall'])


def key(r):return(r['metrics']['score'],-r['metrics']['local']['techCsConfusionMs'],r['config']['ridge']if r['config']else 1.,-r['id'])


def inner_job(outer):
    folder=ROOT/str(outer);plan=PLANS[outer];baseline=pooled([evaluate(D.baseline,s)for s in plan['inner']]);rows=[]
    for ci,c in enumerate(CONFIGS):
        path=folder/('config-'+str(ci)+'.json')
        if path.exists():row=read(path)
        else:
            details=[];fits=[]
            for j,split in enumerate(plan['inner']):
                tr,tu,_,_=scope(split);va,vu,_,_=scope(split,False)
                if tu&vu or{W.records[i]['group']for i in tr}&{W.records[i]['group']for i in va}:raise ValueError('Source leakage')
                frames,models,receipts=train(c,split);details.append(evaluate(frames,split));fits.append(dict(inner=j,trainUids=sorted(tu),models=models,receipts=receipts))
            row=dict(id=ci,config=c,metrics=pooled(details),details=details,fits=fits);write(path,row)
        rows.append(row);print('joint original core',outer,ci,row['metrics'],flush=True)
    base=dict(id=-1,config=None,metrics=baseline);choices={}
    for weight,free in itertools.product([0.,1.],[False,True]):
        arm=('joint'if weight else'local')+('/free'if free else'/fixed');eligible=[r for r in rows if r['config']['wholeWeight']==weight and r['config']['learnGain']==free];choices[arm]=dict(candidate=max(eligible,key=key),product=max([base]+eligible,key=key))
    write(folder/'selection.json',dict(usesOuterTruth=False,baseline=base,choices=choices));return outer


def run(workers,resume_repair=False):
    protocol=dict(id='joint-original-core-source-r1',researchBaseline='v1.1.0',productControl='v1.2.4',configs=CONFIGS,plans=PLANS,
      training='Direct original b/W in TRAIN-scaled original56D coordinates and Helmert class contrasts; ridge all effective coordinates. Original score gain1.4 fixed versus learned log gain. Local strong allowed sets plus native-duration bags; three-consumer integral whole-label loss. No original-affinity-sum conservation. Smooth32 is a conservative TRAIN surrogate for hard max/cap; no separate chart head. Family and early route fixed, actual late rest gate recomputed at evaluation. All weak labels and scalers source-TRAIN only.',
      selection='INNER .5*(.75 manual + .25 real local when available) + .25 mean macroF1 + .125 macroPrecision + .125 macroRecall; less Tech/CS then stronger ridge breaks ties. Four arms independently tune ridge; report both forced best candidate and product selection allowing baseline. All INNER saved before any OUTER consumption. Not blind: historical source data and earlier diagnoses reused.',
      source={p:Path(__file__).with_name(p).read_text(encoding='utf8')for p in ['core.py','joint_core.py','joint_study.py']})
    previous=read(ROOT/'protocol.json')if(ROOT/'protocol.json').exists()else None
    if previous is not None and previous!=protocol:
        if not resume_repair or{k:v for k,v in previous.items()if k!='source'}!={k:v for k,v in protocol.items()if k!='source'}or not read(ROOT/'numeric-reuse-check.json')['allReusable']:raise ValueError('Frozen joint protocol changed without validated arithmetic-only repair')
        write(ROOT/'execution-repair.json',dict(reason='Stable log-domain evaluation of the same local bag log-mass ratio; models/labels/grid/selection unchanged. Saved models all checked in original coordinates; initial protocol preserved.',sources=protocol['source'],reuse=read(ROOT/'numeric-reuse-check.json')))
    elif previous is None:write(ROOT/'protocol.json',protocol)
    if workers>1:
        with ProcessPoolExecutor(max_workers=workers,initializer=init)as pool:list(pool.map(inner_job,range(6)))
    else:
        init()
        for o in range(6):inner_job(o)
    write(ROOT/'inner-complete.json',dict(configs=len(CONFIGS),innerScopes=18,familyFits=432,starts=864));init();result={}
    # Freeze common pooled INNER choice before reading or fitting OUTER.
    common=[]
    for ci,c in enumerate(CONFIGS):
        rows=[read(ROOT/str(o)/('config-'+str(ci)+'.json'))for o in range(6)];common.append(dict(id=ci,config=c,metrics=pooled([v for r in rows for v in r['details']])))
    write(ROOT/'common-inner.json',common)
    selections=[read(ROOT/str(o)/'selection.json')for o in range(6)];arms=list(selections[0]['choices']);outcomes={arm+'/'+mode:[]for arm in arms for mode in ['candidate','product']}
    for outer,plan in enumerate(PLANS):
        split=dict(train=plan['train'],validation=plan['test'],strongTrain=plan['strongTrain'],strongValidation=plan['strongTest']);states={-1:(D.baseline,{},{} )};saved={}
        for arm in arms:
            for mode in ['candidate','product']:
                chosen=selections[outer]['choices'][arm][mode];ci=chosen['id']
                if ci not in states:states[ci]=train(chosen['config'],split)
                frames,models,receipts=states[ci];score=evaluate(frames,split);name=arm+'/'+mode;outcomes[name].append(score);saved[name]=dict(choice=chosen,metrics=score,models=models,receipts=receipts)
        write(ROOT/str(outer)/'outer.json',saved)
    for name,rows in outcomes.items():result[name]=dict(metrics=pooled(rows),details=rows)
    baseline=pooled([evaluate(D.baseline,dict(train=p['train'],validation=p['test'],strongTrain=p['strongTrain'],strongValidation=p['strongTest']))for p in PLANS]);write(ROOT/'oof.json',dict(scope=protocol['training'],baseline=baseline,arms=result));print('JOINT ORIGINAL CORE COMPLETE',baseline,{k:v['metrics']for k,v in result.items()},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=1);parser.add_argument('--resume-numeric-repair',action='store_true');args=parser.parse_args();run(args.workers,args.resume_numeric_repair)
