"""Bounded R6: genuine local activity, confirmed labels, matched weak feedback."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,time,argparse,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity import *
from activity_evaluate import evaluate
from competition import summary,local_metrics
from reanalyze_allowed import corrected
import checkpoint_io

ROOT=Path('research/local_feedback/results/activity-r6');CACHE=Path('.cache/local-feedback-activity-r6-study');D=None
ARMS=['all','local_only','whole_only','joint','no_extra','no_context','no_obligations']


def configurations():
    tasks=[(1.,e,w)for e,w in itertools.product([0.,1.,4.],[0.,1.,3.])]+[(0.,0.,1.)]
    return [dict(widthId=w,context=c,obligations=g,ridge=r,strongWeight=s,extraWeight=e,wholeWeight=b)
        for w,c,g,(s,e,b),r in itertools.product(range(3),[False,True],[False,True],tasks,[3.,.3,.03])]


def eligible(c,arm):
    if arm=='all':return True
    if arm=='local_only':return c['strongWeight']>0 and c['wholeWeight']==0
    if arm=='whole_only':return c['strongWeight']==0
    joint=c['strongWeight']>0 and c['wholeWeight']>0
    if arm=='joint':return joint
    if arm=='no_extra':return joint and c['extraWeight']==0
    if arm=='no_context':return joint and not c['context']
    if arm=='no_obligations':return joint and not c['obligations']
    raise ValueError(arm)


def mean_metrics(values):
    return{k:float(np.mean([v[k]for v in values]))for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']}


def score(r):
    m=mean_metrics(list(r['consumers'].values()));return .5*r['local']['accuracy']+.25*m['macroF1']+.125*m['microF1']+.125*m['sampleF1']


def select(rows,baseline,arm):
    good=[r for r in rows if eligible(r['config'],arm)and r['local']['accuracy']>baseline['local']['accuracy']+1e-12
        and r['local']['techCsConfusionMs']<=baseline['local']['techCsConfusionMs']+1e-8
        and all(r['consumers'][n][k]>=baseline['consumers'][n][k]-.01 for n in baseline['consumers']for k in ['macroPrecision','macroRecall','macroF1','microF1'])]
    peak=max([score(baseline)]+[score(r)for r in good]);near=[r for r in good if score(r)>=peak-.002]
    if not near:return baseline
    # Match the existing stable selector: prefer a broad, regularized well.
    return min(near,key=lambda r:(int(r['config']['context'])+int(r['config']['obligations']),-r['config']['ridge'],-score(r),r['configId']))


def initialize():
    global D
    D=Data()


def baseline(d,plan):
    ids=np.concatenate([d.main[s['validation']]for s in plan['inner']]);loc=corrected([r for s in plan['inner']for r in local_metrics(d,d.manual[s['strongValidation']])])
    return dict(configId=-1,config={},local=summary(loc),consumers={n:metrics(d.y[ids],p[ids],d.weight[ids])for n,p in d.fixed.items()})


def job(o):
    d=D;folder=ROOT/str(o);cache=CACHE/str(o);cache.mkdir(parents=True,exist_ok=True)
    if(folder/'inner-complete.json').exists():return o,'resumed'
    begin=time.monotonic();plan=PLANS[o];va=np.concatenate([d.main[s['validation']]for s in plan['inner']]);rows=[];warm={};kernels={};previous=None;cfgs=configurations()
    for ci,cfg in enumerate(cfgs):
        path=cache/f'config-{ci}.json'
        key=(cfg['widthId'],cfg['context'],cfg['obligations'])
        if key!=previous:kernels={};warm={};previous=key
        if checkpoint_io.exists(path):record=checkpoint_io.read(path)
        else:
            inner=[];out={n:[]for n in d.consumer_names};local=[]
            for j,s in enumerate(plan['inner']):
                if j not in kernels:kernels[j]=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg)
                k=kernels[j];k.set_config(cfg);wk=(j,cfg['strongWeight'],cfg['extraWeight'],cfg['wholeWeight']);theta,receipt=fit(k,warm=warm.get(wk));warm[wk]=theta
                q,loc=evaluate(k,theta,d.main[s['validation']],d.manual[s['strongValidation']]);local.extend(loc)
                for n in out:out[n].append(q[n])
                inner.append(dict(theta=theta.tolist(),receipt=receipt))
            consumers={n:metrics(d.y[va],np.concatenate(q),d.weight[va])for n,q in out.items()};row=dict(configId=ci,config=cfg,local=summary(local),consumers=consumers)
            record=dict(row=row,inner=inner,predictions={n:np.concatenate(q).tolist()for n,q in out.items()},localDetails=local);checkpoint_io.write(path,record)
        for j,saved in enumerate(record['inner']):warm[j,cfg['strongWeight'],cfg['extraWeight'],cfg['wholeWeight']]=np.array(saved['theta'])
        rows.append(record['row'])
        if ci%15==0:print('activity',o,ci,'/',len(cfgs),round(time.monotonic()-begin,1),'seconds',flush=True)
    base=baseline(d,plan);choices={arm:select(rows,base,arm)for arm in ARMS};write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,baseline=base,choices=choices))
    write(folder/'inner-complete.json',dict(configs=len(cfgs),innerFits=len(cfgs)*3,seconds=time.monotonic()-begin));return o,read(folder/'inner-complete.json')


def declare():
    protocol=dict(id='local-feedback-activity-r6',researchBaseline='v1.1.0',productControl='v1.2.3',configs=configurations(),plans=PLANS,
        hypothesis='Counterfactual real RC/LN experts are useful but a full old natural-frame router cannot distinguish activity changes inside a slot. Measure physical activity at1/2/4s on500ms intervals. Continuous future-tail retained-finger obligations distinguish staggered releases from indistinguishable simultaneous-held prefixes; guidance is combined with head/tail/occupancy context, never isolated Rbar.',
        study='Same349whole+6manual original six-by-three source/song splits. Active real E/F/G/I labels are now TRAIN-only observations restricted by the source song group. Synthetic B/C/CTX/LN cases are mechanism TRAIN constraints and separately reported, never called independent accuracy. Original local primary family supervision; concrete type experts remain real inherited parent candidates. Original fine-type geometry and Speed/Tech errors inaccessible to routing are reported, not claimed repaired.',
        controls='Full Cartesian three window widths, context yes/no, obligations yes/no, ridge3/.3/.03, extra labels0/1/4, whole feedback0/1/3. Whole-only fixes arbitrary task scale1. Local-only/whole-only/joint/no-extra/no-context/no-obligation each reselects its complete active parameter domain. No pseudo-label teacher. Smallest ridge.03 is explicit regularization floor rather than returning to previous unstable.003 peak.',
        readout='Same actual fixed Detailed/actions/Compact profiles. Mean of separately normalized native burden fields in TRAIN weak fine integral; auxiliary soft-family bag objective, actual MAP physical family coverage for evaluation. Old classic whole prior remains frozen. All old consumer marginals conserved to5.11e-15, no scorer calls.',
        selection='Pooled INNER common selection frozen before OUTER. Positive local mean accuracy gain, no allowed-set Tech/CS regression, each consumer macroP/R/F1/micro within1pp. Score.5local+.25meanMacroF1+.125meanMicroF1+.125meanSampleF1. Within.002 of best prefer fewer groups and stronger ridge. Frozen if no eligible gain. This is retrospective reused historical data, not a blind pipeline validation.',
        budget='360 configs x18 INNER=6480 fits, two starts per fit. No new family or enlarged search during this run. Exact optimized objective audited against original implementation. Source snapshots permit only separately audited numerical fixes, no silent hypothesis changes.',
        resource={k:v for k,v in read('research/local_feedback/results/activity-resource-fast-r6.json').items()if k!='source'},
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['activity.py','activity_evaluate.py','activity_study.py','release_obligations.cjs','augment_activity.cjs','prepare_obligations.py','test_activity.py','test_release_obligations.cjs']})
    path=ROOT/'protocol.json'
    if path.exists()and read(path)!=protocol:
        original=read(path);repair=read(ROOT/'storage-repair.json');before=original['sources']['activity_study.py'];after=protocol['sources']['activity_study.py']
        old=dict(original,sources=dict(original['sources']));new=dict(protocol,sources=dict(protocol['sources']));old['sources'].pop('activity_study.py');new['sources'].pop('activity_study.py')
        if old!=new or before!=repair['originalStudySource']or after!=repair['resumedStudySource']:raise ValueError('Frozen activity protocol changed beyond the audited storage repair')
    if not path.exists():write(path,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args();declare()
    if not args.declare_only:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize)as pool:
            for f in as_completed([pool.submit(job,o)for o in range(6)]):print('activity INNER complete',f.result(),flush=True)
        write(ROOT/'inner-complete.json',dict(scopes=6,configs=len(configurations()),innerFits=18*len(configurations())))
