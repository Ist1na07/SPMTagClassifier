"""Reconcile physical family coverage with three real aggregation consumers.

An uncertain RC-versus-LN route is not a physical RC+LN mixture. Preserve the
original Hybrid gate meaning (coverage of selected families), while fine
prototype percentages still integrate the actual soft family-supported degree.
The fine integral retains its differentiable training path. Family bag loss is
an auxiliary confidence surrogate; no derivative through argmax is fabricated.
"""
from pathlib import Path
import sys,importlib.util,time
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from routing import Data,Kernel,fit,routed_local,read,write,metrics,local_metrics,summary,PLANS
from anchored import predictions
from reanalyze_allowed import corrected
spec=importlib.util.spec_from_file_location('route_consumer_parent',Path(__file__).with_name('routing_study.py'));study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
ROOT=Path('research/local_feedback/results/route-consumers-r5');CACHE=Path('.cache/local-feedback-route-consumers-r5');D=None
ARMS=['all','local_only','whole_only','joint','no_context','no_rating','pressure','no_teacher']


class Consumers(Data):
    def __init__(self):
        super().__init__();self.profiles=read('src/models/perception7k.json')['profiles'];self.masses={'detail':self.mass};meta=read('.cache/perception-extended-r2/manifest.json')
        with np.load('.cache/perception-extended-r2/arrays.npz')as z:self.masses['actions']=z['mass'][meta['weights'].index(self.profiles['actions']['recipe'])].copy()
        meta=read('.cache/perception-compact-r2/manifest.json')
        with np.load('.cache/perception-compact-r2/arrays.npz')as z:self.masses['compact']=z['mass'][meta['weights'].index(self.profiles['compact']['recipe'])].copy()
        self.fixed={}
        for name,mass in self.masses.items():
            share=np.zeros((len(self.records),len(self.fine)));family=np.zeros((len(self.records),2));np.add.at(share,self.arrays['bag'],mass[:,None]*self.support[:,:len(self.fine)]);np.add.at(family,self.arrays['bag'],mass[:,None]*np.eye(3)[self.route,1:]);p=self.profiles[name]
            self.fixed[name]=predictions(self,share,family,self.baseline,[p['policy']],[p['family']])[0,0]

    def consume(self,support,probabilities,name):
        mass=self.masses[name];share=np.zeros((len(self.records),len(self.fine)));family=np.zeros((len(self.records),2));np.add.at(share,self.arrays['bag'],mass[:,None]*support)
        # MAP family coverage has the same meaning as before the upgrade.
        np.add.at(family,self.arrays['bag'],mass[:,None]*np.eye(3)[probabilities.argmax(1),1:]);p=self.profiles[name]
        return predictions(self,share,family,self.baseline,[p['policy']],[p['family']])[0,0]


def initialize():
    global D
    D=Consumers()


def mean_metrics(values):
    return {k:float(np.mean([v[k]for v in values]))for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']}


def select(rows,baseline,arm):
    eligible=[r for r in rows if study.eligible(r['config'],arm)and r['local']['accuracy']>baseline['local']['accuracy']+1e-12
        and r['local']['techCsConfusionMs']<=baseline['local']['techCsConfusionMs']+1e-8
        and all(r['consumers'][n][k]>=baseline['consumers'][n][k]-.01 for n in baseline['consumers']for k in ['macroPrecision','macroRecall','macroF1','microF1'])]
    peak=max([study.score(baseline)]+[study.score(r)for r in eligible]);near=[r for r in eligible if study.score(r)>=peak-.002]
    if not near:return baseline
    return min(near,key=lambda r:({'absolute':0,'context':1,'pressure':2}[r['config']['features']],-r['config']['ridge'],r['config']['teacherWeight'],-study.score(r),r['configId']))


def job(o):
    d=D;folder=ROOT/str(o);folder.mkdir(parents=True,exist_ok=True);plan=PLANS[o];va=np.concatenate([d.main[s['validation']]for s in plan['inner']]);rows=[];kernels={};begin=time.monotonic()
    for ci,cfg in enumerate(study.configurations()):
        path=CACHE/str(o)/f'config-{ci}.json'
        if path.exists():row=read(path)
        else:
            saved=read(study.CACHE/str(o)/f'config-{ci}.json');outputs={n:[]for n in d.profiles}
            for j,s in enumerate(plan['inner']):
                key=(j,cfg['features'],cfg['priorStrength'])
                if key not in kernels:kernels[key]=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg)
                k=kernels[key];k.cfg=cfg;theta=np.array(saved['inner'][j]['theta']);p,support,_,_,_=k.forward(theta,all_rows=True)
                for name in outputs:outputs[name].append(d.consume(support,p,name)[d.main[s['validation']]])
            consumers={n:metrics(d.y[va],np.concatenate(q),d.weight[va])for n,q in outputs.items()};row=dict(configId=ci,config=cfg,local=summary(corrected(saved['localDetails'])),whole=mean_metrics(list(consumers.values())),consumers=consumers);write(path,row)
        rows.append(row)
    base_consumers={n:metrics(d.y[va],q[va],d.weight[va])for n,q in d.fixed.items()};baseline=dict(configId=-1,config=dict(features='frozen',strongWeight=0.,wholeWeight=0.,teacherWeight=0.,ridge=0.),local=summary(corrected([r for s in plan['inner']for r in local_metrics(d,d.manual[s['strongValidation']])])),consumers=base_consumers,whole=mean_metrics(list(base_consumers.values())))
    choices={arm:select(rows,baseline,arm)for arm in ARMS};write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,baseline=baseline,choices=choices));return o,dict(seconds=time.monotonic()-begin,configs=len(rows))


def declare():
    protocol=dict(researchBaseline='v1.1.0',productControl='v1.2.3',originalStudy='routing-r3',reusedInnerFits=360*18,newScorerCalls=0,
        reason=__doc__,selection='Each actual Detailed/actions/Compact consumer macroP/R/F1/micro within1pp on INNER, positive local accuracy gain and no allowed-set Tech/CS regression. Score uses their mean metrics with original local weight. Within.002, prefer fewer feature groups, stronger ridge, no teacher assumption. Each ablation reselects all active original parameters; frozen if no eligible gain.',
        retrospective='Contract/consumer correction after the original completed study; source-group cross-fitted local heads, same historical data, not blind confirmation.',sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['route_consumers.py','routing.py']})
    path=ROOT/'protocol.json'
    if path.exists()and read(path)!=protocol:raise ValueError('Consumer correction protocol changed')
    if not path.exists():write(path,protocol)


if __name__=='__main__':
    declare()
    with ProcessPoolExecutor(max_workers=6,initializer=initialize)as pool:
        for f in as_completed([pool.submit(job,o)for o in range(6)]):print('consumer reinterpretation',f.result(),flush=True)
    write(ROOT/'inner-complete.json',dict(scopes=6,reusedInnerFits=6480,newInnerFits=0))
