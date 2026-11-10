"""Complete independently tuned route/local/whole feedback comparisons."""
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
from pathlib import Path
import sys,time,argparse,itertools
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from routing import Data,Kernel,fit,routed_local,read,write,metrics,local_metrics,summary,PLANS

ROOT=Path('research/local_feedback/results/routing-r3');CACHE=Path('.cache/local-feedback-routing-r3');D=None
ARMS=['all','local_only','whole_only','joint','no_context','no_rating','pressure','no_teacher']


def configurations():
    tasks=[(1.,0.),(0.,1.),(1.,.3),(1.,1.),(1.,3.)]
    return [dict(features=f,priorStrength=p,strongWeight=s,wholeWeight=w,teacherWeight=t,ridge=r)
        for f,p,(s,w),t,r in itertools.product(['absolute','context','pressure'],[2.,4.,6.],tasks,[0.,1.],[3.,.3,.03,.003])]


def eligible(c,arm):
    if arm=='all':return True
    if arm=='local_only':return c['strongWeight']>0 and c['wholeWeight']==0
    if arm=='whole_only':return c['strongWeight']==0 and c['wholeWeight']>0
    joint=c['strongWeight']>0 and c['wholeWeight']>0
    if arm=='joint':return joint
    if arm=='no_context':return joint and c['features']=='absolute'
    if arm=='no_rating':return joint and c['features']=='context'
    if arm=='pressure':return joint and c['features']=='pressure'
    if arm=='no_teacher':return joint and c['teacherWeight']==0
    raise ValueError(arm)


def score(r):
    m=r['whole'];return .5*r['local']['accuracy']+.25*m['macroF1']+.125*m['microF1']+.125*m['sampleF1']


def select(rows,baseline,arm,protected):
    available=[r for r in rows if eligible(r['config'],arm)]
    if protected:
        available=[r for r in available if r['local']['accuracy']>=baseline['local']['accuracy']-1e-12
            and r['local']['techCsConfusionMs']<=baseline['local']['techCsConfusionMs']+1e-8
            and all(r['whole'][k]>=baseline['whole'][k]-.01 for k in ['macroPrecision','macroRecall','macroF1','microF1'])]
        available.append(baseline)
    return max(available,key=lambda r:(score(r),-r['local']['techCsConfusionMs'],r['whole']['macroRecall'],-r['configId']))


def initialize():
    global D
    D=Data()


def job(o):
    d=D;folder=ROOT/str(o);cache=CACHE/str(o);cache.mkdir(parents=True,exist_ok=True)
    if(folder/'complete.json').exists():return o,'resumed'
    begin=time.monotonic();plan=PLANS[o];va=np.concatenate([d.main[s['validation']]for s in plan['inner']]);old,_=d.chart_predictions(d.support);loc=[r for s in plan['inner']for r in local_metrics(d,d.manual[s['strongValidation']])]
    baseline=dict(configId=-1,config=dict(features='frozen',strongWeight=0.,wholeWeight=0.,teacherWeight=0.,ridge=0.),whole=metrics(d.y[va],old[va],d.weight[va]),local=summary(loc))
    rows=[];ps=[];warm={};kernels={};cfgs=configurations()
    for ci,c in enumerate(cfgs):
        path=cache/f'config-{ci}.json'
        if path.exists():record=read(path)
        else:
            inner=[];predictions=[];locals=[]
            for j,s in enumerate(plan['inner']):
                key=(j,c['features'],c['priorStrength']);wkey=(key,c['strongWeight'],c['wholeWeight'],c['teacherWeight'])
                if key not in kernels:kernels[key]=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],c)
                k=kernels[key];k.cfg=c;theta,receipt=fit(k,warm=warm.get(wkey));warm[wkey]=theta
                p,support,_,_,_=k.forward(theta,all_rows=True);pred,_,_=d.chart_predictions_routed(support,p);predictions.append(pred[d.main[s['validation']]]);locals.extend(routed_local(d,d.manual[s['strongValidation']],p))
                inner.append(dict(theta=theta.tolist(),receipt=receipt))
            q=np.concatenate(predictions);row=dict(configId=ci,config=c,whole=metrics(d.y[va],q,d.weight[va]),local=summary(locals));record=dict(row=row,inner=inner,predictions=q.tolist(),localDetails=locals);write(path,record)
        for j,r in enumerate(record['inner']):warm[((j,c['features'],c['priorStrength']),c['strongWeight'],c['wholeWeight'],c['teacherWeight'])]=np.array(r['theta'])
        rows.append(record['row']);ps.append(record['predictions'])
        if ci%24==0:print('routing',o,ci,'/',len(cfgs),round(time.monotonic()-begin,1),'seconds',flush=True)
    choices={arm+'/'+guard:select(rows,baseline,arm,guard=='protected')for arm in ARMS for guard in ['balanced','protected']}
    write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,baseline=baseline,choices=choices));np.savez_compressed(folder/'inner-vectors.npz',ids=va,predictions=np.array(ps,bool))
    models={};output={};local={};test=d.main[plan['test']]
    for name,r in choices.items():
        ci=r['configId']
        if ci not in models:
            if ci<0:models[ci]=dict(model=None,receipt=None,probability=None);pred=old;share=d.chart_predictions(d.support)[1]
            else:
                k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],r['config']);theta,receipt=fit(k);p,support,_,_,_=k.forward(theta,all_rows=True);pred,share,_=d.chart_predictions_routed(support,p);models[ci]=dict(model=k.export(theta),receipt=receipt,probability=p)
            models[ci]['predictions']=pred;models[ci]['share']=share
        state=models[ci];pred=state['predictions'];share=state['share']
        output[name]=[dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[c for c,v in zip(d.classes,pred[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))for i in test]
        local[name]=local_metrics(d,d.manual[plan['strongTest']])if ci<0 else routed_local(d,d.manual[plan['strongTest']],state['probability'])
    write(folder/'outer-models.json',{str(ci):dict(model=s['model'],receipt=s['receipt'])for ci,s in models.items()});write(folder/'outer-predictions.json',output);write(folder/'outer-local.json',local)
    np.savez_compressed(folder/'route-probabilities.npz',**{str(ci):s['probability']for ci,s in models.items()if ci>=0})
    write(folder/'complete.json',dict(configs=len(cfgs),innerFits=len(cfgs)*3,outerFits=sum(ci>=0 for ci in models),seconds=time.monotonic()-begin));return o,read(folder/'complete.json')


def declare():
    d=Data();protocol=dict(id='local-feedback-routing-r3',researchBaseline='v1.1.0',fixedProductControl='v1.2.3',configs=configurations(),plans=PLANS,readout=d.profile,
        evidence=read('research/local_feedback/results/learnability-r2.json')['totalsMs'],resource=read('research/local_feedback/results/routing-resource-r3.json'),
        hypothesis='About46% of baseline annotated wrong time is blocked by Break/family routing. Learn a3-way context/pressure router with actual counterfactual experts rather than continuing to tune an incapable within-family head.',
        controls='Absolute structural inputs; +relative within-map head/row/tail rates and occupancy and hand geometry; +actual SPM demand/action/component contrasts. Local-only/whole-only/joint, priorStrength2/4/6,ridge4levels,teacher0/1,whole feedback.3/1/3 independently reselected. Whole-only fixes its arbitrary overall loss scale1; only ratios are identifiable.',
        targets='Same original349 and6manual only, original6x3sourcegroups. Partial local allowed-family labels on exact overlap. Whole fine BCE ordinary only; Mix hidden primitives unknown. Family targets come from marked main-impression tag families, Hybrid both. Fine and family task means averaged. No E/F/G/I/constructed labels fitted.',
        representation='Real RC/LN expert candidates extracted for ALL natural intervals including old Break. Soft route probability multiplies family-specific bounded expert support; family probability is also burden-integrated for the actual Hybrid gate. No scorer calls. Local visible route uses argmax then the actual original type expert, not a copied whole label.',
        selection='Same frozen local/whole objective and protected1pp INNER budget as competition-r2. Common config mean INNER before reading OUTER. Fixed prior/readout historical; new route parameters source-held-out, no blind full-pipeline claim.',
        optimizer='Whitened TRAIN-only features, Helmert route contrasts, explicit ridge, two starts with exact-scope larger-ridge continuation, every accepted gradient<=1e-5.',
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['routing.py','routing_study.py','extract_routing.cjs']})
    path=ROOT/'protocol.json'
    if path.exists()and read(path)!=protocol:raise ValueError('Frozen routing study changed')
    if not path.exists():write(path,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args();declare()
    if not args.declare_only:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize)as pool:
            for f in as_completed([pool.submit(job,o)for o in range(6)]):print('complete routing',f.result(),flush=True)
        cs=[read(ROOT/str(o)/'complete.json')for o in range(6)];write(ROOT/'complete.json',dict(scopes=6,innerFits=sum(c['innerFits']for c in cs),outerFits=sum(c['outerFits']for c in cs)))
