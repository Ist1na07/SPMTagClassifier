"""Consume frozen joint INNER choices, never choose by outer/replay success."""
from pathlib import Path
import os,sys,time
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from joint_study import ROOT,CACHE,ARMS,configurations,select,mean_consumers,score,cases,local_summary
from joint_activity import Data,Kernel,fit,PLANS,read,write,local,parent
from competition import local_metrics
from reanalyze_allowed import corrected
from activity_report import frame_stats,class_metrics


def pooled_local(rows):
    out={}
    for field,count in [('accuracy','observations'),('manualAccuracy','manualObservations'),('realAccuracy','realObservations')]:
        n=sum(r['local'][count]for r in rows);out[field]=sum((r['local'][field]or 0)*r['local'][count]for r in rows)/n if n else None
    for field in ['observations','manualObservations','realObservations']:out[field]=sum(r['local'][field]for r in rows)
    for field in ['techToDenseMs','techCsConfusionMs']:out[field]=float(np.mean([r['local'][field]for r in rows]))
    return out


def run():
    read(ROOT/'inner-complete.json');d=Data();grids=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)];selections=[read(ROOT/str(o)/'selection.json')for o in range(6)];pool=[]
    for ci,cfg in enumerate(configurations()):
        rs=[g[ci]for g in grids];pool.append(dict(configId=ci,config=cfg,local=pooled_local(rs),consumers={n:mean_consumers([r['consumers'][n]for r in rs])for n in d.consumer_names},mechanismPass=all(r['mechanismPass']for r in rs)))
    bases=[s['baseline']for s in selections];base=dict(configId=-1,config={},local=pooled_local(bases),consumers={n:mean_consumers([r['consumers'][n]for r in bases])for n in d.consumer_names},mechanismPass=False);common={arm:select(pool,base,arm)for arm in ARMS};choice=dict(usesOuterTruth=False,choices=common,baseline=base,innerMeans=pool)
    if(ROOT/'common-selection.json').exists()and read(ROOT/'common-selection.json')!=choice:raise ValueError('Frozen common joint selection changed')
    write(ROOT/'common-selection.json',choice)
    raw=[f for _,f in d.frame_records]
    zero=Kernel(d,d.main,d.manual,configurations()[0]);zf,zp,zs=zero.frames(np.zeros(zero.dimension));zm=local(d,d.manual,zf);zr=cases(d,zf,[e for e in d.extra if e['group']is not None]);structured=dict(scope='No fitted coefficients: physical availability, fixed old family prior and unchanged type affinities with the new head readout. Separate this architecture effect from learning.',local=local_summary(zm,zr),manual=zm,real=zr,consumers={n:parent.metrics(d.y[d.main],d.consume(zs,zp,n)[d.main],d.weight[d.main])for n in d.consumer_names})
    for o,plan in enumerate(PLANS):
        folder=ROOT/str(o)
        if(folder/'outer-complete.json').exists():continue
        begin=time.monotonic();models={};outputs={};details={};test=d.main[plan['test']];manual=d.manual[plan['strongTest']];groups={d.records[i]['group']for i in test};obs=[e for e in d.extra if e['group']in groups]
        for arm,row in selections[o]['choices'].items():
            ci=row['configId']
            if ci not in models:
                if ci<0:state=dict(model=None,receipt=None,predictions=d.fixed,manual=corrected(local_metrics(d,manual)),real=cases(d,raw,obs),display=frame_stats(d,raw,test),mechanisms=cases(d,raw,[e for e in d.extra if e['group']is None]))
                else:
                    k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],row['config']);path=CACHE/str(o)/f'outer-{ci}.json'
                    if path.exists():saved=read(path);theta=np.array(saved['theta']);receipt=saved['receipt']
                    else:theta,receipt=fit(k);write(path,dict(theta=theta.tolist(),receipt=receipt,model=k.export(theta)))
                    frames,p,support=k.frames(theta);state=dict(model=k.export(theta),receipt=receipt,predictions={n:d.consume(support,p,n)for n in d.consumer_names},manual=local(d,manual,frames),real=cases(d,frames,obs),display=frame_stats(d,frames,test),mechanisms=cases(d,frames,[e for e in d.extra if e['group']is None]))
                models[ci]=state
            state=models[ci];outputs[arm]={n:[dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[c for c,v in zip(d.classes,q[i])if v])for i in test]for n,q in state['predictions'].items()};details[arm]={k:state[k]for k in ['manual','real','display','mechanisms']}
        write(folder/'outer-models.json',{str(ci):dict(model=s['model'],receipt=s['receipt'])for ci,s in models.items()});write(folder/'outer-predictions.json',outputs);write(folder/'outer-local.json',details);write(folder/'outer-complete.json',dict(outerFits=sum(ci>=0 for ci in models),seconds=time.monotonic()-begin));print('joint OUTER',o,'fits',len(models),'seconds',round(time.monotonic()-begin,1),flush=True)
    baseline_manual=corrected(local_metrics(d,d.manual));baseline_real=cases(d,raw,[e for e in d.extra if e['group']is not None]);baseline=dict(local=local_summary(baseline_manual,baseline_real),manual=baseline_manual,real=baseline_real,manualPerClass=class_metrics(baseline_manual),consumers={n:parent.metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in d.fixed.items()},display=frame_stats(d,raw,d.main));results={}
    for arm in ARMS:
        manual=[];real=[];display=[];pred={n:np.zeros_like(d.baseline)for n in d.consumer_names}
        for o in range(6):
            detail=read(ROOT/str(o)/'outer-local.json')[arm];manual.extend(detail['manual']);real.extend(detail['real']);display.append(detail['display'])
            for name,rows in read(ROOT/str(o)/'outer-predictions.json')[arm].items():
                for r in rows:pred[name][d.lookup[r['uid']]]=[c in r['predicted']for c in d.classes]
        if len({r['id']for r in real})!=len(real)or len(real)!=len(baseline_real):raise ValueError('Real source evaluation duplicated or omitted a case')
        results[arm]=dict(local=local_summary(manual,real),manual=manual,real=real,manualPerClass=class_metrics(manual),consumers={n:parent.metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in pred.items()},display={key:sum(v[key]for v in display)for key in ['segments','tagSwitches','subsecondSegments']})
    write(ROOT/'oof.json',dict(scope='Original6 manual charts plus8 actively queried real passages reported separately and equally by passage in combined accuracy. Historical data, new parameters/source queries source-held-out, not a blind whole-pipeline benchmark. Product adds fixed mechanism-dev checks; matched joint/local controls use identical selection without that adoption condition.',baseline=baseline,structuredZeroCorrection=structured,metrics=results))
    models={}
    for arm,row in common.items():
        ci=row['configId']
        if ci<0:continue
        path=ROOT/f'candidate-{arm}.json'
        if path.exists():continue
        if ci not in models:
            k=Kernel(d,d.main,d.manual,row['config']);theta,receipt=fit(k);frames,p,support=k.frames(theta);pred={n:d.consume(support,p,n)for n in d.consumer_names};mechanisms=cases(d,frames,[e for e in d.extra if e['group']is None]);whole=[]
            for e in d.extra:
                if e['group']is not None:continue
                i=e['index'];a,b=d.records[i]['offset'];whole.append(dict(id=e['id'],consumers={n:dict(tags=[c for c,v in zip(d.classes,pred[n][i])if v],percentages=dict(zip(d.fine,(100*np.sum(d.a['mass'][j,a:b,None]*support[a:b],axis=0)).tolist())))for j,n in enumerate(d.consumer_names)}))
            models[ci]=dict(model=k.export(theta),receipt=receipt,mechanisms=mechanisms,mechanismWhole=whole,realTrainFit=cases(d,frames,[e for e in d.extra if e['group']is not None]),manualTrainFit=local(d,d.manual,frames),replay={n:parent.metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in pred.items()},display=frame_stats(d,frames,d.main))
        write(path,dict(**models[ci],selection=row,status='Not adopted; full TRAIN replay and mechanism-dev cases are not held-out performance. Native/overlay and final acceptance required.'))
    print('baseline',baseline['local'],flush=True)
    for arm,r in results.items():print(arm,r['local'],{n:round(m['macroF1']*100,2)for n,m in r['consumers'].items()},'common',common[arm]['configId'],flush=True)
    write(ROOT/'complete.json',dict(scopes=6,configs=len(configurations()),innerFits=18*len(configurations()),outerFits=sum(read(ROOT/str(o)/'outer-complete.json')['outerFits']for o in range(6))))


if __name__=='__main__':run()
