"""Frozen INNER choices, complete OUTER ablations and real source regressions."""
from pathlib import Path
import sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity_study import ROOT,CACHE,ARMS,configurations,select,mean_metrics,score
from activity import Data,Kernel,fit,read,write,metrics,PLANS
from activity_evaluate import evaluate
from competition import local_metrics,summary
from reanalyze_allowed import corrected
from human_regression import coverage
from rc_action_runtime import smooth


def average_local(rows):
    return{k:float(np.mean([r['local'][k]for r in rows]))for k in ['accuracy','techToDenseMs','techCsConfusionMs']}


def class_metrics(rows):
    counts={}
    for r in rows:
        for tag,v in r['perClass'].items():
            c=counts.setdefault(tag,dict(tp=0.,fp=0.,fn=0.))
            for key in c:c[key]+=v[key]
    for c in counts.values():
        tp,fp,fn=c['tp'],c['fp'],c['fn'];c.update(precision=tp/max(tp+fp,1e-30),recall=tp/max(tp+fn,1e-30),f1=2*tp/max(2*tp+fp+fn,1e-30))
    return counts


def frame_stats(d,frames,ids):
    switches=segments=short=0;durations=[]
    for i in ids:
        a,b=d.records[i]['offset'];fs=smooth(frames[a:b]);segments+=len(fs);short+=sum(f['end']-f['start']<1000 for f in fs);switches+=sum(u['tag']!=v['tag']for u,v in zip(fs,fs[1:]));durations.extend(f['end']-f['start']for f in fs)
    return dict(segments=segments,tagSwitches=switches,subsecondSegments=short,medianSegmentMs=float(np.median(durations)))


def human_cases(d,frames,groups=None):
    cases=[]
    for e in d.extra:
        if e['group']is None or(groups is not None and e['group']not in groups):continue
        i=e['index'];a,b=d.records[i]['offset'];start,end=e['start'],e['end'];fs=smooth(frames[a:b]);base=smooth([f for _,f in d.frame_records[a:b]])
        family=lambda ff:sum(max(0.,min(end,f['end'])-max(start,f['start']))for f in ff if f['family']in e['allowed'])/(end-start)
        cases.append(dict(id=e['id'],sourceUid=d.records[i]['uid'],sourceGroup=e['group'],allowedFamily=e['allowed'],baseline=coverage(base,start,end),candidate=coverage(fs,start,end),baselineFamilyCoverage=family(base),candidateFamilyCoverage=family(fs)))
    return cases


def mechanisms(d,frames,probabilities,support):
    expected={'B01':['break'],'B02':['release'],'C16':['speed'],'C24':['tech'],'CTX-L':['dense chordstream','fast chordstream'],'CTX-H':['break'],'LN-L':['release'],'LN-H':['release']};out=[]
    whole={name:d.consume(support,probabilities,name)for name in d.consumer_names}
    for name,tags in expected.items():
        i=d.lookup['query/'+name];a,b=d.records[i]['offset'];start,end=(9000.,17000.)if name.startswith(('CTX','LN-'))else(d.records[i]['frames'][0]['start'],d.records[i]['frames'][-1]['end']);base=coverage(smooth([f for _,f in d.frame_records[a:b]]),start,end);after=coverage(smooth(frames[a:b]),start,end)
        consumers={consumer:dict(tags=[c for c,v in zip(d.classes,whole[consumer][i])if v],percentages=dict(zip(d.fine,(100*np.sum(d.a['mass'][j,a:b,None]*support[a:b],axis=0)).tolist())))for j,consumer in enumerate(d.consumer_names)}
        out.append(dict(id=name,allowedPrimary=tags,baseline=base,candidate=after,allowedCoverageDelta=sum(after.get(t,0)for t in tags)-sum(base.get(t,0)for t in tags),wholeByConsumer=consumers,expectedOverallAbsent=['release']if name=='LN-H'else[]))
    return out


def run():
    read(ROOT/'inner-complete.json');d=Data();selections=[read(ROOT/str(o)/'selection.json')for o in range(6)];grids=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)];pooled=[]
    for ci,cfg in enumerate(configurations()):
        rs=[g[ci]for g in grids];pooled.append(dict(configId=ci,config=cfg,local=average_local(rs),consumers={n:mean_metrics([r['consumers'][n]for r in rs])for n in d.consumer_names}))
    bs=[s['baseline']for s in selections];baseline_row=dict(configId=-1,config={},local=average_local(bs),consumers={n:mean_metrics([r['consumers'][n]for r in bs])for n in d.consumer_names});common={arm:select(pooled,baseline_row,arm)for arm in ARMS}
    # No OUTER outcomes have been opened. Save common choices first.
    common_path=ROOT/'common-selection.json';choice=dict(usesOuterTruth=False,baseline=baseline_row,choices=common,innerMeans=pooled)
    if common_path.exists()and read(common_path)!=choice:raise ValueError('Common INNER choices changed')
    write(common_path,choice)
    for o,plan in enumerate(PLANS):
        folder=ROOT/str(o)
        if(folder/'outer-complete.json').exists():continue
        begin=time.monotonic();models={};outputs={};locals={};real={};display={};test=d.main[plan['test']];manual=d.manual[plan['strongTest']];testgroups={d.records[i]['group']for i in test}
        for arm,row in selections[o]['choices'].items():
            ci=row['configId']
            if ci not in models:
                if ci<0:state=dict(model=None,receipt=None,predictions={n:p[test]for n,p in d.fixed.items()},local=corrected(local_metrics(d,manual)),p=np.eye(3)[d.route])
                else:
                    k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],row['config']);path=CACHE/str(o)/f'outer-{ci}.json'
                    if path.exists():saved=read(path);receipt=saved['receipt'];theta=np.array(saved['theta'])
                    else:
                        theta,receipt=fit(k);write(path,dict(theta=theta.tolist(),receipt=receipt,model=k.export(theta)))
                    pred,loc=evaluate(k,theta,test,manual);p,*_=k.forward(theta,all_rows=True);state=dict(model=k.export(theta),receipt=receipt,predictions=pred,local=loc,p=p)
                frames=d.routed_scores(state['p'])if ci>=0 else[f for _,f in d.frame_records];state['real']=human_cases(d,frames,testgroups);state['display']=frame_stats(d,frames,test);models[ci]=state
            state=models[ci];outputs[arm]={n:[dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[c for c,v in zip(d.classes,q[j])if v])for j,i in enumerate(test)]for n,q in state['predictions'].items()};locals[arm]=state['local'];real[arm]=state['real'];display[arm]=state['display']
        write(folder/'outer-models.json',{str(ci):dict(model=s['model'],receipt=s['receipt'])for ci,s in models.items()});write(folder/'outer-predictions.json',outputs);write(folder/'outer-local.json',locals);write(folder/'outer-real.json',real);write(folder/'outer-display.json',display)
        np.savez_compressed(folder/'probabilities.npz',**{str(ci):s['p']for ci,s in models.items()if ci>=0});write(folder/'outer-complete.json',dict(outerFits=sum(ci>=0 for ci in models),seconds=time.monotonic()-begin));print('activity OUTER',o,'fits',len(models),'seconds',round(time.monotonic()-begin,1),flush=True)
    base_local=corrected(local_metrics(d,d.manual));baseline=dict(local=summary(base_local),localDetails=base_local,localPerClass=class_metrics(base_local),consumers={n:metrics(d.y[d.main],p[d.main],d.weight[d.main])for n,p in d.fixed.items()},display=frame_stats(d,[f for _,f in d.frame_records],d.main));results={}
    for arm in ARMS:
        ls=[];pred={n:np.zeros_like(d.baseline)for n in d.consumer_names};real=[];displays=[]
        for o in range(6):
            ls.extend(read(ROOT/str(o)/'outer-local.json')[arm]);real.extend(read(ROOT/str(o)/'outer-real.json')[arm]);displays.append(read(ROOT/str(o)/'outer-display.json')[arm])
            for n,rows in read(ROOT/str(o)/'outer-predictions.json')[arm].items():
                for r in rows:pred[n][d.lookup[r['uid']]]=[c in r['predicted']for c in d.classes]
        results[arm]=dict(local=summary(ls),localDetails=ls,localPerClass=class_metrics(ls),consumers={n:metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in pred.items()},realSourceHeldOut=real,display={k:sum(v[k]for v in displays)for k in ['segments','tagSwitches','subsecondSegments']})
    write(ROOT/'oof.json',dict(scope='Historical reused six source-song groups; new local head and active real labels source-held-out. Not a fresh blind full-pipeline validation. Macro of six manual-chart accuracies, exact duration metrics and actual three consumers.',baseline=baseline,metrics=results))
    candidates={}
    for arm,row in common.items():
        ci=row['configId']
        if ci<0:continue
        path=ROOT/f'candidate-{arm}.json'
        if path.exists():continue
        if ci not in candidates:
            k=Kernel(d,d.main,d.manual,row['config']);theta,receipt=fit(k);p,support,*_=k.forward(theta,all_rows=True);frames=d.routed_scores(p);candidates[ci]=dict(model=k.export(theta),receipt=receipt,replay={n:metrics(d.y[d.main],d.consume(support,p,n)[d.main],d.weight[d.main])for n in d.consumer_names},mechanisms=mechanisms(d,frames,p,support),realTrainFit=human_cases(d,frames),display=frame_stats(d,frames,d.main))
        write(path,dict(**candidates[ci],selection=row,status='Research candidate, not adopted; common full TRAIN replay and mechanism constraints are not validation.'))
    print('baseline',baseline['local'],flush=True)
    for arm,r in results.items():print(arm,r['local'],{n:round(m['macroF1']*100,2)for n,m in r['consumers'].items()},'common',common[arm]['configId'],flush=True)
    write(ROOT/'complete.json',dict(scopes=6,configs=len(configurations()),innerFits=len(configurations())*18,outerFits=sum(read(ROOT/str(o)/'outer-complete.json')['outerFits']for o in range(6))))


if __name__=='__main__':run()
