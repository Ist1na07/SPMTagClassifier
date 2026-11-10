"""Select common/outer candidates before consuming their held-out outcomes."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route_consumers import Consumers,Kernel,fit,read,write,metrics,local_metrics,summary,corrected,routed_local,PLANS,ARMS,ROOT,study,select,mean_metrics


def run():
    read(ROOT/'inner-complete.json');d=Consumers();selections=[read(ROOT/str(o)/'selection.json')for o in range(6)];grids=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)];pooled=[]
    for ci,cfg in enumerate(study.configurations()):
        rs=[g[ci]for g in grids];consumers={n:mean_metrics([r['consumers'][n]for r in rs])for n in d.profiles};pooled.append(dict(configId=ci,config=cfg,consumers=consumers,whole=mean_metrics(list(consumers.values())),local={k:float(np.mean([r['local'][k]for r in rs]))for k in ['accuracy','techToDenseMs','techCsConfusionMs']}))
    b=dict(selections[0]['baseline']);b['consumers']={n:mean_metrics([s['baseline']['consumers'][n]for s in selections])for n in d.profiles};b['whole']=mean_metrics(list(b['consumers'].values()));b['local']={k:float(np.mean([s['baseline']['local'][k]for s in selections]))for k in pooled[0]['local']}
    common={arm:select(pooled,b,arm)for arm in ARMS};write(ROOT/'common-selection.json',dict(usesOuterTruth=False,baseline=b,choices=common,innerMeans=pooled))
    for o,plan in enumerate(PLANS):
        folder=ROOT/str(o);saved_models=read('research/local_feedback/results/routing-r3-allowed-set/'+str(o)+'/outer-models.json');models={};outputs={};locals={}
        for arm,row in selections[o]['choices'].items():
            ci=row['configId']
            if ci not in models:
                if ci<0:state=dict(model=None,receipt=None,predictions=d.fixed,local=corrected(local_metrics(d,d.manual[plan['strongTest']])),p=None)
                else:
                    k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],row['config'])
                    if str(ci)in saved_models:receipt=saved_models[str(ci)]['receipt'];theta=np.array(min(receipt['starts'],key=lambda x:x['loss'])['theta'])
                    else:theta,receipt=fit(k)
                    p,support,_,_,_=k.forward(theta,all_rows=True);state=dict(model=k.export(theta),receipt=receipt,predictions={n:d.consume(support,p,n)for n in d.profiles},local=corrected(routed_local(d,d.manual[plan['strongTest']],p)),p=p)
                models[ci]=state
            state=models[ci];outputs[arm]={n:[dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[c for c,v in zip(d.classes,q[i])if v])for i in d.main[plan['test']]]for n,q in state['predictions'].items()};locals[arm]=state['local']
        write(folder/'outer-models.json',{str(ci):dict(model=s['model'],receipt=s['receipt'])for ci,s in models.items()});write(folder/'outer-predictions.json',outputs);write(folder/'outer-local.json',locals);np.savez_compressed(folder/'probabilities.npz',**{str(ci):s['p']for ci,s in models.items()if ci>=0})
    baseline=dict(local=summary(corrected(local_metrics(d,d.manual))),consumers={n:metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in d.fixed.items()});result={}
    for arm in ARMS:
        ls=[];pred={n:np.zeros_like(d.baseline)for n in d.profiles}
        for o in range(6):
            ls.extend(read(ROOT/str(o)/'outer-local.json')[arm]);rows=read(ROOT/str(o)/'outer-predictions.json')[arm]
            for name,values in rows.items():
                for v in values:pred[name][d.lookup[v['uid']]]=[c in v['predicted']for c in d.classes]
        result[arm]=dict(local=summary(ls),consumers={n:metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in pred.items()})
    write(ROOT/'oof.json',dict(scope='Retrospective physical-family contract correction; local training and hyperparameters source-group cross-fitted, not a fresh blind study',baseline=baseline,metrics=result))
    for arm,row in common.items():
        if row['configId']<0:continue
        k=Kernel(d,d.main,d.manual,row['config']);theta,receipt=fit(k);p,support,_,_,_=k.forward(theta,all_rows=True)
        write(ROOT/f'candidate-{arm}.json',dict(model=k.export(theta),receipt=receipt,selection=row,replay={n:metrics(d.y[d.main],d.consume(support,p,n)[d.main],d.weight[d.main])for n in d.profiles},status='Not adopted until real-local regressions and integration pass'))
    print('baseline',baseline['local'],{n:round(m['macroF1']*100,2)for n,m in baseline['consumers'].items()},flush=True)
    for arm,v in result.items():print(arm,v['local'],{n:{k:round(m[k]*100,2)for k in ['macroPrecision','macroRecall','macroF1','microF1','exactMatch']}for n,m in v['consumers'].items()},'common',common[arm]['configId'],flush=True)
    write(ROOT/'complete.json',dict(scopes=6,reusedInnerFits=6480,newInnerFits=0,consumers=list(d.profiles)))


if __name__=='__main__':run()
