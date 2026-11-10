"""Repair a concrete allowed-set metric bug without re-running valid INNER fits.

If an annotation permits both Tech and a CS type, either permitted prediction
is correct. It must not simultaneously count as a Tech/CS confusion.
Original results/protocols are retained; corrected selections have new paths.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import competition as c
import routing as r


def corrected(rows):
    out=[]
    for old in rows:
        v=dict(old);td=conf=0.
        for key,dt in old['transitions'].items():
            truth,pred=key.split(' -> ');tags=truth.split('|')
            if pred in tags:continue
            td+=dt*('tech'in tags and pred=='dense chordstream')
            conf+=dt*(('tech'in tags and pred in ['dense chordstream','fast chordstream'])or(pred=='tech'and any(t in tags for t in ['dense chordstream','fast chordstream'])))
        v['techToDenseMs']=td;v['techCsConfusionMs']=conf;out.append(v)
    return out


def run(kind):
    module=c if kind=='competition'else r
    name='study.py'if kind=='competition'else'routing_study.py';spec=importlib.util.spec_from_file_location('allowed_study_'+kind,Path(__file__).with_name(name));study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
    upstream=study.ROOT;root=upstream.with_name(upstream.name+'-allowed-set');d=module.Data();old_pred,_=d.chart_predictions(d.support)
    selections=[];grids=[];examples=[]
    for o,plan in enumerate(c.PLANS):
        records=[c.read(Path(study.CACHE)/str(o)/f'config-{ci}.json')for ci in range(len(study.configurations()))];rows=[]
        for saved in records:
            row=dict(saved['row']);details=corrected(saved['localDetails']);row['local']=c.summary(details);rows.append(row)
        prior=c.read(upstream/str(o)/'selection.json')['baseline'];baseline=dict(prior);ls=[z for s in plan['inner']for z in c.local_metrics(d,d.manual[s['strongValidation']])];baseline['local']=c.summary(corrected(ls))
        choice={arm+'/'+guard:study.select(rows,baseline,arm,guard=='protected')for arm in study.ARMS for guard in ['balanced','protected']};selections.append(dict(usesOuterTruth=False,baseline=baseline,choices=choice));grids.append(rows)
        c.write(root/str(o)/'inner-grid.json',rows);c.write(root/str(o)/'selection.json',selections[-1])
    # Freeze corrected common selection BEFORE consuming any OUTER results.
    pooled=[]
    for ci,cfg in enumerate(study.configurations()):
        rows=[g[ci]for g in grids];pooled.append(dict(configId=ci,config=cfg,whole={k:float(np.mean([v['whole'][k]for v in rows]))for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']},local={k:float(np.mean([v['local'][k]for v in rows]))for k in ['accuracy','techToDenseMs','techCsConfusionMs']}))
    b=dict(selections[0]['baseline']);b['whole']={k:float(np.mean([s['baseline']['whole'][k]for s in selections]))for k in pooled[0]['whole']};b['local']={k:float(np.mean([s['baseline']['local'][k]for s in selections]))for k in pooled[0]['local']}
    common={arm+'/'+guard:study.select(pooled,b,arm,guard=='protected')for arm in study.ARMS for guard in ['balanced','protected']};c.write(root/'common-selection.json',dict(usesOuterTruth=False,baseline=b,choices=common))
    for o,plan in enumerate(c.PLANS):
        accepted=c.read(upstream/str(o)/'outer-models.json');models={};output={};local={}
        for name,row in selections[o]['choices'].items():
            ci=row['configId']
            if ci not in models:
                if ci<0:models[ci]=dict(model=None,receipt=None,pred=old_pred,locals=corrected(c.local_metrics(d,d.manual[plan['strongTest']])))
                else:
                    k=module.Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],row['config'])
                    if str(ci)in accepted:
                        model=accepted[str(ci)];receipt=model['receipt'];theta=np.array(min(receipt['starts'],key=lambda x:x['loss'])['theta'])
                    else:theta,receipt=module.fit(k)
                    if kind=='competition':
                        support,scores,_,_=k.forward(theta,all_rows=True);pred,_=d.chart_predictions(support);ls=c.local_metrics(d,d.manual[plan['strongTest']],scores)
                    else:
                        p,support,_,_,_=k.forward(theta,all_rows=True);pred,_,_=d.chart_predictions_routed(support,p);ls=r.routed_local(d,d.manual[plan['strongTest']],p)
                    models[ci]=dict(model=k.export(theta),receipt=receipt,pred=pred,locals=corrected(ls))
            state=models[ci];pred=state['pred'];output[name]=[dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[x for x,y in zip(d.classes,pred[i])if y])for i in d.main[plan['test']]];local[name]=state['locals']
        c.write(root/str(o)/'outer-models.json',{str(i):dict(model=m['model'],receipt=m['receipt'])for i,m in models.items()});c.write(root/str(o)/'outer-predictions.json',output);c.write(root/str(o)/'outer-local.json',local)
    result={}
    for name in common:
        pred=np.zeros_like(old_pred);ls=[]
        for o in range(6):
            for row in c.read(root/str(o)/'outer-predictions.json')[name]:pred[d.lookup[row['uid']]]=[x in row['predicted']for x in d.classes]
            ls.extend(c.read(root/str(o)/'outer-local.json')[name])
        result[name]=dict(local=c.summary(ls),whole=c.metrics(d.y[d.main],pred[d.main],d.weight[d.main]))
    base=dict(local=c.summary(corrected(c.local_metrics(d,d.manual))),whole=c.metrics(d.y[d.main],old_pred[d.main],d.weight[d.main]))
    c.write(root/'oof.json',dict(scope='Allowed-set metric correction and complete reselection; original results retained. Same source-group folds and objective. All original INNER numerical fits reused.',baseline=base,metrics=result))
    c.write(root/'complete.json',dict(scopes=6,originalInnerFitsReused=len(study.configurations())*18,correction='A prediction inside the allowed label set cannot count as Tech/CS confusion.'))
    print(kind,'baseline',base['local'],flush=True)
    for name,v in result.items():
        if name.endswith('protected'):print(name,v['local'],{k:round(v['whole'][k]*100,2)for k in ['macroPrecision','macroRecall','macroF1']},flush=True)


if __name__=='__main__':
    for kind in ['competition','routing']:run(kind)
