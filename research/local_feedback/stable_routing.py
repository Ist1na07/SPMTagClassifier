"""Prefer a real local gain in a stable INNER parameter well, not a tiny peak.

Retrospective selection refinement after inspecting the completed study. No
new grid and no OUTER-guided choice among parameter values: the same declared
rule is applied within each INNER scope and to the pooled INNER common model.
"""
from pathlib import Path
import sys,importlib.util
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from routing import Data,Kernel,fit,routed_local,read,write,metrics,local_metrics,summary,PLANS
spec=importlib.util.spec_from_file_location('stable_parent_study',Path(__file__).with_name('routing_study.py'));study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
UP=Path('research/local_feedback/results/routing-r3-allowed-set');ROOT=Path('research/local_feedback/results/routing-stable-r4')


def choose(rows,b):
    eligible=[r for r in rows if r['local']['accuracy']>b['local']['accuracy']+1e-12 and r['local']['techCsConfusionMs']<=b['local']['techCsConfusionMs']+1e-8
        and all(r['whole'][k]>=b['whole'][k]-.01 for k in ['macroPrecision','macroRecall','macroF1','microF1'])]
    peak=max([study.score(b)]+[study.score(r)for r in eligible]);near=[r for r in eligible if study.score(r)>=peak-.002]
    if not near:return b
    return min(near,key=lambda r:({'absolute':0,'context':1,'pressure':2}[r['config']['features']],-r['config']['ridge'],r['config']['teacherWeight'],-study.score(r),r['configId']))


def run():
    d=Data();sels=[read(UP/str(o)/'selection.json')for o in range(6)];grids=[read(UP/str(o)/'inner-grid.json')for o in range(6)];pooled=read('research/local_feedback/results/routing-r3/common-selection.json')['innerMeans'];old_common=read(UP/'common-selection.json');common=choose(pooled,old_common['baseline']);folds=[choose(g,s['baseline'])for g,s in zip(grids,sels)]
    write(ROOT/'selection.json',dict(usesOuterTruth=False,retrospective=True,nearScoreTolerance=.002,
        rule='Positive INNER local gain; no Tech/CS confusion regression; each whole macroP/R/F1/micro within1pp. Within.002 of best scalar objective, prefer fewer physical input groups, stronger ridge, no teacher pseudo-label assumption, then actual score. Return explicit frozen if no eligible improvement.',
        reason='The previous pooled peak used ridge.003+teacher1. A simpler ridge.3/no-teacher solution is only.000239 in score behind, with a larger local INNER gain. Extra parameter freedom is not an acceptance benefit.',
        common=common,folds=folds,source=Path(__file__).read_text(encoding='utf8')))
    pred=np.zeros_like(d.baseline);ls=[];outputs=[]
    for o,(plan,row)in enumerate(zip(PLANS,folds)):
        ci=row['configId'];folder=ROOT/str(o);folder.mkdir(parents=True,exist_ok=True)
        if ci<0:
            actual,_=d.chart_predictions(d.support);local=local_metrics(d,d.manual[plan['strongTest']]);saved=dict(model=None,receipt=None)
        else:
            k=Kernel(d,d.main[plan['train']],d.manual[plan['strongTrain']],row['config']);old=read(UP/str(o)/'outer-models.json')
            if str(ci)in old:
                receipt=old[str(ci)]['receipt'];theta=np.array(min(receipt['starts'],key=lambda x:x['loss'])['theta'])
            else:theta,receipt=fit(k)
            p,support,_,_,_=k.forward(theta,all_rows=True);actual,_,_=d.chart_predictions_routed(support,p);local=routed_local(d,d.manual[plan['strongTest']],p);saved=dict(model=k.export(theta),receipt=receipt)
            np.savez_compressed(folder/'probabilities.npz',route=p)
        ids=d.main[plan['test']];pred[ids]=actual[ids];ls.extend(local);write(folder/'model.json',saved);write(folder/'local.json',local)
        outputs.extend([dict(uid=d.records[i]['uid'],actual=d.records[i]['tags'],predicted=[c for c,v in zip(d.classes,actual[i])if v])for i in ids])
    baseline_pred,_=d.chart_predictions(d.support);result=dict(scope='Retrospective stability-prior selection refinement, source-group cross-fitted new routing parameters. Not a new blind validation.',
        baseline=dict(local=summary(local_metrics(d,d.manual)),whole=metrics(d.y[d.main],baseline_pred[d.main],d.weight[d.main])),
        candidate=dict(local=summary(ls),whole=metrics(d.y[d.main],pred[d.main],d.weight[d.main])),localDetails=ls,predictions=outputs)
    write(ROOT/'oof.json',result);print('stable routing',result['baseline']['local'],result['candidate']['local'],{k:round(result['candidate']['whole'][k]*100,2)for k in ['macroPrecision','macroRecall','macroF1','microF1','exactMatch']},flush=True)
    if common['configId']>=0:
        k=Kernel(d,d.main,d.manual,common['config']);theta,receipt=fit(k);write(ROOT/'candidate.json',dict(model=k.export(theta),receipt=receipt,selection=common,status='Requires actual consumer and human-regression acceptance'))
    write(ROOT/'complete.json',dict(scopes=6,newInnerFits=0,gridConfigsReused=360*18))


if __name__=='__main__':run()
