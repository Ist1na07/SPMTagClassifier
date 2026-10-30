"""Train-only convergence stress test; never reads validation or outer labels.

Some fits hit the LBFGS iteration budget. Select cases by training residual
gradient only and measure whether a longer same-objective polish materially
changes the fit. This does not select a model or change the running search.
"""
from mil import *
def main_run():
    source=Path('.cache/rework-mil-stage1-r2-fits.jsonl');by=defaultdict(list)
    with source.open(encoding='utf8')as f:
        for line in f:
            r=json.loads(line);by[r['variant']].append(r)
    protocol=json.loads(Path('research/rework/results/mil-stage1-r2/protocol.json').read_text(encoding='utf8'))
    main,manual,_=load();cases=[]
    for v in ['original56','original56_context','modern40_context','corrected56_context','union_context']:
        if not by[v]:continue
        ordered=sorted(by[v],key=lambda r:r['fit']['gradientMax'])
        cases.extend([(ordered[-1],'worst_train_gradient'),(ordered[len(ordered)//2],'median_train_gradient')])
    results=[]
    for row,kind in cases:
        split=protocol['plans'][row['outer']]['inner'][row['inner']];cfg=row['config'];v=row['variant']
        train=[main[i]for i in split['train']]+[manual[i]for i in split['strongTrain']]
        model,data,trace=fit(train,v,cfg)
        with torch.no_grad():p0,z0,_=model(data,cfg);base=float(model.objective(data,cfg))
        optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=500,tolerance_grad=1e-7,tolerance_change=1e-12,line_search_fn='strong_wolfe');calls=0
        def closure():
            nonlocal calls
            calls+=1;optimizer.zero_grad();loss=model.objective(data,cfg);loss.backward();return loss
        optimizer.step(closure)
        with torch.no_grad():p1,z1,_=model(data,cfg);loss=float(model.objective(data,cfg))
        r={'fitKey':row['key'],'case':kind,'variant':v,'original':trace,'polishEvaluations':calls,
          'trainingObjectiveBefore':base,'trainingObjectiveAfter':loss,'objectiveDecrease':base-loss,
          'trainingSegmentPrimaryFlipFraction':float((p0.argmax(1)!=p1.argmax(1)).double().mean()),
          'trainingSegmentProbabilityMeanAbsoluteChange':float((p1-p0).abs().mean()),
          'trainingMapProbabilityMeanAbsoluteChange':float((torch.sigmoid(z1)-torch.sigmoid(z0)).abs().mean())}
        results.append(r);print(v,kind,r['objectiveDecrease'],r['trainingSegmentPrimaryFlipFraction'],flush=True)
        dump('research/rework/results/optimizer-audit.json',{'scope':'training-only; no validation/outer labels or selection scores consulted','cases':results})
if __name__=='__main__':main_run()
