"""Bounded direct-core local selection, before any whole-label feedback.

Same six x three source splits; full original b/W retune versus bias-only,
original natural parents versus the viable 4s physical-context mechanism.
1/2s contexts failed the preceding expressivity/routing check and are retained
as diagnostics, not expanded into a new expensive family of fits.
"""
from core import *

ROOT=Path('research/core_retune/results/local-source-r1')
PLANS=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']


def configurations():
    return [dict(input=inp,mode=mode,ridge=r)for inp in ['parent','context4000']for mode in ['bias','weights']for r in [.1,.01,.001]]+[dict(input='context4000',mode='unchanged',ridge=None)]


def subset_metric(d,frames,manual,real):
    local=d.local(frames,manual);all_real=d.real(frames)['details'];rows=[r for r in all_real if r['id']in real]
    return dict(manual=local,real=dict(accuracy=float(np.mean([r['accuracy']for r in rows]))if rows else None,details=rows))


def mean_metrics(rows):
    manual=[x for r in rows for x in r['manual']['details']];real=[x for r in rows for x in r['real']['details']]
    a=float(np.mean([r['accuracy']for r in manual]));b=float(np.mean([r['accuracy']for r in real]))if real else None
    return dict(accuracy=a,realAccuracy=b,techToDenseMs=sum(r['techToDenseMs']for r in manual),techCsConfusionMs=sum(r['techCsConfusionMs']for r in manual),objective=.75*a+.25*b if b is not None else a)


def fit_frames(d,c,uids):
    if c['mode']=='unchanged':return d.baseline,{},{}
    changes={};models={};receipts={}
    for family in ['rc','ln']:
        k=Kernel(d,family,uids,mode=c['mode'],ridge=c['ridge']);theta,receipts[family]=fit(k);changes[family]=k.changes(theta);models[family]=k.export(theta)
    return d.frames(changes),models,receipts


def run():
    manifest=read('.cache/local-feedback-activity-r6b/manifest.json');records={r['uid']:r for r in manifest['records']};main=manifest['mainUids'];manual=manifest['manualUids'];datasets={'parent':Data(),'context4000':Data('.cache/core-retune-local-context-4000.jsonl.gz')};d=datasets['parent'];configs=configurations()
    protocol=dict(id='direct-core-local-source-r1',researchBaseline='v1.1.0',productControl='v1.2.4',configs=configs,plans=PLANS,
      selection='Pooled INNER .75 original-six per-chart accuracy + .25 real-eight per-case accuracy when present; then less Tech/CS confusion, stronger regularization, lower ID. Baseline included, no all-metric guard. Save all INNER selections before OUTER. Whole labels NOT used in this local stage. Not blind; historical study exposure acknowledged.',
      exactPath='Original modelScores softmax W/b, unchanged fixed rule wildcard/vibro blending, original gates, wildcard demotion, secondary and smoothing. Frozen parent routing versus recomputed context routing. No old-affinity-sum constraint. Real labels supervise duration-weighted interval bags; source groups govern inclusion.',
      source={p:Path(__file__).with_name(p).read_text(encoding='utf8')for p in ['core.py','local_study.py']})
    path=ROOT/'protocol.json'
    if path.exists()and read(path)!=protocol:raise ValueError('Frozen protocol changed')
    write(path,protocol)

    def scope(split,train=True):
        groups={records[main[i]]['group']for i in split['train'if train else'validation']};strong=split['strongTrain'if train else'strongValidation'];ms=[manual[i]for i in strong];extras=[e for e in d.extra if records[e['uid']]['group']in groups];uids=set(ms)|{e['uid']for e in extras}
        return uids,ms,[e['id']for e in extras]

    # Finish and freeze INNER for every scope before consuming OUTER.
    for outer,plan in enumerate(PLANS):
        folder=ROOT/str(outer);rows=[]
        base_details=[]
        for split in plan['inner']:
            tr,_,_=scope(split);va,ms,es=scope(split,False)
            if tr&va:raise ValueError('Local source leakage')
            tg={records[u]['group']for u in tr};vg={records[u]['group']for u in va}
            if tg&vg:raise ValueError('Source group leakage')
            base_details.append(subset_metric(d,d.baseline,ms,es))
        rows.append(dict(id=-1,config=None,inner=base_details,metrics=mean_metrics(base_details)))
        for ci,c in enumerate(configs):
            path=folder/('config-'+str(ci)+'.json')
            if path.exists():row=read(path)
            else:
                details=[];receipt=[]
                for inner,split in enumerate(plan['inner']):
                    tr,_,_=scope(split);va,ms,es=scope(split,False);frames,models,fit_receipt=fit_frames(datasets[c['input']],c,tr);details.append(subset_metric(datasets[c['input']],frames,ms,es));receipt.append(dict(inner=inner,trainUids=sorted(tr),validationUids=sorted(va),fits=fit_receipt))
                row=dict(id=ci,config=c,inner=details,metrics=mean_metrics(details),receipts=receipt);write(path,row)
            rows.append(row);print('core INNER',outer,ci,row['metrics'],flush=True)
        def key(r):return(r['metrics']['objective'],-r['metrics']['techCsConfusionMs'],r['config']['ridge']if r['config']and r['config']['ridge']is not None else 1.,-r['id'])
        choices={}
        for arm in ['all','parent','context4000','bias','weights','unchanged']:
            eligible=[r for r in rows if r['id']==-1 or arm=='all'or r['config']['input']==arm or r['config']['mode']==arm];winner=max(eligible,key=key);choices[arm]=dict(id=winner['id'],config=winner['config'],metrics=winner['metrics'])
        write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,choices=choices))
    write(ROOT/'inner-complete.json',dict(configurations=len(configs),fitConfigs=12,innerScopes=18,familyFits=12*18*2,starts=12*18*2*2))
    outcomes={k:[]for k in ['all','parent','context4000','bias','weights','unchanged']};baseline=[]
    for outer,plan in enumerate(PLANS):
        choices=read(ROOT/str(outer)/'selection.json')['choices'];split=dict(train=plan['train'],validation=plan['test'],strongTrain=plan['strongTrain'],strongValidation=plan['strongTest']);tr,_,_=scope(split);_,ms,es=scope(split,False);baseline.append(subset_metric(d,d.baseline,ms,es));states={-1:(d.baseline,{},{} )};saved={}
        for arm,choice in choices.items():
            ci=choice['id'];c=choice['config'];dd=d if ci==-1 else datasets[c['input']]
            if ci not in states:states[ci]=fit_frames(dd,c,tr)
            frames,models,receipt=states[ci];outcomes[arm].append(subset_metric(dd,frames,ms,es));saved[arm]=dict(choice=choice,metrics=outcomes[arm][-1],models=models,receipts=receipt,trainUids=sorted(tr))
        write(ROOT/str(outer)/'outer.json',saved)
    result=dict(scope='Nested source-group selection of directly retuned original expert; local stage only, not whole-product acceptance. Historical data, not blind.',baseline=dict(metrics=mean_metrics(baseline),details=baseline),arms={k:dict(metrics=mean_metrics(v),details=v)for k,v in outcomes.items()});write(ROOT/'oof.json',result);print('CORE LOCAL COMPLETE',{k:v['metrics']for k,v in result['arms'].items()},flush=True)


if __name__=='__main__':run()
