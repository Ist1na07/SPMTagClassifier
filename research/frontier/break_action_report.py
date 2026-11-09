"""Final bounded rest-gate grid: activity rate x relative NPS x original post."""
from pathlib import Path
import numpy as np
from pressure_data import mil, read
from whole_reference import write
from rc_action_runtime import sufficient, policy_predictions, macro_scores, POST
from break_ratio_report import metrics

ROOT=Path('research/overlay-quality/break-regularity-r3')
CONFIGS=[dict(ratio=1.,rowRateMax=0.)]+[dict(ratio=r,rowRateMax=c)for r in [0.,.25,.4,.55,.7,.85,.95]for c in [0.,4.,8.,12.,16.,24.,32.,48.]]


def choose(rows,selection,reference):
    if selection=='original':rows=[r for r in rows if r['configId']==0]
    elif selection=='no_action_rate':rows=[r for r in rows if r['config']['rowRateMax']==0]
    elif selection=='overlay_guarded':rows=[r for r in rows if r['macroF1']>=reference-1e-12]
    elif selection!='macro':raise ValueError('Unknown selection')
    tie=lambda r:(r['configId']==0,r['config']['rowRateMax']==0,r['config']['ratio'],-r['config']['rowRateMax'],-r['policyId'])
    return max(rows,key=(lambda r:(r['localAccuracy'],r['macroF1'],r['macroRecall'],*tie(r)))if selection=='overlay_guarded'else(
        lambda r:(r['macroF1'],r['macroRecall'],r['localAccuracy'],*tie(r))))


def declare():
    p=dict(configs=CONFIGS,post=POST,
        scope='Original349whole/6manual, original6outer x3INNER physical/song groups; repeated historical development evidence, not blind test.',
        fitting='No new trained classifier. Local native gate outputs fixed per configuration. Jointly select both active gate coordinates and all245post from pooled3INNER; F01/F02 never selection targets.',
        selection='overlay_guarded: INNER macroF1>=best original, then manual equal-chart accuracy,F1,recall,prefer original/no-new-action-coordinate,larger relative bound,smaller row cap,lower policyId. macro control F1,recall,manual,same ties. Original and cap0 matched ablations independently retuned.',
        final='Mean6INNER tables; same overlay_guarded and macro criteria. Save before outer metrics, not best outer model.',
        profiles='Fix EACH relative bound/action cap, reselect OTHER gate coordinate AND all245post. Original has no active row-rate cap. All posterior hard-tag metrics only.',
        stop='No further rest-gate grid extension in current product iteration. Positive candidate still requires actual Node/browser parity, local error inspection and integration.',
        sourceSnapshots={f:Path(__file__).with_name(f).read_text(encoding='utf8')for f in ['break_action_report.py','break_ratio_report.py']})
    file=ROOT/'post-protocol.json'
    if file.exists()and read(file)!=p:raise ValueError('Frozen action-grid protocol changed')
    if not file.exists():write(file,p)


def run():
    declare();data=read(ROOT/'complete.json')
    if data['configs']!=CONFIGS:raise ValueError('Configuration inventory mismatch')
    main,manual,_=mil.load();uids=[r['uid']for r in main];groups=[r['group']for r in main]
    rows={r['uid']:r for r in read(ROOT/'frames.json')};local={r['uid']:r for r in read(ROOT/'local.json')}
    y=mil.encode(main);w=np.array([r['recordWeight']for r in main])
    p=np.array([policy_predictions([sufficient(rows[u]['frames'][ci])for u in uids])for ci in range(len(CONFIGS))])
    plans=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']
    local_acc=np.array([[local[r['uid']]['scores'][ci]['accuracy']for r in manual]for ci in range(len(CONFIGS))])
    names=['original','no_action_rate','macro','overlay_guarded'];outputs={s:np.zeros_like(y)for s in names}
    held_local={s:[]for s in names};selections=[];tables=[];profiles=[]
    for outer,plan in enumerate(plans):
        ids=[i for scope in plan['inner']for i in scope['validation']];mids=[i for scope in plan['inner']for i in scope['strongValidation']]
        if len(ids)!=len(set(ids))or set(ids)!=set(plan['train'])or len(mids)!=len(set(mids))or set(mids)!=set(plan['strongTrain']):raise ValueError('INNER scope mismatch')
        candidates=[]
        for ci,cfg in enumerate(CONFIGS):
            f,r=macro_scores(y[ids],p[ci][:,ids],w[ids])
            candidates.extend(dict(configId=ci,config=cfg,policyId=pid,policy=POST[pid],macroF1=float(f[pid]),macroRecall=float(r[pid]),
                localAccuracy=float(local_acc[ci,mids].mean()))for pid in range(len(POST)))
        tables.append(candidates);ref=max(r['macroF1']for r in candidates if r['configId']==0)
        selected={s:choose(candidates,s,ref)for s in names}
        selections.extend(dict(outer=outer,selection=s,choice=c,usesOuterTruth=False)for s,c in selected.items());write(ROOT/'selection.json',selections)
        for s,c in selected.items():
            outputs[s][plan['test']]=p[c['configId'],c['policyId'],plan['test']]
            held_local[s].extend(dict(uid=manual[i]['uid'],config=c['config'],**local[manual[i]['uid']]['scores'][c['configId']])for i in plan['strongTest'])
        target=selected['overlay_guarded'];q=p[target['configId'],target['policyId'],ids];curves={}
        for coordinate in ['ratio','rowRateMax']:
            curve=[]
            for value in dict.fromkeys(c[coordinate]for c in CONFIGS):
                subset=[r for r in candidates if r['configId']!=0 and r['config'][coordinate]==value]
                if not subset:curve.append(dict(value=value,active=False));continue
                best=choose(subset,'macro',ref);feasible=[r for r in subset if r['macroF1']>=ref-1e-12]
                guarded=choose(feasible,'overlay_guarded',ref)if feasible else None
                z=p[best['configId'],best['policyId'],ids]
                curve.append(dict(value=value,active=True,macro=best,guarded=guarded,
                    changedTagSetMass=float(w[ids]@np.any(q!=z,axis=1)/w[ids].sum())))
            curves[coordinate]=curve
        profiles.append(dict(outer=outer,reference=ref,profiles=curves))
    pooled=[]
    for i,c in enumerate(tables[0]):
        if any((t[i]['configId'],t[i]['policyId'])!=(c['configId'],c['policyId'])for t in tables):raise ValueError('Table identity changed')
        pooled.append({**c,**{k:float(np.mean([t[i][k]for t in tables]))for k in ['macroF1','macroRecall','localAccuracy']}})
    ref=max(r['macroF1']for r in pooled if r['configId']==0)
    final={s:choose(pooled,s,ref)for s in names};write(ROOT/'final-inner-selection.json',dict(selections=final,usesOuterTruth=False))
    result={s:metrics(y,v,w,uids,groups)for s,v in outputs.items()}
    local_result={s:dict(equalChartAccuracy=float(np.mean([r['accuracy']for r in rs])),byChart={r['uid']:r['accuracy']for r in rs},
        falseBreakMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']=='break'and'break'not in d['truth'])for r in rs),
        missedBreakMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']!='break'and d['truth']==['break'])for r in rs))for s,rs in held_local.items()}
    write(ROOT/'analysis.json',dict(metrics=result,local=local_result,selections=selections,profiles=profiles,final=final,
        uids=uids,predictions={s:v.astype(int).tolist()for s,v in outputs.items()},adopted=False))
    lines=['# 休息门控：实际行速率×相对物量×聚合','',
        '| 选择 | macro P | macro R | macro F1 | micro F1 | 集合全对 | 局部准确率 |','|---|---:|---:|---:|---:|---:|---:|']
    for s,m in result.items():lines.append('| '+s+' | '+' | '.join(f'{100*v:.2f}%'for v in
        [m['macro']['precision'],m['macro']['recall'],m['macro']['f1'],m['micro']['f1'],m['exactMatch'],local_result[s]['equalChartAccuracy']])+' |')
    lines+=['',str(local_result),'','共同部署选择（仅INNER）：'+str(final),'',
        '完整57非冗余门控配置×245后处理；cap0保留全部相对密度消融，ratio1原规则仅保留一次。未重新拟合类型专家。旧资料上的回顾性结果，不是新盲测。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    import sys
    declare()if'--declare-only'in sys.argv else run()
