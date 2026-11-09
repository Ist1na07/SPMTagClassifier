"""Matched cadence-input ablation; no new classifier coefficients are fit."""
from pathlib import Path
import itertools
import json
import numpy as np
from pressure_data import mil, read
from whole_reference import write
from rc_action_runtime import sufficient, policy_predictions, macro_scores, POST
from break_ratio_report import metrics

ROOT=Path('research/overlay-quality/grid-input-r1')
GRID=[dict(physicalFloor=b,relativeTolerance=q)for b in [False,True]for q in [None,.1,.15,.2,.25]]
CONFIGS=[dict(gridId=gi,grid=g,partition=pi)for gi,g in enumerate(GRID)for pi in range(9)]
ARMS={'original':[0],'floor':[5],'relative':[1,2,3,4],'both':[6,7,8,9],'joint':list(range(10))}


def choose(rows,mode,reference):
    if mode=='overlay_guarded':rows=[r for r in rows if r['macroF1']>=reference-1e-12]
    elif mode!='macro':raise ValueError('Unknown selection')
    tie=lambda r:(r['config']['gridId']==0,-abs(r['config']['partition']-4),
        not r['config']['grid']['physicalFloor'],r['config']['grid']['relativeTolerance']is None,-r['configId'],-r['policyId'])
    return max(rows,key=(lambda r:(r['localAccuracy'],r['macroF1'],r['macroRecall'],*tie(r)))if mode=='overlay_guarded'else(
        lambda r:(r['macroF1'],r['macroRecall'],r['localAccuracy'],*tie(r))))


def declare():
    p=dict(configs=CONFIGS,post=POST,arms=ARMS,
        scope='Original349whole/6manual, historical6outer x3INNER song/duplicate groups; retrospective development evaluation, not new blind test.',
        method='No new class coefficients. Complete relative tolerance x physical spacing floor x9partitions x245post. Each2x2 arm retunes all its declared active coordinates; other legacy parameters fixed for attribution.',
        selection='Per outer TRAIN, pool3INNER validation identities. Macro control: F1,R,manual. Overlay guard: F1>=best original grid with9partitions/245post, then manual,F1,R. All arms allow original-grid fallback; deterministic ties prefer original grid,partition closest4,no physical floor,no relative tolerance,lower config/policy id.',
        final='Mean six INNER score tables, same criteria before outer metrics. No outer coefficient or outer winning model copied.',
        profiles='Fix grid tolerance/floor/partition and fully reselect all OTHER declared coordinates and245post; retain infeasibility and actual changed tag sets. q=None is an algorithm control, not a numeric well.',
        player='C/H/F queried charts are regressions only, not selection targets. No copied whole labels into windows.',
        budget='One bounded cadence-input block; no residual-head retraining or automatic grid extension.',
        sourceSnapshots={f:Path(__file__).with_name(f).read_text(encoding='utf8')for f in ['grid_input_report.py','rc_action_runtime.py','break_ratio_report.py']})
    file=ROOT/'post-protocol.json'
    if file.exists()and read(file)!=p:raise ValueError('Frozen grid-input consumer changed')
    if not file.exists():write(file,p)


def run():
    declare();receipt=read(ROOT/'complete.json')
    if [{k:c[k]for k in ['gridId','grid','partition']}for c in receipt['configs']]!=CONFIGS:raise ValueError('Grid configuration inventory mismatch')
    main,manual,_=mil.load();uids=[r['uid']for r in main];groups=[r['group']for r in main];mids=[r['uid']for r in manual]
    y=mil.encode(main);weights=np.array([r['recordWeight']for r in main])
    predictions=np.empty((len(CONFIGS),len(POST),len(main),len(mil.MAP_CLASSES)),bool)
    local_details={};local_acc=np.empty((len(CONFIGS),len(manual)))
    for gi in range(len(GRID)):
        with(ROOT/f'grid-{gi}.jsonl').open(encoding='utf8')as file:rows={r['uid']:r for r in map(json.loads,file)}
        for pi in range(9):
            ci=gi*9+pi
            predictions[ci]=policy_predictions([sufficient(rows[u]['partitions'][pi]['frames'])for u in uids])
            local_details[ci]=[rows[u]['partitions'][pi]['local']for u in mids]
            local_acc[ci]=[r['accuracy']for r in local_details[ci]]
    plans=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']
    names=[a+'/'+s for a,s in itertools.product(ARMS,['macro','overlay_guarded'])]
    held={n:np.zeros_like(y)for n in names};held_local={n:[]for n in names};selections=[];tables=[];profiles=[]
    for outer,plan in enumerate(plans):
        ids=[i for s in plan['inner']for i in s['validation']];mi=[i for s in plan['inner']for i in s['strongValidation']]
        if len(ids)!=len(set(ids))or set(ids)!=set(plan['train'])or len(mi)!=len(set(mi))or set(mi)!=set(plan['strongTrain']):raise ValueError('INNER scope changed')
        candidates=[]
        for ci,cfg in enumerate(CONFIGS):
            f,r=macro_scores(y[ids],predictions[ci][:,ids],weights[ids])
            candidates.extend(dict(configId=ci,config=cfg,policyId=pid,policy=POST[pid],macroF1=float(f[pid]),macroRecall=float(r[pid]),
                localAccuracy=float(local_acc[ci,mi].mean()))for pid in range(len(POST)))
        tables.append(candidates);ref=max(r['macroF1']for r in candidates if r['config']['gridId']==0)
        selected={}
        for name in names:
            arm,mode=name.split('/');eligible=[r for r in candidates if r['config']['gridId']in ARMS[arm]+[0]]
            c=choose(eligible,mode,ref);selected[name]=c
            selections.append(dict(outer=outer,name=name,choice=c,usesOuterTruth=False))
            held[name][plan['test']]=predictions[c['configId'],c['policyId'],plan['test']]
            held_local[name].extend(dict(uid=mids[i],**local_details[c['configId']][i])for i in plan['strongTest'])
        write(ROOT/'selection.json',selections)
        target=selected['joint/overlay_guarded'];q=predictions[target['configId'],target['policyId'],ids];curves={}
        for coordinate,values,get in [('relativeTolerance',[None,.1,.15,.2,.25],lambda r:r['config']['grid']['relativeTolerance']),
                ('physicalFloor',[False,True],lambda r:r['config']['grid']['physicalFloor']),('partition',range(9),lambda r:r['config']['partition'])]:
            points=[]
            for value in values:
                rows=[r for r in candidates if get(r)==value];best=choose(rows,'macro',ref)
                feasible=[r for r in rows if r['macroF1']>=ref-1e-12]
                z=predictions[best['configId'],best['policyId'],ids]
                points.append(dict(value=value,macro=best,guarded=choose(feasible,'overlay_guarded',ref)if feasible else None,
                    changedTagSetMass=float(weights[ids]@np.any(q!=z,axis=1)/weights[ids].sum())))
            curves[coordinate]=points
        profiles.append(dict(outer=outer,profiles=curves))
    pooled=[]
    for i,c in enumerate(tables[0]):
        if any((t[i]['configId'],t[i]['policyId'])!=(c['configId'],c['policyId'])for t in tables):raise ValueError('Table identity changed')
        pooled.append({**c,**{k:float(np.mean([t[i][k]for t in tables]))for k in ['macroF1','macroRecall','localAccuracy']}})
    ref=max(r['macroF1']for r in pooled if r['config']['gridId']==0)
    final={n:choose([r for r in pooled if r['config']['gridId']in ARMS[n.split('/')[0]]+[0]],n.split('/')[1],ref)for n in names}
    write(ROOT/'final-inner-selection.json',dict(selections=final,usesOuterTruth=False))
    result={n:metrics(y,v,weights,uids,groups)for n,v in held.items()}
    local_result={n:dict(equalChartAccuracy=float(np.mean([r['accuracy']for r in rs])),byChart={r['uid']:r['accuracy']for r in rs},
        techToDenseMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']=='dense chordstream'and'tech'in d['truth']and'dense chordstream'not in d['truth'])for r in rs),
        falseBreakMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']=='break'and'break'not in d['truth'])for r in rs),
        missedBreakMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']!='break'and d['truth']==['break'])for r in rs))for n,rs in held_local.items()}
    write(ROOT/'analysis.json',dict(metrics=result,local=local_result,choices=selections,profiles=profiles,final=final,uids=uids,
        predictions={n:v.astype(int).tolist()for n,v in held.items()},adopted=False))
    lines=['# BPM网格输入×物理周期下限×分段×聚合','',
        '| 处理／选择 | macro P | macro R | macro F1 | micro F1 | 集合全对 | 局部准确率 | Tech→Dense ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,m in result.items():lines.append('| '+n+' | '+' | '.join(f'{100*v:.2f}%'for v in
        [m['macro']['precision'],m['macro']['recall'],m['macro']['f1'],m['micro']['f1'],m['exactMatch'],local_result[n]['equalChartAccuracy']])+f" | {local_result[n]['techToDenseMs']:.1f} |")
    lines+=['','原专家系数固定；90个完整原生配置各自联调245策略，外层只评估，不选参。所有处理均允许原网格回退。完整逐类误报漏报、局部及INNER共同定型见analysis.json。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    import sys
    declare()if'--declare-only'in sys.argv else run()
