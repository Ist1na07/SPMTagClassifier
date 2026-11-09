"""Re-use the finished gate grid, fit its product decision to source-TRAIN queries.

The original six-chart metric is retained unchanged. Query TRAIN compliance
is NOT renamed validation accuracy; source-held-out query results are separate.
"""
from pathlib import Path
import json
import numpy as np
from pressure_data import mil, read
from whole_reference import write
from rc_action_runtime import sufficient, policy_predictions, macro_scores, POST
from break_ratio_report import metrics

INPUT=Path('research/overlay-quality/break-regularity-r3')
ROOT=Path('research/overlay-quality/overlay-impression-r1')
FIXED_POST_ONLY=False


def choose(rows,mode,shipped,retuned):
    if mode.endswith('_fixed_post'):
        original_post=dict(dominantShare=.52,mixShare=.07,secondRel=.75)
        rows=[r for r in rows if r['policy']==original_post]
        mode=mode.removesuffix('_fixed_post')
    if mode=='original':rows=[r for r in rows if r['configId']==0]
    elif mode=='strict':rows=[r for r in rows if r['macroF1']>=retuned-1e-12]
    elif mode in ['experience','no_action_rate']:
        rows=[r for r in rows if r['macroF1']>=shipped-1e-12 and(mode!='no_action_rate'or r['config']['rowRateMax']==0)]
    elif mode!='macro':raise ValueError('Unknown product selection')
    tie=lambda r:(r['configId']==0,r['config']['rowRateMax']==0,r['config']['ratio'],-r['config']['rowRateMax'],-r['policyId'])
    if mode in ['experience','no_action_rate','strict']:
        key=lambda r:(r['queryTrainPrimaryAccuracy'],r['localAccuracy'],r['macroF1'],r['macroRecall'],*tie(r))
    else:key=lambda r:(r['macroF1'],r['macroRecall'],r['localAccuracy'],*tie(r))
    return max(rows,key=key)


def declare():
    value=dict(
        purpose='Player-visible Speed->Break bug. Re-use EXACT completed57gate x245post grid; no new gate coordinates, no new classifier fits or grid expansion.',
        addedSupervision='F01/F02 target9-16.5s primarySpeed, original r5 raw answers preserved. F02 Tech component remains possible, not a Tech-negative. Labels belong to EXCERPTS, never copied to full source charts.',
        scope='Only queries whose inherited physical/song group is in the corresponding INNER TRAIN contribute TRAIN compliance. Queries in validation/test never contribute that scope. Source-held-out query primary accuracy is evaluated only after original OUTER selection.',
        productDecision='Experience: first require INNER macroF1>=actual shipped fixed classifier in same scope, then maximize source-TRAIN query primary fit, original manual INNER accuracy,F1,recall,prefer original/no extra action rate. Strict retuned-best guard and macro-first retained as controls. Original arm independently retunes all245post. cap0 independently retunes all declared ratios/post.',
        rationale='The user prioritized conspicuous overlay errors over small boundary gains. Best-retuned macroF1 is a comparison, not the sole product objective. Original17-class and6-chart metrics keep EXACT meanings; no metrics/labels rewritten.',
        final='Use mean6INNER tables with same criterion, never an OUTER winning configuration. QueryTrainPrimaryAccuracy is TRAIN fit, not validation performance.',
        limitations='Two actively selected real Speed excerpts, not random population accuracy; historical teacher exposure unknown and data repeatedly studied. All per-class FP/FN and old manual results remain visible.',
        originalPostControl=FIXED_POST_ONLY,
        aggregationDomain='Only original caller aggregation policy; retune all57gate configurations. Full245post comparison remains in overlay-impression-r1.'if FIXED_POST_ONLY else 'Complete245post jointly with all57gate configurations.',
        sourceSnapshots={f:Path(__file__).with_name(f).read_text(encoding='utf8')for f in ['overlay_impression_selection.py','rc_action_runtime.py','break_ratio_report.py']},
        annotation=read('research/frontier/annotations-r5.json'))
    file=ROOT/'protocol.json'
    if file.exists()and read(file)!=value:raise ValueError('Frozen impression protocol changed')
    if not file.exists():write(file,value)


def query_accuracy(frames,start,end,primary):
    correct=sum(max(0,min(end,f['end'])-max(start,f['start']))for f in frames if f['tag']==primary)
    return correct/(end-start)


def run():
    declare();configs=read(INPUT/'complete.json')['configs']
    main,manual,_=mil.load();uids=[r['uid']for r in main];groups=[r['group']for r in main];mids=[r['uid']for r in manual]
    main_by_uid={r['uid']:r for r in main}
    rows={r['uid']:r for r in read(INPUT/'frames.json')};local={r['uid']:r for r in read(INPUT/'local.json')}
    player=read(INPUT/'player.json');annotation=read('research/frontier/annotations-r5.json')
    queries=[]
    for item in player:
        a=annotation['cases'][item['uid']];source=main_by_uid[a['sourceUid']]
        if a.get('primaryCategory',a['localCategory'])!='speed':raise ValueError('Query primary meaning changed')
        queries.append(dict(uid=item['uid'],group=source['group'],sourceUid=source['uid'],targetMs=item['targetMs'],primary='speed',frames=item['frames']))
    query_scores=np.array([[query_accuracy(q['frames'][ci],*q['targetMs'],q['primary'])for q in queries]for ci in range(len(configs))])
    p=np.array([policy_predictions([sufficient(rows[u]['frames'][ci])for u in uids])for ci in range(len(configs))])
    fixed=next(i for i,r in enumerate(POST)if r==dict(dominantShare=.52,mixShare=.07,secondRel=.75))
    y=mil.encode(main);w=np.array([r['recordWeight']for r in main]);local_acc=np.array([[local[u]['scores'][ci]['accuracy']for u in mids]for ci in range(len(configs))])
    plans=read('research/rework/results/mil-stage1-r4/protocol.json')['plans'];names=['original_fixed_post','experience_fixed_post']if FIXED_POST_ONLY else ['original','macro','strict','no_action_rate','experience']
    out={n:np.zeros_like(y)for n in names};old_local={n:[]for n in names};query_test={n:[]for n in names}
    tables=[];choices=[];scope_inventory=[];refs=[]
    for outer,plan in enumerate(plans):
        ids=[i for s in plan['inner']for i in s['validation']];mi=[i for s in plan['inner']for i in s['strongValidation']]
        if len(ids)!=len(set(ids))or set(ids)!=set(plan['train'])or len(mi)!=len(set(mi))or set(mi)!=set(plan['strongTrain']):raise ValueError('INNER scope changed')
        # TRAIN-only local supervision. Do not score a withheld query here.
        training_occurrences=[]
        for inner,scope in enumerate(plan['inner']):
            train_groups={main[i]['group']for i in scope['train']};val_groups={main[i]['group']for i in scope['validation']}
            if train_groups&val_groups:raise ValueError('Song group leakage')
            eligible=[j for j,q in enumerate(queries)if q['group']in train_groups]
            withheld=[j for j,q in enumerate(queries)if q['group']in val_groups]
            if set(eligible)&set(withheld):raise ValueError('Query supervision leakage')
            training_occurrences.extend(eligible)
            scope_inventory.append(dict(outer=outer,inner=inner,trainQueryUids=[queries[j]['uid']for j in eligible],withheldQueryUids=[queries[j]['uid']for j in withheld],usesWithheldQueryLabels=False))
        train_fit=query_scores[:,training_occurrences].mean(1)if training_occurrences else np.zeros(len(configs))
        candidates=[]
        for ci,cfg in enumerate(configs):
            f,r=macro_scores(y[ids],p[ci][:,ids],w[ids])
            candidates.extend(dict(configId=ci,config=cfg,policyId=pid,policy=POST[pid],macroF1=float(f[pid]),macroRecall=float(r[pid]),
                localAccuracy=float(local_acc[ci,mi].mean()),queryTrainPrimaryAccuracy=float(train_fit[ci]))for pid in range(len(POST)))
        shipped=float(macro_scores(y[ids],p[0,fixed][None,ids],w[ids])[0][0]);retuned=max(r['macroF1']for r in candidates if r['configId']==0)
        refs.append(dict(shipped=shipped,retuned=retuned));tables.append(candidates)
        for n in names:
            c=choose(candidates,n,shipped,retuned);choices.append(dict(name=n,outer=outer,choice=c,usesOuterTruth=False))
            out[n][plan['test']]=p[c['configId'],c['policyId'],plan['test']]
            old_local[n].extend(dict(uid=mids[i],**local[mids[i]]['scores'][c['configId']])for i in plan['strongTest'])
            test_groups={main[i]['group']for i in plan['test']}
            for j,q in enumerate(queries):
                if q['group']in test_groups:query_test[n].append(dict(uid=q['uid'],outer=outer,group=q['group'],sourceUid=q['sourceUid'],
                    primaryAccuracy=float(query_scores[c['configId'],j]),frames=q['frames'][c['configId']],choice=c))
        write(ROOT/'selection.json',choices);write(ROOT/'query-training-scopes.json',scope_inventory)
    pooled=[{**c,**{k:float(np.mean([t[i][k]for t in tables]))for k in ['macroF1','macroRecall','localAccuracy','queryTrainPrimaryAccuracy']}}for i,c in enumerate(tables[0])]
    final={n:choose(pooled,n,float(np.mean([r['shipped']for r in refs])),float(np.mean([r['retuned']for r in refs])))for n in names}
    write(ROOT/'final-inner-selection.json',dict(selections=final,usesOuterTruth=False))
    result={n:metrics(y,v,w,uids,groups)for n,v in out.items()}
    result['shipped_fixed']=metrics(y,p[0,fixed],w,uids,groups)
    local_result={n:dict(originalSixChartAccuracy=float(np.mean([r['accuracy']for r in rs])),byChart={r['uid']:r['accuracy']for r in rs},
        falseBreakMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']=='break'and'break'not in d['truth'])for r in rs),
        missedBreakMs=sum(sum(d['end']-d['start']for d in r['details']if d['predicted']!='break'and d['truth']==['break'])for r in rs))for n,rs in old_local.items()}
    for n,rs in query_test.items():
        if len(rs)!=len(queries)or len({r['uid']for r in rs})!=len(queries):raise ValueError('Incomplete or duplicate source-held-out query inventory')
    write(ROOT/'analysis.json',dict(metrics=result,originalManual=local_result,sourceHeldOutQueries=query_test,choices=choices,final=final,
        queryScores=query_scores.tolist(),queryUids=[q['uid']for q in queries],uids=uids,predictions={n:v.astype(int).tolist()for n,v in out.items()},adopted=False))
    lines=['# Overlay明显误判：原参数网格，新增源TRAIN玩家主类监督','',
        '| 策略 | macro P | macro R | macro F1 | micro F1 | 整谱全对 | 原六谱局部 | 源留出两条Speed摘录 |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for n,m in result.items():
        values=[m['macro']['precision'],m['macro']['recall'],m['macro']['f1'],m['micro']['f1'],m['exactMatch']]
        suffix='— | —'if n=='shipped_fixed'else f"{100*local_result[n]['originalSixChartAccuracy']:.2f}% | {100*np.mean([r['primaryAccuracy']for r in query_test[n]]):.2f}%"
        lines.append('| '+n+' | '+' | '.join(f'{100*v:.2f}%'for v in values)+' | '+suffix+' |')
    lines+=['','共同部署选择（INNER＋对应TRAIN监督）：'+str(final),'',
        '原17类／六谱指标含义不变。两条玩家摘录是主动选择的回归，不是随机准确率。TRAIN主类符合度不是验证成绩；所有源留出结果按原外折组隔离，摘录标签未复制到整张来源谱。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed-post-only',action='store_true')
    parser.add_argument('--declare-only',action='store_true')
    parser.add_argument('--output',type=Path,help='Fresh output root; completed frozen protocols are never overwritten')
    args=parser.parse_args()
    if args.fixed_post_only:
        FIXED_POST_ONLY=True;ROOT=Path('research/overlay-quality/overlay-impression-fixed-post-r1')
    if args.output is not None:ROOT=args.output
    declare()if args.declare_only else run()
