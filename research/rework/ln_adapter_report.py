"""Audit complete anchored updates and score definite Release primary labels."""
import argparse,gzip
from pathlib import Path
from collections import Counter,defaultdict
from mil import json,np,load
from analyse import main_run


def read(path):
    path=Path(path)
    if path.suffix=='.gz':
        with gzip.open(path,'rt',encoding='utf8')as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf8'))


def definite_release(manual,rows):
    byid=defaultdict(list)
    for row in rows:byid[row['uid']].append(row)
    charts=[]
    for r in manual:
        positive=negative=tp=fp=0.
        for a in r['annotations']:
            if len(a['tags'])!=1 or a['tags'][0]in ['wildcard','ln wildcard','rc mix','ln mix','hybrid']:continue
            is_release=a['tags']==['release'];duration=a['end']-a['start']
            if is_release:positive+=duration
            else:negative+=duration
            for s in byid[r['uid']]:
                overlap=max(0,min(a['end'],s['end'])-max(a['start'],s['start']))
                if s['tag']=='release':
                    if is_release:tp+=overlap
                    else:fp+=overlap
        charts.append({'chart':r['id'],'positiveMs':positive,'negativeMs':negative,'truePositiveMs':tp,'falsePositiveMs':fp,
          'recall':tp/positive if positive else None,'precision':tp/(tp+fp)if tp+fp else None})
    p=sum(r['positiveMs']for r in charts);tp=sum(r['truePositiveMs']for r in charts);fp=sum(r['falsePositiveMs']for r in charts)
    return {'charts':charts,'positiveCharts':sum(r['positiveMs']>0 for r in charts),'positiveMs':p,'truePositiveMs':tp,'falsePositiveMs':fp,
            'recall':tp/p if p else None,'precision':tp/(tp+fp)if tp+fp else None}


def run(directory):
    root=Path(directory);complete=read(root/'complete.json');protocol=read(root/'protocol.json')
    main_run(directory);inner=Counter();outer=Counter();unaccepted=[];gradients=[];seen=set();chosen={};candidate_counts={}
    records=protocol['records'];main=[r for r in records if not r['uid'].startswith('manual/')];manual=[r for r in records if r['uid'].startswith('manual/')]
    for variant in complete['variants']:
        chosen[variant]=[]
        for o,plan in enumerate(protocol['plans']):
            directory=root/variant/str(o);choice=read(directory/'selection.json');chosen[variant].append(choice['head'])
            with np.load(directory/'inner-predictions.npz')as saved:
                policy_count=len(protocol['postPolicies']);fit_count=int(np.prod([len(v)for v in protocol['fitGrid'].values()]))
                expected=policy_count if variant=='no_update'else(fit_count+1)*policy_count
                if saved['map'].shape!=(expected,len(plan['train']),17)or saved['segmentAccuracy'].shape!=(expected,len(plan['strongTrain'])):raise RuntimeError('Joint candidate prediction matrix shape differs')
                candidate_counts[variant]=expected
            for row in read(directory/'inner-fits.json.gz'):
                trace=row['fit'];identity=(variant,o,row['inner'],row['configId'])
                if identity in seen:raise RuntimeError('Repeated learned fit trace')
                seen.add(identity);inner[trace['status']]+=1;gradients.append(trace['checks'][-1]['gradientMax'])
                split=plan['inner'][row['inner']];allowed={main[i]['uid']for i in split['train']}|{manual[i]['uid']for i in split['strongTrain']}
                if not set(trace['trainUids'])<=allowed:raise RuntimeError('New LN parameters/scaler saw held-out record')
                if trace['status']!='small_gradient'or trace['checks'][-1]['gradientMax']>1e-5:unaccepted.append({'variant':variant,'outer':o,**row})
            saved=read(directory/'outer-model.json');trace=saved['fit'];outer[trace['status']]+=1
            allowed={main[i]['uid']for i in plan['train']}|{manual[i]['uid']for i in plan['strongTrain']}
            if set(saved['trainUids'])!=allowed:raise RuntimeError('Outer saved scope differs')
            if trace['status']!='frozen_archived_prior':
                if not set(trace['trainUids'])<=allowed:raise RuntimeError('Outer new parameters saw held-out record')
                gradients.append(trace['checks'][-1]['gradientMax'])
                if trace['status']!='small_gradient'or trace['checks'][-1]['gradientMax']>1e-5:unaccepted.append({'variant':variant,'outer':o,'fit':trace})
    if len(seen)!=complete['learnedInnerFits']:raise RuntimeError('Learned fit count differs')
    audit={'uniqueLearnedInnerFits':len(seen),'innerStatuses':dict(inner),'outerStatuses':dict(outer),'maxGradient':max(gradients),
      'unaccepted':unaccepted,'accepted':not unaccepted,'selectedHeadsByVariant':chosen,'jointCandidatesPerOuter':candidate_counts,
      'meaning':'frozen_archived_prior is an exact zero-update head, not a newly optimized fit; learned scalers and coefficients are checked against actual train scopes'}
    (root/'optimizer-audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    _,manual_records,_=load();rows=read(root/'outer-segments.json')
    release={v:definite_release(manual_records,[r for r in rows if r['variant']==v])for v in complete['variants']}
    (root/'definite-release.json').write_text(json.dumps({'scope':'Release PRIMARY label on definite singleton annotations only; ambiguous/mix/wildcard time excluded, not generic Release-presence truth','metrics':release},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['','## 实际拟合、选择与 Release 主标签检查','',
      f"{len(seen):,} 项不同内层拟合，状态 `{dict(inner)}`；外层选择状态 `{dict(outer)}`。最大实际训练梯度 {max(gradients):.3g}，阈值1e-5；训练作用域核对包含新标准化与校正系数。零头单独标为冻结先验，不算重新拟合。",'',
      '| 方案 | 六折所选头（0为不更新） | Release召回 | Release精确率 | 真阳性秒 | 假阳性秒 |','|---|---|---:|---:|---:|---:|']
    pct=lambda x:'—'if x is None else f'{x:.2%}'
    for v,r in release.items():lines.append(f"| {v} | {chosen[v]} | {pct(r['recall'])} | {pct(r['precision'])} | {r['truePositiveMs']/1000:.2f} | {r['falsePositiveMs']/1000:.2f} |")
    lines+=['','这里仅评价确定的单主标签时间；只要人工区间含多个备选、wildcard或Mix就不把它强行当作二分类真值。Release阳性只有两张谱，逐谱时长、误报和召回见 `definite-release.json`；不以大量时间采样点冒充独立来源。',
            '', '弱整谱单调校准器仅用于学习局部校正，不作为新的自由整谱输出头。实际输出仍经同一个原专家/规则路由、局部分布和后处理聚合。']
    if unaccepted:lines+=['','**存在未接受拟合，本轮不能定型。**']
    with(root/'REPORT.md').open('a',encoding='utf8')as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in audit.items()if k not in ['unaccepted','meaning']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='research/rework/results/ln-adapter-r1');run(p.parse_args().directory)
