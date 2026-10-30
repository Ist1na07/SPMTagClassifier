"""Score complete outer predictions, preserving actual outputs and fold choices.

Refuses partial runs. Full 17-label F1 stays primary; group-balanced and supported
class views are supplementary, not replacements selected after seeing results.
"""
import argparse
from mil import *
from sklearn.metrics import classification_report
def hard_measures(y,p,w):
    return {'macroF1':float(f1_score(y,p,average='macro',sample_weight=w,zero_division=0)),
      'microF1':float(f1_score(y,p,average='micro',sample_weight=w,zero_division=0)),
      'sampleF1':float(f1_score(y,p,average='samples',sample_weight=w,zero_division=0)),
      'exactMatch':float(accuracy_score(y,p,sample_weight=w))}
def f1_counts(tp,fp,fn):
    return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0).mean(-1)
def encode_prediction(rows):
    return np.array([[int(c in r['predicted'])for c in MAP_CLASSES]for r in rows])
def score_segments(manual,rows):
    byid=defaultdict(list)
    for r in rows:byid[r['uid']].append(r)
    stats=[];calibration=[];confusions=defaultdict(float)
    for r in manual:
        pred=sorted(byid[r['uid']],key=lambda s:s['start']);total=correct=nll=0.;interior_total=interior_good=0.
        bgood=btotal=ntotal=ngood=0.
        labelled=sum(a['end']-a['start']for a in r['annotations']if len(a['tags'])==1)
        have_probability=any('probability'in s for s in pred)
        for a in r['annotations']:
            if not a['tags']:continue
            total+=a['end']-a['start'];covered=0
            if a['tags']==['break']:btotal+=a['end']-a['start']
            else:ntotal+=a['end']-a['start']
            if a['end']-a['start']>2000:interior_total+=a['end']-a['start']-2000
            for s in pred:
                d=max(0,min(a['end'],s['end'])-max(a['start'],s['start']));covered+=d
                if not d:continue
                ok=s['tag'] in a['tags'];correct+=d*ok
                if a['tags']==['break']:bgood+=d*ok
                else:ngood+=d*ok
                if a['end']-a['start']>2000:
                    di=max(0,min(a['end']-1000,s['end'])-max(a['start']+1000,s['start']));interior_good+=di*ok
                if not ok:confusions['|'.join(a['tags'])+' -> '+s['tag']]+=d
                if 'probability' in s:
                    p=np.array(s['probability']);allowed=[CLASSES.index(t)for t in a['tags']]
                    nll-=d*np.log(max(p[allowed].sum(),1e-15))
                    # Calibration only where annotation names ONE class; a
                    # set-valued acceptable label cannot define singleton truth.
                    if len(a['tags'])==1:
                        k=CLASSES.index(a['tags'][0]);truth=np.zeros(len(CLASSES));truth[k]=1
                        calibration.append({'confidence':float(p.max()),'correct':ok,'weight':d/labelled/len(manual),
                          'brier':float(((p-truth)**2).sum())})
            if have_probability:nll-=(a['end']-a['start']-covered)*np.log(1e-15)
        stats.append({'chart':r['id'],'labelledMs':total,'correctMs':correct,'accuracy':correct/total,
          'interiorAccuracy':interior_good/max(1,interior_total),'partialLabelNll':nll/total if have_probability else None,
          'breakRecall':bgood/max(1,btotal),'nonBreakAccuracy':ngood/max(1,ntotal)})
    bins=[]
    for i in range(10):
        values=[r for r in calibration if min(9,int(r['confidence']*10))==i]
        w=sum(r['weight']for r in values)
        if not w:continue
        bins.append({'lower':i/10,'upper':(i+1)/10,'weight':w,
          'confidence':sum(r['weight']*r['confidence']for r in values)/w,
          'accuracy':sum(r['weight']*r['correct']for r in values)/w})
    mass=sum(r['weight']for r in calibration)
    return {'charts':stats,'chartMeanAccuracy':float(np.mean([r['accuracy']for r in stats])),
      'durationAccuracy':sum(r['correctMs']for r in stats)/sum(r['labelledMs']for r in stats),
      'meanInteriorAccuracy':float(np.mean([r['interiorAccuracy']for r in stats])),
      'singletonCalibration':{'bins':bins,'ece':sum(r['weight']*abs(r['confidence']-r['accuracy'])for r in bins)/mass if mass else None,
       'brier':sum(r['weight']*r['brier']for r in calibration)/mass if mass else None},
      'largestErrors':sorted(confusions.items(),key=lambda kv:-kv[1])[:20]}
def main_run(directory):
    root=Path(directory);complete=json.loads((root/'complete.json').read_text(encoding='utf8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    main,manual,_=load();byid={r['uid']:r for r in main}
    baseline=json.loads(Path('research/rework/results/aggregation/predictions.json').read_text(encoding='utf8'))
    rows=json.loads((root/'outer-predictions.json').read_text(encoding='utf8'))+baseline
    variants=list(dict.fromkeys(r['variant']for r in rows));y=encode(main);w=np.array([r['recordWeight']for r in main])
    groups=sorted({r['group']for r in main});index={g:i for i,g in enumerate(groups)}
    gi=np.array([index[r['group']]for r in main]);mass=np.bincount(gi,weights=w)
    gw=w/mass[gi];supported=[c for c in MAP_CLASSES if len({r['group']for r in main if c in target(r)})>=5]
    predictions={};metrics={};counts={}
    for v in variants:
        vr=[r for r in rows if r['variant']==v]
        assert len(vr)==len(main)and {r['uid']for r in vr}==set(byid),v+' incomplete/repeated outer predictions'
        ordered={r['uid']:r for r in vr};p=encode_prediction([ordered[r['uid']]for r in main]);predictions[v]=p
        metrics[v]=hard_measures(y,p,w)
        metrics[v]['groupBalanced']=hard_measures(y,p,gw)
        si=[MAP_CLASSES.index(c)for c in supported]
        metrics[v]['macroF1_atLeast5Groups']=float(f1_score(y[:,si],p[:,si],average='macro',sample_weight=w,zero_division=0))
        metrics[v]['perClass']=classification_report(y,p,target_names=MAP_CLASSES,sample_weight=w,output_dict=True,zero_division=0)
        block=[]
        for a in [y*p,(1-y)*p,y*(1-p)]:
            b=np.zeros((len(groups),len(MAP_CLASSES)));np.add.at(b,gi,a*w[:,None]);block.append(b)
        counts[v]=block
    rng=np.random.default_rng(20261030);draw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    boot={v:f1_counts(*(draw@a for a in counts[v]))for v in variants};reference=boot['original_fixed']
    for v in variants:
        metrics[v]['pairedGroupBootstrapDeltaVsOriginal95']=np.quantile(boot[v]-reference,[.025,.975]).tolist()
    segments=json.loads((root/'outer-segments.json').read_text(encoding='utf8'))
    segmetrics={v:score_segments(manual,[r for r in segments if r['variant']==v])for v in complete['variants']}
    raw=json.loads(Path('.cache/rework-legacy-full.json').read_text(encoding='utf8'))
    base_segments=[{'uid':r['uid'],**s}for r in raw if r['uid'].startswith('manual/')for s in r['segments']]
    segmetrics['original_fixed']=score_segments(manual,base_segments)
    ref=np.array([r['accuracy']for r in segmetrics['original_fixed']['charts']])
    md=rng.multinomial(6,np.full(6,1/6),size=2000)/6
    for v,s in segmetrics.items():
        acc=np.array([r['accuracy']for r in s['charts']]);s['pairedChartBootstrapDeltaVsOriginal95']=np.quantile(md@(acc-ref),[.025,.975]).tolist()
    summary={'scope':'retrospective nested grouped CV; fixed original automatic boundaries; historical prior exposure unknown',
      'supportedClasses_atLeast5Groups':supported,'map':metrics,'segment':segmetrics}
    dump(root/'metrics.json',summary)
    for v in variants:print(v,'map',round(metrics[v]['macroF1'],4),'segment',round(segmetrics.get(v,{}).get('chartMeanAccuracy',float('nan')),4))
    lines=['# 7K 回炉：固定原分段的联合拟合结果','',
      '本阶段固定原自动边界，比较完整表示、先验适配容量、强弱监督及聚合。结果用于决定下一步；分段器与释放尺度尚未联调。',
      '', '| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |','|---|---:|---:|---:|']
    for v in variants:
        s=segmetrics.get(v,segmetrics['original_fixed']);lines.append(f"| {v} | {metrics[v]['macroF1']:.2%} | {s['chartMeanAccuracy']:.2%} | {s['durationAccuracy']:.2%} |")
    lines+=['','整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。',
      '', '模型配置由各外层的内层预测联合选择；完整配置与分数见 `search.json`，实际训练/验证/测试记录见 `protocol.json` 和 `supports.json`。',
      '', '原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。']
    if protocol['id'].endswith('-r2'):
        lines+=['','## 本轮结果的使用限制',
          '', '这是固定 Adam/LBFGS 预算的完整网格结果。训练审计发现多项拟合延长优化后仍明显变化，不能据此定最终方案。另发现仅强监督/复制弱标签控制组的 detached 目标与线搜索不一致；r3 改为先拟合其明确的局部目标、冻结局部，再拟合整谱头。',
          '', 'r3 保持数据划分、联合参数网格和最终内层选择目标，用训练梯度或训练目标及预测的连续稳定检查结束拟合，再重新选择参数和评分。已满足相同梯度条件、且训练目标无上述问题的旧拟合保留，不做无意义的重复训练。详见 `../OPTIMIZATION.md`。',
          '', '当前观察是：较大容量的残差更新没有胜过原先验；复制整谱标签也损伤了分段。这个方向需要检验更强收缩及原生分段耦合，而不是把“补了逻辑漏洞”直接等同于效果改善。优化修订后的数字出来前，这些只作为有明确预算的诊断记录。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',default='research/rework/results/mil-stage1-r2')
    main_run(parser.parse_args().directory)
