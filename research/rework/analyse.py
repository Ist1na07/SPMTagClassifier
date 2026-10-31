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
    if 'no_update'in metrics and 'no_update'in segmetrics:
        reference=boot['no_update'];seg_reference=np.array([r['accuracy']for r in segmetrics['no_update']['charts']])
        for v in metrics:metrics[v]['pairedGroupBootstrapDeltaVsNoUpdate95']=np.quantile(boot[v]-reference,[.025,.975]).tolist()
        for v,s in segmetrics.items():
            acc=np.array([r['accuracy']for r in s['charts']]);s['pairedChartBootstrapDeltaVsNoUpdate95']=np.quantile(md@(acc-seg_reference),[.025,.975]).tolist()
    native=protocol['id'].startswith('native-');family=protocol['id'].startswith('family-cold-');capacity=protocol['id'].startswith('native-nested-capacity-');lnadapter=protocol['id'].startswith('ln-anchored-release-')
    summary={'scope':'retrospective grouped CV; '+('native boundaries/expert blend/gating/smoothing/aggregation jointly calibrated in the declared bounded space; expert coefficients frozen'if native else'fixed original automatic boundaries; cold two-family expert refits'if family else'fixed original automatic boundaries')+'; historical reference exposure unknown',
      'supportedClasses_atLeast5Groups':supported,'map':metrics,'segment':segmetrics}
    if capacity or lnadapter:summary['scope']=protocol['scope']
    optimizer=None
    if protocol['id'].startswith('7k-mil-') and (root/'outer-fits.json').exists():
        from optimizer_audit import audit_directory
        optimizer=audit_directory(root);summary['optimizer']=optimizer
    dump(root/'metrics.json',summary)
    for v in variants:print(v,'map',round(metrics[v]['macroF1'],4),'segment',round(segmetrics.get(v,{}).get('chartMeanAccuracy',float('nan')),4))
    lines=['# 7K 回炉：'+('原生流程联合校准结果'if native else'原双专家冷启动联合拟合结果'if family else'固定原分段的联合拟合结果'),'',
      ('本阶段保留原专家系数先验，在同一最终任务目标下联合选择原生边界、专家混合/LN 门控、显示策略、相对负荷门控及聚合。每项消融重选全部可用参数；这是预先给定空间和候选预算内的比较，不是所有连续参数的全局最优。'if native else
       '本阶段保留原 RC/LN 双专家、规则覆盖、wildcard 与 Break 结构，但只用各训练折从零估计标准化与两个专家权重。源风格逐段复制、等谱权重复制和弱整谱 bag 监督分别重训，全部专家配置与最终门控、规则混合、显示和聚合共同选择；不先按局部代理成绩淘汰模型。固定原边界，尚未检验冷启动专家与原生分段联调。'if family else
       '本阶段固定原自动边界，比较完整表示、先验适配容量、强弱监督及聚合。结果用于决定下一步；分段器与释放尺度尚未联调。'),
      '', '| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |','|---|---:|---:|---:|']
    for v in variants:
        s=segmetrics.get(v,segmetrics['original_fixed']);lines.append(f"| {v} | {metrics[v]['macroF1']:.2%} | {s['chartMeanAccuracy']:.2%} | {s['durationAccuracy']:.2%} |")
    lines+=['','整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。',
      '', ('模型配置由各外层的内层预测联合选择；各方案的 `candidates.json`、`search.json.gz`、`selection.json` 与 `candidate-predictions.npz` 保留完整候选、评分和实际输出。实际分组见 `protocol.json`'+(' 和 `supports.json`'if(root/'supports.json').exists()else'')+'。'if native else
           '每个方案/外层目录保存 `inner-models.json.gz`、`candidates.json.gz`、`search.json.gz`、`inner-predictions.npz`、先写入的 `selection.json` 及 `outer-model.json`。实际划分和源码快照见 `protocol.json`。全零规则输出按家族 wildcard 弃权，不任意采用字典首类；这项明确修正与归档纯规则路径有差别。'if family else
           '模型配置由各外层的内层预测联合选择；完整配置与分数见 `search.json`，实际训练/验证/测试记录见 `protocol.json` 和 `supports.json`。'),
      '', '原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。']
    if protocol['id']=='7k-mil-fixed-original-segmentation-r2':
        lines+=['','## 本轮结果的使用限制',
          '', '这是固定 Adam/LBFGS 预算的完整网格结果。训练审计发现多项拟合延长优化后仍明显变化，不能据此定最终方案。另发现仅强监督/复制弱标签控制组的 detached 目标与线搜索不一致；r3 改为先拟合其明确的局部目标、冻结局部，再拟合整谱头。',
          '', 'r3 保持数据划分、联合参数网格和最终内层选择目标，用训练梯度或训练目标及预测的连续稳定检查结束拟合，再重新选择参数和评分。已满足相同梯度条件、且训练目标无上述问题的旧拟合保留，不做无意义的重复训练。详见 `../OPTIMIZATION.md`。',
          '', '当前观察是：较大容量的残差更新没有胜过原先验；复制整谱标签也损伤了分段。这个方向需要检验更强收缩及原生分段耦合，而不是把“补了逻辑漏洞”直接等同于效果改善。优化修订后的数字出来前，这些只作为有明确预算的诊断记录。']
    if optimizer is not None:
        from optimizer_audit import report_lines
        lines+=report_lines(optimizer)
        if complete.get('reusedOuterFits') is not None:
            lines+=['',f"本轮只续训未达条件的内层候选，再重新联合选择；外层新拟合 {complete['newOuterFits']}，相同配置及划分复用 {complete['reusedOuterFits']}。"]
    if capacity:
        union=protocol['id']=='native-nested-capacity-union-r1'
        lines[0]='# 7K：嵌套选择流程更新容量'
        lines[2]='保持父实验全部固定预测不变。'+('先合并全部非历史候选，逐项限制从同一并集筛选，不让控制组另抽一套候选。'if union else'')+'在每个外层的内层训练子集上，各容量预算同时选择整条策略；其内层验证输出再选择容量，最后在完整外层训练组上重选策略。主候选为 '+('pool_repaired_nested'if union else'metric_repaired')+'，其余限制各自重调；固定最小/不限制容量只作机制对照。'
        lines=[line for line in lines if not line.startswith('模型配置由各外层的内层预测联合选择；各方案的')]
        lines+=['',('全部联合候选见根目录 `candidates.json.gz`，位打包的完整预测及来源见 `candidate-predictions.npz` / `candidate-origins.npz`；类序与编码、重复逐值核对见 `pool-audit.json`。所有实际训练选择作用域的分数在 `scope-scores/`，每项限制的候选索引与内层容量/策略选择在其目录。根目录 `costs.json` 保存机制改变数。原始父预测及原生分段缓存见 `../native-metric-r1/`，实际划分、源参数与生产源码见 `protocol.json`。'if union else'父候选、全部固定预测和分段缓存见 `../native-metric-r1/`。本轮各方案 `selection.json` 保存实际内层容量成绩、每次整策略选择与最终外层配置，`costs.json` 保存所有候选的机制改变数；实际嵌套划分和父协议保存在 `protocol.json`。')]
    if lnadapter:
        lines[0]='# 7K：原 LN 专家的 Release 小幅校正'
        lines[2]='固定原自动边界、原RC专家及LN先验/规则，只学习Release相对其余LN类的可收缩logit校正。主表示geometry3为隔离、其他指持键和交互；其余为重训消融。每个训练配置完整交叉同一512项后处理策略，并包含精确零更新头；全部按最终内层任务共同选择。弱标签监督整谱印象，不复制为局部真值。'
        lines=[line for line in lines if not line.startswith('模型配置由各外层的内层预测联合选择；完整配置')]
        lines+=['','## 与同一后处理池的零更新控制比较','','下表为各方案减no_update的百分点差及成对95%区间；原固定算法仍作为上表的历史参考。主方案固定为geometry3，不从外层消融中另选赢家。',
                '', '| 方案 | Δ整谱macro-F1［95%］ | Δ分段按谱［95%］ |','|---|---:|---:|']
        fmt=lambda x:f'[{100*x[0]:+.2f}, {100*x[1]:+.2f}]'
        for v in complete['variants']:
            lines.append(f"| {v} | {100*(metrics[v]['macroF1']-metrics['no_update']['macroF1']):+.2f} {fmt(metrics[v]['pairedGroupBootstrapDeltaVsNoUpdate95'])} | {100*(segmetrics[v]['chartMeanAccuracy']-segmetrics['no_update']['chartMeanAccuracy']):+.2f} {fmt(segmetrics[v]['pairedChartBootstrapDeltaVsNoUpdate95'])} |")
        lines+=['','各方案/外层保存全部头×共同策略评分、原始内层预测、先写入的选择、所选外层模型和实际预测；内层拟合轨迹在 `inner-fits.json.gz`，完整训练作用域和旧专家系数在 `protocol.json`。此实验不是与原生边界重训联调。']
    if protocol['id'].startswith('ln-extra-gate-'):
        lines[0]='# 7K：修订额外负荷门控后的完整联合选择'
        lines[2]='保留原自动边界、旧版Break/家族逻辑和原RC/LN专家先验。只修订研究流程新增的相对头部负荷门控；原辅助LN专家训练目标不变，复用所有有效拟合后，完整交叉并重选头配置与最终后处理。主表示固定geometry3，其余为消融，不以外层最高者替换主方案。'
        lines=[line for line in lines if not line.startswith('模型配置由各外层的内层预测联合选择；完整配置')]
        lines+=['','各表示/外层保存全头×策略的 `search.json.gz`、实际 `inner-predictions.npz`、`selection.json`、`outer-model.json` 和源拟合复用轨迹。根研究协议、门控成对比较、优化作用域审计与Release误报/召回见父目录，不能把复用专家计成新增内层训练。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',default='research/rework/results/mil-stage1-r2')
    main_run(parser.parse_args().directory)
