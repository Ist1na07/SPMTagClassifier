"""Consume the existing Release diagnostic; no repeated model fitting."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def f1(tp,fp,fn):
    return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=2*tp+fp+fn>0)

def report(directory):
    root=Path(directory);protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    complete=json.loads((root/'complete.json').read_text(encoding='utf8'));metrics=json.loads((root/'metrics.json').read_text(encoding='utf8'))
    raw=json.loads((root/'outer-predictions.json').read_text(encoding='utf8'));records=protocol['records'];ids=[r['uid']for r in records]
    groups=sorted({r['group']for r in records});gi=np.array([groups.index(r['group'])for r in records])
    y=np.array([r['truth']for r in records]);w=np.array([r['weight']for r in records]);rng=np.random.default_rng(20261031)
    draw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000);boots={}
    for variant in complete['variants']:
        byid={r['uid']:r for r in raw if r['variant']==variant};assert set(byid)==set(ids)
        rows=[byid[i]for i in ids];pred=np.array([r['predicted']for r in rows]);prob=np.array([r['probability']for r in rows])
        counts=[np.bincount(gi,weights=v*w,minlength=len(groups))for v in [y*pred,(1-y)*pred,y*(1-pred),(1-y)*(1-pred)]]
        tp,fp,fn,tn=[draw@v for v in counts];release=f1(tp,fp,fn);macro=(release+f1(tn,fp,fn))/2
        order=np.argsort(-prob,kind='stable');ends=np.r_[np.flatnonzero(np.diff(prob[order])!=0),len(prob)-1]
        mass=draw[:,gi[order]]*w[order];positive=mass*y[order];cum=np.cumsum(positive,axis=1)[:,ends]
        precision=cum/np.maximum(np.cumsum(mass,axis=1)[:,ends],1e-300)
        recall=cum/np.maximum(positive.sum(1,keepdims=True),1e-300)
        ap=(np.diff(np.c_[np.zeros(len(draw)),recall],axis=1)*precision).sum(1)
        # Verify the weighted/tied ranking formula against the established metric.
        test_weight=w.copy();o=order;tp0=np.cumsum(test_weight[o]*y[o])[ends];mass0=np.cumsum(test_weight[o])[ends]
        recall0=tp0/(test_weight*y).sum();manual_ap=np.dot(np.diff(np.r_[0,recall0]),tp0/mass0)
        assert abs(manual_ap-average_precision_score(y,prob,sample_weight=w))<1e-12
        boots[variant]={'macroF1':macro,'releaseF1':release,'averagePrecision':ap}
        fold_ap=[]
        for outer in range(6):
            rr=[r for r in rows if r['outer']==outer]
            if sum(r['truth']for r in rr):fold_ap.append(float(average_precision_score([r['truth']for r in rr],[r['probability']for r in rr],sample_weight=[r['weight']for r in rr])))
        metrics[variant]['meanOuterAveragePrecision']=float(np.mean(fold_ap))
        metrics[variant]['outerAveragePrecision']=fold_ap
    for variant in complete['variants']:
        for key in ['macroF1','releaseF1','averagePrecision']:
            metrics[variant][key+'95']=np.quantile(boots[variant][key],[.025,.975]).tolist()
            metrics[variant][key+'DeltaVsLegacy5695']=np.quantile(boots[variant][key]-boots['legacy56'][key],[.025,.975]).tolist()
    (root/'metrics-with-intervals.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    names=complete['variants'];fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True)
    for ax,key,title in [(axes[0],'releaseF1','Release F1 (selected inner threshold)'),(axes[1],'averagePrecision','Average precision (pooled outer scores)')]:
        for i,v in enumerate(names):
            lo,hi=metrics[v][key+'95'];ax.hlines(i,lo,hi,color='#a1b5ba',lw=2);ax.plot(metrics[v][key],i,'o',ms=5,color='#126878')
        ax.axvline(metrics['legacy56'][key],color='#7c8c91',ls='--',lw=1);ax.set_xlim(0,1);ax.set_title(title,loc='left',fontsize=11,pad=13)
        ax.grid(axis='x',alpha=.17);ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    axes[0].set_yticks(range(len(names)),names,fontsize=10);axes[0].invert_yaxis()
    fig.suptitle('Release impression in explicitly labelled LN charts',x=.035,ha='left',fontsize=15,fontweight='bold',y=.97)
    fig.text(.035,.035,'97 records / 42 song groups (17 Release-positive groups). Retuned ablations; paired group bootstrap of saved outer predictions. Not local accuracy.',fontsize=9,color='#47555a')
    fig.subplots_adjust(left=.22,right=.98,top=.86,bottom=.13,wspace=.12);fig.savefig(root/'ablation.png',dpi=170,facecolor='white');plt.close(fig)
    def interval(x):return f'[{100*x[0]:+.1f}, {100*x[1]:+.1f}]'
    lines=['# Release 整谱印象：重调后的表示消融','',
      '任务是复现明确 LN 细标签中的**整谱 Release 印象**，不是检测任意 Release 成分是否存在，也不是局部 Release 准确率。97 条记录来自 42 个歌曲/谱面组，28 条 Release 阳性来自 17 个组；重复物理文件总权重为 1。Rice-only、Mix-only、wildcard-only、未标注和六张精标谱均未混入这个二分类任务。',
      '', '## 每项消融重新拟合，并同时选择参数和阈值',
      '', f"六个外层歌曲组折，各用内层三折联合选择 C、类别权重、按时长/effort 池化，以及内层概率可以形成的全部不同阈值分割。标准化仅用训练折。所有消融使用同一范围 `{min(protocol['grid']['C']):g}…{max(protocol['grid']['C']):g}`；本轮复用 {complete.get('reusedInnerFits',0):,} 项有效拟合，新拟合 {complete.get('newInnerFits',complete['innerFits']):,} 项。所选配置及输入均不变的外层预测也复用。",
      '', '原56、修正56、删旧 tailInd、九项几何、27项机制、拼接、替换，以及几何中隔离/持键/次序三项消融均完整比较。机制表示使用固定难度模型先验，历史训练暴露未知。',
      '', '| 表示 | binary macro-F1 | Release F1 | AP | 分折平均 AP | Δmacro-F1 的95%组区间（百分点） |',
      '|---|---:|---:|---:|---:|---:|']
    for v in names:
        m=metrics[v];lines.append(f"| {v} | {m['macroF1']:.2%} | {m['releaseF1']:.2%} | {m['averagePrecision']:.2%} | {m['meanOuterAveragePrecision']:.2%} | {interval(m['macroF1DeltaVsLegacy5695'])} |")
    if protocol.get('geometryRepairSpecification',{}).get('repair',{}).get('changedSegmentCoordinates') is not None:
        lines+=['',f"本轮先修复释放序列的端点同时音歧义：三组尾时刻均为单个事件时才定义三指先后。原固定边界缓存中 {protocol['geometryRepairSpecification']['repair']['changedSegmentCoordinates']:,} 项次序坐标改变，其他八项逐值一致。不含该坐标的消融保留有效拟合；含该坐标的消融重训全部配置，不沿用旧选参。"]
    concat=metrics['legacy_plus_geometry'];legacy=metrics['legacy56'];delta=concat['macroF1DeltaVsLegacy5695']
    verdict='未支持直接拼接作为更好的决策器'if delta[0]<=0 else'在该整谱任务上支持直接拼接，局部应用仍需单独验证'
    historical=metrics['historical_fixed'];lines+=['',f"原完整流程的固定硬决策在这批记录上的 Release F1 为 **{historical['releaseF1']:.2%}**。其历史训练暴露未知，不能作为双方训练材料相同的泛化优胜；也未把硬标签伪称为连续概率来与新模型比较 AP。",'',
      '![Release特征消融](ablation.png)','',
      '## 采用什么，暂不采用什么','',
      f"事件几何能观察到原 Rbar 在头部采样栅格上漏掉的孤立尾事件，并保留没有中间头部行的连续释放次序。直接拼接原56，macro-F1 从 {legacy['macroF1']:.2%} 变为 {concat['macroF1']:.2%}，差值95%组区间为 {interval(delta)} 个百分点；{verdict}。各消融的点估计及区间见表，不用外层最高点选择特征或模型。",
      '', '合并 AP 还受到不同外折模型分数标度的影响，因此同时报告分折平均 AP。原56相关训练先验的后验诊断保存在 `../release-impression-r2/prior-score-diagnostic.json`；该变换未作为重新调参后的分类收益。',
      '', '接下来优先检验**保留原专家、限制更新容量的 LN 条件适配**，再与原生分段及聚合联调，而不将这个二分类器塞回全部局部类别头。精标 Release 仅五个区间、两个阳性谱来源，局部结论仍应使用这些真实独立支持。',
      '', '完整搜索、选参、逐折训练数据、模型和外层预测见本目录 JSON；区间与逐折 AP 见 `metrics-with-intervals.json`。区间由保存的外层预测按歌曲组重采样，已见数据的回顾性比较不冒充全新盲测。',
      '', '说明勘误：首轮协议的池化文字误写为 legacy burden；执行源码一直读取现代特征缓存的 `effort`，扩展轮协议已按实际含义写明。数值计算与旧拟合复用均未改变。']
    inner=json.loads((root/'fits.json').read_text(encoding='utf8'))
    outer=[json.loads(p.read_text(encoding='utf8'))['fit']for p in(root/'models').glob('*.json')]
    unique={(r['variant'],r['outer'],r['inner'],r['configId'])for r in inner}
    audit={'innerFitRecords':len(inner),'uniqueInnerFits':len(unique),'duplicateFitRecords':len(inner)-len(unique),'outerModels':len(outer),'innerMaxGradient':max(r['gradientMax']for r in inner),
           'outerMaxGradient':max(r['gradientMax']for r in outer),'threshold':1e-5}
    audit['accepted']=audit['innerMaxGradient']<=audit['threshold']and audit['outerMaxGradient']<=audit['threshold']
    (root/'optimizer-audit.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf8')
    lines+=['',f"优化审计：{len(inner):,} 项内层拟合、{len(outer)} 个外层模型；最大训练梯度分别为 {audit['innerMaxGradient']:.3g} / {audit['outerMaxGradient']:.3g}，接受阈值 1e-5。"]
    if len(unique)!=len(inner):
        lines+=['',f"上述轨迹中有 {len(unique):,} 项不同的配置/折拟合、{len(inner)-len(unique):,} 条重复记录。部分续跑未合并祖先的完整评分，导致重复拟合；原记录全部保留，重复项不计作新消融或独立证据。续跑读取现已修正为先合并同表示祖先、按实际配置/折去重，并保持外层模型与其选择对应。"]
    if not audit['accepted']:lines+=['','**优化审计未通过，当前结果不能作为最终模型结论。**']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='research/rework/results/release-impression-r2');report(p.parse_args().directory)
