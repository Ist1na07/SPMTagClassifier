"""Audit the complete feature experiment and plot actual inner profiles."""
import argparse
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
def divide(a,b):return np.divide(a,b,out=np.zeros_like(np.asarray(a),dtype=float),where=np.asarray(b)>0)


def binary(y,p,w):
    tp=(w*(y==1)*(p==1)).sum(-1);fp=(w*(y==0)*(p==1)).sum(-1)
    fn=(w*(y==1)*(p==0)).sum(-1);tn=(w*(y==0)*(p==0)).sum(-1)
    return {'macroF1':.5*(divide(2*tp,2*tp+fp+fn)+divide(2*tn,2*tn+fp+fn)),
        'releaseF1':divide(2*tp,2*tp+fp+fn),'precision':divide(tp,tp+fp),'recall':divide(tp,tp+fn)}


def weighted_ap(y,probability,weights):
    order=np.argsort(-probability,kind='stable');p=probability[order];target=y[order]
    w=weights[...,order];tp=np.cumsum(w*target,axis=-1);mass=np.cumsum(w,axis=-1)
    ends=np.flatnonzero(np.r_[p[:-1]!=p[1:],True]);a=tp[...,ends];b=mass[...,ends]
    increments=np.diff(a,axis=-1,prepend=np.zeros(a.shape[:-1]+(1,)))
    return divide((divide(a,b)*increments).sum(-1),tp[...,-1])


def run(directory):
    root=Path(directory);protocol=read(root/'protocol.json');complete=read(root/'complete.json');rows=read(root/'outer-predictions.json');metrics=read(root/'metrics.json')
    inventory=protocol['records'];uids=[r['uid']for r in inventory];y=np.array([r['truth']for r in inventory]);w=np.array([r['weight']for r in inventory]);groups=[r['group']for r in inventory]
    unique,gi=np.unique(groups,return_inverse=True);draw=np.random.default_rng(20261101).multinomial(len(unique),np.full(len(unique),1/len(unique)),size=2000);bw=draw[:,gi]*w
    boots={};profiles={};audit={'innerFits':0,'outerFits':0,'maximumGradient':0.};predictions={}
    for v in complete['variants']:
        byid={r['uid']:r for r in rows if r['variant']==v}
        if set(byid)!=set(uids)or sum(r['variant']==v for r in rows)!=len(uids):raise ValueError('Outer inventory differs')
        p=np.array([byid[i]['predicted']for i in uids]);probability=np.array([byid[i]['probability']for i in uids]);predictions[v]=p
        boots[v]=binary(y,p,bw);boots[v]['averagePrecision']=weighted_ap(y,probability,bw)
        point=binary(y,p,w)
        if any(abs(float(value)-metrics[v][key])>1e-12 for key,value in point.items()):raise ValueError('Independent metric calculation differs')
        profiles[v]=[]
        for o in range(6):
            folder=root/v/str(o);search=read(folder/'search.json');traces=read(folder/'fits.json');selected=read(folder/'selection.json')
            with np.load(folder/'inner-predictions.npz') as saved:
                inner=saved['probability'];ids=saved['validationIds']
            reference=inner[selected['configId']];reference_tag=reference>=selected['threshold'];iw=w[ids]
            if inner.shape!=(len(search),len(ids)) or len(set(ids.tolist()))!=len(ids):raise ValueError('Inner prediction inventory differs')
            if len(search)!=len(protocol['configs'][v])or len(traces)!=3*len(search):raise ValueError('Missing complete joint grid')
            for trace in traces+[read(folder/'outer-model.json')['fit']]:
                audit['maximumGradient']=max(audit['maximumGradient'],trace['gradientMax'])
                if trace['gradientMax']>1e-5:raise ValueError('Unaccepted optimizer gradient')
            audit['innerFits']+=len(traces);audit['outerFits']+=1
            for coordinate in ['C','releaseScaleMs']:
                values=list(dict.fromkeys(r['config'][coordinate]for r in search))
                if values==[None]:continue
                curve=[]
                for value in values:
                    candidates=[r for r in search if r['config'][coordinate]==value]
                    best=max(candidates,key=lambda r:(r['macroF1'],r['releaseF1'],r['AP'],-r['config']['C'],-r['configId']))
                    probability=inner[best['configId']];predicted=probability>=best['threshold']
                    curve.append({'value':value,'bestConfig':best['config'],'threshold':best['threshold'],
                        'innerMacroF1':best['macroF1'],'innerReleaseF1':best['releaseF1'],'innerAP':best['AP'],
                        'predictionFlipMassFromSelected':float(iw@(predicted!=reference_tag)/iw.sum()),
                        'meanAbsoluteProbabilityChange':float(iw@np.abs(probability-reference)/iw.sum()),
                        'brier':float(iw@((probability-y[ids])**2)/iw.sum()),
                        'nll':float(-iw@(y[ids]*np.log(np.clip(probability,1e-15,1))+(1-y[ids])*np.log(np.clip(1-probability,1e-15,1)))/iw.sum())})
                profiles[v].append({'outer':o,'coordinate':coordinate,'selectedValue':selected['config'][coordinate],'points':curve})
    if audit['innerFits']!=complete['innerFits']:raise ValueError('Fit accounting differs')
    effects={}
    for candidate,baseline in protocol['primaryComparisons']:
        effects[candidate+' vs '+baseline]={k:{'pointDelta':metrics[candidate][k]-metrics[baseline][k],
            'delta95':np.quantile(boots[candidate][k]-boots[baseline][k],[.025,.975]).tolist()}for k in boots[candidate]}
    write(root/'paired-effects.json',{'scope':'matched saved outer outputs, paired song-group bootstrap2000draws; auxiliary frozen rating prior exposure retained','effects':effects})
    write(root/'parameter-profiles.json',{'scope':'actual inner profiles; every coordinate value reoptimizes all other effective declared parameters and threshold partitions','profiles':profiles})
    write(root/'optimizer-audit.json',audit)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    panels=[(['old3','retained3'],'C'),(['rating_plus_old3','rating_plus_retained3'],'C'),(['retained3'],'releaseScaleMs'),(['rating_plus_retained3'],'releaseScaleMs')]
    colors=['#167b80','#ba5c72']
    for ax,(variants,coordinate)in zip(axes.flat,panels):
        for color,v in zip(colors,variants):
            curves=[r['points']for r in profiles[v]if r['coordinate']==coordinate];x=np.array([r['value']for r in curves[0]])
            a=np.array([[p['innerMacroF1']for p in c]for c in curves])
            for line in a:ax.plot(x,line,color=color,alpha=.2,lw=.8)
            ax.plot(x,a.mean(0),color=color,label=v,lw=2)
        ax.set_xscale('log');ax.set_xlabel(coordinate);ax.set_ylabel('Inner binary macro-F1');ax.grid(alpha=.15);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Release mechanism study: complete conditional parameter profiles',fontweight='bold')
    fig.savefig(root/'parameter-profiles.png',dpi=160,facecolor='white');plt.close(fig)
    lines=['# 放手机制与SPMRating复用：完整联合选参结果','',protocol['scope'],'',
        '97条明确LN细标签记录；使用历史相同六外层/三内层歌曲组。构造谱只定义问题，没有进入拟合或评价。SPMRating是冻结的外部难度先验，结论按回顾性分组评价报告。',
        '', '| 方案 | binary macro-F1 | Release P | Release R | Release F1 | pooled AP | mean-fold AP |','|---|---:|---:|---:|---:|---:|---:|']
    for v,m in metrics.items():lines.append(f"| {v} | {m['macroF1']:.2%} | {m['precision']:.2%} | {m['recall']:.2%} | {m['releaseF1']:.2%} | {m['averagePrecision']:.2%} | {m['meanFoldAP']:.2%} |")
    lines+=['','预先指定的两项主比较与成对95%区间：']
    for name,values in effects.items():
        lines.append('\n- '+name)
        for key,e in values.items():lines.append(f"  - {key}: {100*e['pointDelta']:+.2f}pp [{100*e['delta95'][0]:+.2f}, {100*e['delta95'][1]:+.2f}]")
    lines+=['',f"完整审计：{audit['innerFits']}项内层拟合，{audit['outerFits']}项外层拟合，最大实际归一梯度{audit['maximumGradient']:.3g}，全部达到原1e−5停止条件。",'',
        '每个有效配置重新拟合，阈值穷尽实际内层分数的全部不同预测分区。retained尺度与正则、类别权重、聚合指数共同选择；无该尺度的控制不重复无效拟合。profile在固定某个值时重选其余参数，不用未经匹配的随机池估计曲率。', '',
        'pooled AP混合不同折模型的分数尺度，因此同时给出mean-fold AP；未增加事后外层校准。此任务只检验整谱Release证据，不提供分段准确率，也不等于完整17类tag方案。', '',
        '## 参数辨识仍有缺口', '',
        '图中的粗线是六折条件最优分数的均值，细线保留每折。每点均重选其余参数，因此不是固定其他参数的局部曲率。parameter-profiles.json同时保存每点相对本折选择的预测翻转质量、概率变化、NLL和Brier，避免把同分误读为同预测。', '',
        'retained3的正则C出现宽平台；释放尺度在部分折及均值上仍向120ms边界改善，当前四尺度网格没有证明尺度最优点已被包围。需要在内层补边界/极限形状对照后，才决定是否保留独立尺度。不能把这次有限网格的完整搜索称为参数已经辨识。', '',
        '主比较的区间均覆盖零；现有证据支持继续研究放手后的保留指约束，尚不足以替换完整分类器。SPMRating拼接控制的下降也保留：新增物理描述并不保证在已有相关特征上产生增益。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print(metrics,audit,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='research/frontier/results/release-feature-r1');run(p.parse_args().directory)
