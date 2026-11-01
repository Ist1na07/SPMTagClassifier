"""Audit matched scale profiles and selection stability on saved INNER scores."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from release_feature_report import read,write,divide,weighted_ap


def threshold_optima(y,p,weights):
    """All distinct >= threshold partitions, for many group reweightings."""
    unique=np.unique(p);thresholds=np.r_[np.nextafter(1.,2.),((unique[:-1]+unique[1:])/2)[::-1],0.]
    if np.any(((unique[:-1]+unique[1:])/2)<=unique[:-1]):raise ValueError('Unrepresentable midpoint; threshold enumeration needs upper-endpoint fallback')
    order=np.argsort(-p,kind='stable');ends=np.flatnonzero(np.r_[p[order][:-1]!=p[order][1:],True]);w=weights[:,order]
    tp=np.c_[np.zeros(len(w)),np.cumsum(w*y[order],axis=1)[:,ends]]
    fp=np.c_[np.zeros(len(w)),np.cumsum(w*(1-y[order]),axis=1)[:,ends]]
    fn=tp[:,-1,None]-tp;tn=fp[:,-1,None]-fp
    positive=divide(2*tp,2*tp+fp+fn);macro=.5*(positive+divide(2*tn,2*tn+fp+fn))
    t=np.broadcast_to(thresholds,macro.shape)
    at=np.lexsort((-t,-abs(t-.5),positive,macro),axis=1)[:,-1];r=np.arange(len(w))
    return thresholds[at],macro[r,at],positive[r,at]


def run():
    root=Path('research/frontier/results/scale-boundary-r2');protocol=read(root/'protocol.json');complete=read(root/'complete.json')
    records=protocol['records'];y=np.array([r['truth']for r in records]);weight=np.array([r['weight']for r in records]);profiles={};stability={};gradient=0.;fit_count=0
    for variant in protocol['variants']:
        profiles[variant]=[];stability[variant]=[]
        for outer in range(6):
            folder=root/variant/str(outer);search=read(folder/'search.json');selection=read(folder/'selection.json');traces=read(folder/'new-fits.json')
            fit_count+=len(traces);gradient=max(gradient,max(t['gradientMax']for t in traces))
            if len(traces)!=3*len(protocol['extensionConfigs']):raise ValueError('Incomplete extension grid')
            with np.load(folder/'inner-predictions.npz')as saved:p=saved['probability'];ids=saved['validationIds']
            if p.shape!=(len(search),len(ids)):raise ValueError('Prediction/configuration inventory differs')
            iw=weight[ids];iy=y[ids];ref=p[selection['configId']]>=selection['threshold'];points=[]
            for shape in [15,30,60,120,240,480,960,'infinity']:
                best=max((r for r in search if r['config']['releaseScaleMs']==shape),key=lambda r:(r['macroF1'],r['releaseF1'],r['AP'],-r['config']['C'],-r['configId']))
                q=p[best['configId']];tag=q>=best['threshold']
                points.append({'shape':shape,'config':best['config'],'threshold':best['threshold'],'macroF1':best['macroF1'],'releaseF1':best['releaseF1'],'AP':best['AP'],
                    'predictionFlipMassFromSelected':float(iw@(tag!=ref)/iw.sum()),'probabilityChangeFromSelected':float(iw@abs(q-p[selection['configId']])/iw.sum())})
            profiles[variant].append({'outer':outer,'selection':selection,'points':points})
            groups,gi=np.unique([records[i]['group']for i in ids],return_inverse=True)
            draw=np.random.default_rng(20261102+outer).multinomial(len(groups),np.full(len(groups),1/len(groups)),size=200)
            bw=np.r_[iw[None,:],draw[:,gi]*iw]
            choices=[]
            for row,probability in zip(search,p):
                t,m,f=threshold_optima(iy,probability,bw);ap=weighted_ap(iy,probability,bw)
                # Row0 is an independent check against the producer's nominal choice.
                if abs(m[0]-row['macroF1'])>1e-12 or abs(t[0]-row['threshold'])>1e-12:raise ValueError('Threshold recomputation differs')
                choices.append((t,m,f,ap))
            selected=[]
            for b in range(len(bw)):
                winner=max(range(len(search)),key=lambda j:(choices[j][1][b],choices[j][2][b],choices[j][3][b],-search[j]['config']['C'],-j))
                threshold=float(choices[winner][0][b]);predicted=p[winner]>=threshold
                selected.append({'configId':winner,'threshold':threshold,'config':search[winner]['config'],
                    'predictionFlipMassFromNominal':float(iw@(predicted!=ref)/iw.sum())})
            if selected[0]['configId']!=selection['configId']:raise ValueError('Nominal complete joint selection differs')
            draws=selected[1:];frequency=Counter(str(r['config']['releaseScaleMs'])for r in draws);flips=[r['predictionFlipMassFromNominal']for r in draws]
            stability[variant].append({'outer':outer,'resamples':200,'songGroups':len(groups),'shapeFrequency':dict(frequency),
                'predictionFlipMassQuantiles':np.quantile(flips,[0,.5,.9,1]).tolist(),'selections':draws})
    if fit_count!=complete['newInnerFits']or gradient>1e-5:raise ValueError('Fit accounting/gradient failed')
    write(root/'profiles.json',{'scope':protocol['scope'],'maximumNewGradient':gradient,'profiles':profiles})
    write(root/'selection-stability.json',{'scope':'200 inner-validation song-group reweightings, full joint configuration AND threshold reselection; fixed fitted coefficients, not a training-refit bootstrap or generalization CI','stability':stability})
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    labels=['15','30','60','120','240','480','960','infinity shape']
    for ax,(variant,rows)in zip(axes,profiles.items()):
        values=np.array([[p['macroF1']for p in r['points']]for r in rows])
        for line in values:ax.plot(range(8),line,color='#318b8d',alpha=.3,lw=1)
        ax.plot(range(8),values.mean(0),color='#194f66',lw=2.5,marker='o',ms=4,label='six-fold mean conditional optimum')
        ax.axvline(3.5,color='#ad6479',ls='--',lw=1,label='original grid boundary');ax.set_xticks(range(8),labels,rotation=30,ha='right')
        ax.set_title(variant);ax.set_xlabel('release scale (ms), then analytic rescaled limit');ax.set_ylabel('Inner binary macro-F1');ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Boundary extension: reoptimize all other parameters at every shape',fontweight='bold');fig.savefig(root/'profiles.png',dpi=160,facecolor='white');plt.close(fig)
    lines=['# 释放尺度：补网格边界，并检查选参是否改变预测','',
        '新增240/480/960ms及τ→∞的解析重标度形状。每个点完整联调C、类别权重、困难聚合指数和所有阈值分区；两个表示共新增7,488项内层拟合，复用原7,488项，不重复训练，也不产生新的外层预测。', '',
        '∞形状为已按秒数×剩余持键秒数，是指数乘积乘τ²后的极限；它不是无穷大的数值参数，也不是[0,1]保留指比例。训练内标准化处理尺度。', '',
        '| 表示 | 外折 | 原尺度 | 扩展选择 | 内层macro-F1增量 | 200次选择重采样的预测翻转质量中位数 |','|---|---:|---:|---:|---:|---:|']
    for variant,rows in profiles.items():
        for row,s in zip(rows,stability[variant]):
            outer=row['outer'];old=read(Path(protocol['sourceStudy'])/variant/str(outer)/'selection.json');new=row['selection']
            lines.append(f"| {variant} | {outer} | {old['config']['releaseScaleMs']} | {new['config']['releaseScaleMs']} | {100*(new['macroF1']-old['macroF1']):+.3f}pp | {s['predictionFlipMassQuantiles'][1]:.1%} |")
    lines+=['',f'新拟合最大实际归一梯度{gradient:.3g}，全部达到1e−5条件。完整条件profile及预测变化见profiles.json；逐次联合重选择和尺度频数见selection-stability.json。', '',
        '重采样按内层验证歌曲组重新加权，对每个配置重新选阈值，再选完整配置。模型系数固定，因而测量的是验证组构成引起的选择不稳定，不是完整重训的不确定性。名义权重的重算与原生产选择逐项一致。', '',
        '单独retained3仅一折扩展到240ms，内层增加0.355pp，其余折保留原尺度；组件拼接有一折到480ms、一折到∞形状。由此可见尺度与表示存在交互，不能把六折均值曲线当作每折同一个井。', '',
        '15ms仍在部分折获选，下边界尚未被包围；零尺度又会退化成对极小放手错位敏感的硬计数，不能直接作为可靠候选。下一步需结合时间扰动稳定性判断短尺度是否真实可辨识，而不是靠更细网格强行制造尖井。', '',
        '这次不更新原外层成绩，也不以扩大网格后的内层提升宣布泛化收益。先与压力路由/动作原型研究衔接，再按同一完整多指标协议评价候选。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print({'newInnerFits':fit_count,'maximumGradient':gradient,'selectionResamples':2400},flush=True)


if __name__=='__main__':run()
