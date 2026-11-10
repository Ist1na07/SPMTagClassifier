"""Consume complete nested results; no partial-OOF performance claims."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import Data,Kernel,fit,read,write,metrics,local_metrics,summary,PLANS
# Avoid the perception study module, already imported by competition.
import importlib.util
spec=importlib.util.spec_from_file_location('competition_study',Path(__file__).with_name('study.py'));study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)
ROOT=study.ROOT


def run():
    read(ROOT/'complete.json');d=Data();baseline_pred,_=d.chart_predictions(d.support);baseline_local=local_metrics(d,d.manual)
    base=dict(whole=metrics(d.y[d.main],baseline_pred[d.main],d.weight[d.main]),local=summary(baseline_local))
    selections=[read(ROOT/str(o)/'selection.json')for o in range(6)];grids=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)]
    # Common production candidate selection is frozen from mean INNER values
    # before any pooled OUTER metric is read below.
    pooled=[]
    for ci,c in enumerate(study.configurations()):
        a=[g[ci]for g in grids];pooled.append(dict(configId=ci,config=c,
            whole={k:float(np.mean([r['whole'][k]for r in a]))for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']},
            local={k:float(np.mean([r['local'][k]for r in a]))for k in ['accuracy','techToDenseMs','techCsConfusionMs']}))
    b=dict(configId=-1,config=dict(features='frozen',strongWeight=0.,wholeWeight=0.,teacherWeight=0.,ridge=0.),
        whole={k:float(np.mean([s['baseline']['whole'][k]for s in selections]))for k in pooled[0]['whole']},
        local={k:float(np.mean([s['baseline']['local'][k]for s in selections]))for k in pooled[0]['local']})
    common={arm+'/'+guard:study.select(pooled,b,arm,guard=='protected')for arm in study.ARMS for guard in ['balanced','protected']}
    write(ROOT/'common-selection.json',dict(usesOuterTruth=False,baseline=b,choices=common,allInnerMeans=pooled))
    result={};details={}
    for name in common:
        actual=np.zeros_like(d.baseline);ls=[];changes=[]
        for o,p in enumerate(PLANS):
            rows=read(ROOT/str(o)/'outer-predictions.json')[name]
            for r in rows:
                i=d.lookup[r['uid']];actual[i]=[c in r['predicted']for c in d.classes]
                if set(r['predicted'])!=set(c for c,v in zip(d.classes,baseline_pred[i])if v):changes.append(r)
            ls.extend(read(ROOT/str(o)/'outer-local.json')[name])
        result[name]=dict(whole=metrics(d.y[d.main],actual[d.main],d.weight[d.main]),local=summary(ls),
            selectedConfigIds=[s['choices'][name]['configId']for s in selections],changedWholeCharts=len(changes))
        details[name]=dict(local=ls,changedWhole=changes)
    write(ROOT/'oof.json',dict(scope='Retrospective local-parameter/hyperparameter source-group held out; fixed historical v1.2.3 readout/prior. Not blind full-pipeline validation.',baseline=base,metrics=result,details=details))
    plateaus=[]
    for o,(s,grid)in enumerate(zip(selections,grids)):
        choice=s['choices']['all/protected'];v=study.score(choice);near=[r for r in grid if abs(study.score(r)-v)<=.002]
        with np.load(ROOT/str(o)/'inner-vectors.npz')as z:ps=z['predictions']
        fingerprints={np.packbits(ps[r['configId']]).tobytes()for r in near}
        plateaus.append(dict(outer=o,selectedConfigId=choice['configId'],nearScoreTolerance=.002,nearConfigs=len(near),distinctWholePredictionVectors=len(fingerprints),
            localAccuracyRange=[min(r['local']['accuracy']for r in near),max(r['local']['accuracy']for r in near)]if near else None))
    write(ROOT/'plateaus.json',plateaus)
    lines=['# 固定积分聚合，局部分类能否真正改善？','',
        '基线 v1.1.0；本表直接对照当前 v1.2.3 的固定局部/固定 R1 readout。数据为历史反复使用的349整谱和6人工分段，按原6×3源/歌曲分组训练局部头与选参，不称新盲测。','',
        '保护选择允许整谱各主指标最多1个百分点的 INNER 代价，要求 INNER 局部均值及 Tech/CS 混淆不退步；OUTER 结果未用于重新挑选。整谱控制列是固定成品配置的回放，不冒充原 R1 的逐折选择分数。','',
        '| 对照/消融 | 局部均值准确率 | Tech→Dense秒 | Tech↔CS秒 | 整谱P | 整谱R | 整谱F1 | microF1 | exact |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in [('fixed v1.2.3',base)]+list(result.items()):
        m=r['whole'];l=r['local'];lines.append('| '+name+' | '+f"{l['accuracy']:.2%} | {l['techToDenseMs']/1000:.3f} | {l['techCsConfusionMs']/1000:.3f} | "+' | '.join(f'{m[k]:.2%}'for k in ['macroPrecision','macroRecall','macroF1','microF1','exactMatch'])+' |')
    lines+=['','## 平地与井','',
        'TRAIN 内先删除常数/线性相关输入，再以正交类别对比删除公共平移。剩余每个参数有显式正曲率惩罚。非线性目标仍可能有多个局部极小；每次两起点和同范围强正则 continuation 的损失差均保留，不称全局最优。',
        'plateaus.json 报告近似等分配置对应多少不同整谱预测向量及局部准确率范围；不会仅凭参数值差异断言输出不稳定。','',
        'Break、家族路由和零分支没有被本模型解锁。learnability-r2.json 单列这部分真实标注误差；不能把它们算成“参数没找到”。E/F/G 等主动实谱局部答案没有用于训练/选参，后续机制回归必须按原歌曲留出模型读取。','',
        'common-selection.json 在汇总 OUTER 前由 INNER 均值写入。当前仅产出研究候选；未自动修改生产默认分段或发布新版本。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[6:]),flush=True)
    selected=common['all/protected']
    if selected['configId']>=0:
        k=Kernel(d,d.main,d.manual,selected['config']);theta,receipt=fit(k)
        write(ROOT/'candidate.json',dict(model=k.export(theta),receipt=receipt,selection=selected,
            status='Research candidate; adoption requires actual local/whole outcomes and runtime acceptance'))


if __name__=='__main__':run()
