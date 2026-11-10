"""Full routing outcomes, matched feedback ablations, common INNER selection."""
from pathlib import Path
import sys,importlib.util
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from routing import Data,Kernel,fit,read,write,metrics,local_metrics,summary,PLANS
spec=importlib.util.spec_from_file_location('routing_study_spec',Path(__file__).with_name('routing_study.py'));study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study);ROOT=study.ROOT


def run():
    read(ROOT/'complete.json');d=Data();old,_=d.chart_predictions(d.support);base=dict(whole=metrics(d.y[d.main],old[d.main],d.weight[d.main]),local=summary(local_metrics(d,d.manual)))
    sels=[read(ROOT/str(o)/'selection.json')for o in range(6)];grids=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)];pooled=[]
    for ci,c in enumerate(study.configurations()):
        rs=[g[ci]for g in grids];pooled.append(dict(configId=ci,config=c,whole={k:float(np.mean([r['whole'][k]for r in rs]))for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']},local={k:float(np.mean([r['local'][k]for r in rs]))for k in ['accuracy','techToDenseMs','techCsConfusionMs']}))
    b=dict(configId=-1,config=dict(features='frozen',strongWeight=0.,wholeWeight=0.,teacherWeight=0.,ridge=0.),whole={k:float(np.mean([s['baseline']['whole'][k]for s in sels]))for k in pooled[0]['whole']},local={k:float(np.mean([s['baseline']['local'][k]for s in sels]))for k in pooled[0]['local']})
    common={arm+'/'+guard:study.select(pooled,b,arm,guard=='protected')for arm in study.ARMS for guard in ['balanced','protected']};write(ROOT/'common-selection.json',dict(usesOuterTruth=False,baseline=b,choices=common,innerMeans=pooled))
    result={};details={}
    for name in common:
        p=np.zeros_like(d.baseline);ls=[];changes=[]
        for o in range(6):
            for r in read(ROOT/str(o)/'outer-predictions.json')[name]:
                i=d.lookup[r['uid']];p[i]=[c in r['predicted']for c in d.classes]
                if np.any(p[i]!=old[i]):changes.append(r)
            ls.extend(read(ROOT/str(o)/'outer-local.json')[name])
        result[name]=dict(whole=metrics(d.y[d.main],p[d.main],d.weight[d.main]),local=summary(ls),selectedConfigIds=[s['choices'][name]['configId']for s in sels],changedWholeCharts=len(changes));details[name]=dict(local=ls,wholeChanges=changes)
    write(ROOT/'oof.json',dict(scope='Historical fixed product/readout, source-group-held-out new routing parameters/selection; not a blind full-pipeline estimate',baseline=base,metrics=result,details=details))
    lines=['# 解锁 Break/家族路由：局部与整谱反馈的配对结果','',
        '仍属 v1.2 迭代，原始基线 v1.1.0，直接控制为固定 v1.2.3。路由先验和整谱 readout 是历史知识；新路由参数及调参按原6×3源/歌曲分组。未训练新增主动标注，也未复制整谱真值到分段。','',
        '| 模型/消融 | 局部均值准确率 | Tech→Dense秒 | Tech↔CS秒 | 整谱P | 整谱R | 整谱F1 | microF1 | exact |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,r in [('fixed v1.2.3',base)]+list(result.items()):
        l=r['local'];m=r['whole'];lines.append('| '+name+' | '+f"{l['accuracy']:.2%} | {l['techToDenseMs']/1000:.3f} | {l['techCsConfusionMs']/1000:.3f} | "+' | '.join(f'{m[k]:.2%}'for k in ['macroPrecision','macroRecall','macroF1','microF1','exactMatch'])+' |')
    lines+=['','## 解释与接入边界','',
        '这里真正改变可见分段的 Break/RC/LN 路由；局部具体键型仍由对应的真实专家给出。整谱支持由软路由概率乘家族专家支持，再沿连续负担积分；Hybrid 家族份额也参与反馈，不再是固定路由常量。',
        'absolute 没有整谱相对上下文；context 加入相对头/行/尾速率、相对持键和手内事件几何；pressure 再加入总需求及 R/P、C/P、J/P 对比，Rbar没有孤立使用。无评分接口可独立选择 context；pressure 需要已有的完整分量曲线，不能默默补零。',
        '保护选择要求 INNER 局部准确率/Tech-CS不退步，整谱 macroP/R/F1/micro各至多1pp损失；它不是保证 OUTER 达标的文字承诺。common-selection 在汇总OUTER前由INNER均值冻结。研究候选未自动替换默认生产模型。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)
    for name in ['all/protected','no_rating/protected','pressure/protected','joint/protected','local_only/protected']:
        r=common[name]
        if r['configId']<0:continue
        k=Kernel(d,d.main,d.manual,r['config']);theta,receipt=fit(k);write(ROOT/('candidate-'+name.replace('/','-')+'.json'),dict(model=k.export(theta),receipt=receipt,selection=r,status='Research candidate, not automatically adopted'))


if __name__=='__main__':run()
