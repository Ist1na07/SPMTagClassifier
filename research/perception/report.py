"""Freeze common INNER choices before consuming any OUTER predictions."""
from pathlib import Path
import sys
import numpy as np
from model import Data,read,metrics
from study import ROOT,NAMES,key,objective,PLANS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def run():
    complete=read(ROOT/'complete.json');d=Data();tables=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)]
    selections=[read(ROOT/str(o)/'selection.json')for o in range(6)];names=list(selections[0]['choices']);pooled=[]
    for ci in range(len(tables[0])):
        c=tables[0][ci];m={k:float(np.mean([t[ci]['metrics'][k]for t in tables]))for k in ['macroF1','macroRecall','macroPrecision','microF1','sampleF1','exactMatch']}
        pooled.append(dict(configId=ci,config=c['config'],metrics=m))
    final={}
    for name in names:
        arm,criterion=name.split('/');rs=pooled if arm=='all'else[r for r in pooled if r['config']['arm']==arm]if arm in NAMES else[r for r in pooled if r['config']['arm']in ['hard_time','branch_time','actions']]
        final[name]=max(rs,key=lambda r:key(r,criterion))
    write(ROOT/'final-inner-selection.json',dict(usesOuterTruth=False,choices=final))
    baseline=np.array([[c in d.records[i]['legacyOverall']['tags']for c in d.classes]for i in d.main]);base=metrics(d.y[d.main],baseline,d.weight[d.main]);results={'shipped_v122':dict(metrics=base,predictions=baseline.tolist())}
    for name in names+['legacy_retuned']:
        lookup={}
        for o in range(6):
            for r in read(ROOT/str(o)/'outer-predictions.json')[name]:
                if r['uid']in lookup:raise ValueError('Duplicate OOF chart')
                lookup[r['uid']]=r
        if set(lookup)!=set(d.manifest['mainUids']):raise ValueError('Incomplete OOF chart inventory')
        pred=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main]);m=metrics(d.y[d.main],pred,d.weight[d.main])
        errors={c:{label:[d.records[d.main[i]]['uid']for i in np.flatnonzero(mask)]for label,mask in [('fp',(d.y[d.main,j]==0)&pred[:,j]),('fn',(d.y[d.main,j]==1)&(~pred[:,j]))]}for j,c in enumerate(d.classes)}
        results[name]=dict(metrics=m,predictions=pred.tolist(),errors=errors,
            correctedRecordSets=int(np.sum(np.any(baseline!=d.y[d.main],axis=1)&np.all(pred==d.y[d.main],axis=1))),
            regressedRecordSets=int(np.sum(np.all(baseline==d.y[d.main],axis=1)&np.any(pred!=d.y[d.main],axis=1))))
    # Conditional profile: reselect all other coordinates and compare actual
    # held-out INNER prediction vectors, not just equal scalar F1 values.
    profiles=[]
    for axis in ['weightId','power','ridge','balance']:
        for value in sorted({r['config'][axis]for r in pooled}):
            rows=[r for r in pooled if r['config'][axis]==value];chosen=max(rows,key=key);ci=chosen['configId'];ref=final['all/balanced']['configId'];changed=[]
            for o in range(6):
                with np.load(ROOT/str(o)/'inner-vectors.npz')as z:
                    ids=z['validationIds'];q=z['predictions'][ci];p=z['predictions'][ref];take=np.any(q!=p,axis=1)
                    changed.append(float(d.weight[ids][take].sum()/d.weight[ids].sum()))
            profiles.append(dict(axis=axis,value=value,choice=chosen,meanChangedPhysicalMass=float(np.mean(changed)),objectiveDeficit=objective(final['all/balanced']['metrics'])-objective(chosen['metrics'])))
    write(ROOT/'conditional-profiles.json',profiles)
    write(ROOT/'analysis.json',dict(complete=complete,results=results,classes=d.classes,adopted=None,finalSelection=final))
    lines=['# 连续感知负担聚合：冻结局部的配对结果','','同一349标注、源分组嵌套选择；回顾性结果。百分数不是正确概率。','',
        '| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        if name.endswith('/macro'):continue
        m=r['metrics'];lines.append('|'+name+'|'+'|'.join(f'{100*m[k]:.2f}%'for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch'])+'|')
    lines+=['','局部头冻结，本阶段没有局部分类准确率增益。旧聚合重调与所有新表达分别联调；组件与评分为冻结外部先验。新裁剪整谱／构造对照没有用于拟合或选择。','',
        '共同部署候选（仅INNER均值选择）：', '```json',__import__('json').dumps(final['all/balanced'],ensure_ascii=False,indent=2),'```','',
        '待决策：受约束整谱反传对照、关键玩家对照、百分数时间／倍率契约及静态消费者。此报告本身不代表已部署。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print('\n'.join(lines[:22]),flush=True)


if __name__=='__main__':run()
