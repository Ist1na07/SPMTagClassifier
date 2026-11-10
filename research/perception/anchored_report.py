"""Common INNER selection and held-out audit for the conservative readout."""
from pathlib import Path
import sys,json
import numpy as np
from model import Data,read,metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parent))
from anchored_study import ROOT,KEYS,choose
from anchored import base,predictions,FINE_POLICIES,FAMILY_POLICIES
from threshold_study import reps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def run():
    read(ROOT/'complete.json');d=Data();baseline=base(d);tables=[];references=[]
    for o in range(6):
        with np.load(ROOT/str(o)/'inner-scores.npz')as z:tables.append(z['scores'].copy())
        references.append(np.array([read(ROOT/str(o)/'selection.json')['baseline'][k]for k in KEYS]))
    mean=np.mean(tables,axis=0);reference=np.mean(references,axis=0);final={}
    for name in ['all','hard_time','branch_time','difficulty','actions','combined']:
        for criterion in ['protected','balanced','macro']:final[name+'/'+criterion]=choose(d,mean,reference,name,criterion)
    for flag in [False,True]:final['hybrid_'+str(flag)]=choose(d,mean,reference,'all','protected',flag)
    write(ROOT/'final-inner-selection.json',dict(usesOuterTruth=False,choices=final))
    # Common choices frozen above, before consuming OUTER truth.
    results={'shipped_v122':dict(metrics=metrics(d.y[d.main],baseline[d.main],d.weight[d.main]),predictions=baseline[d.main].tolist())}
    for name in final:
        lookup={r['uid']:r for o in range(6)for r in read(ROOT/str(o)/'outer-predictions.json')[name]}
        if set(lookup)!=set(d.manifest['mainUids']):raise ValueError('Incomplete OOF inventory')
        q=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main]);m=metrics(d.y[d.main],q,d.weight[d.main])
        results[name]=dict(metrics=m,predictions=q.tolist(),correctedRecordSets=int(np.sum(np.any(baseline[d.main]!=d.y[d.main],axis=1)&np.all(q==d.y[d.main],axis=1))),regressedRecordSets=int(np.sum(np.all(baseline[d.main]==d.y[d.main],axis=1)&np.any(q!=d.y[d.main],axis=1))),
            errors={c:{label:[d.records[d.main[i]]['uid']for i in np.flatnonzero(mask)]for label,mask in [('fp',(d.y[d.main,j]==0)&q[:,j]),('fn',(d.y[d.main,j]==1)&(~q[:,j]))]}for j,c in enumerate(d.classes)})
    write(ROOT/'analysis.json',dict(results=results,classes=d.classes,finalSelection=final,adopted=None))
    profiles=[];ref=final['combined/protected']
    def vector(c):
        r=c['representation'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);return predictions(d,share,family,baseline,[c['policy']],[c['family']])[0,0],share
    refq,refs=vector(ref)
    for axis,values in [('keepFloor',[0.,.03,.07,.15,.25]),('addFloor',[.5,.65,.8,.95,1.01]),('mixOverride',[.65,.8,.95,1.01]),('alpha',[1.,2.]),('beta',[.5,1.]),('width',[1000.,2000.]),('hold',[0.,1.]),('power',[1.,2.])]:
        for value in values:
            score=mean.copy();rep_mask=np.array([not r['hard']and d.manifest['weights'][r['weightId']]['alpha']>0 and d.manifest['weights'][r['weightId']]['beta']>0 and (d.manifest['weights'][r['weightId']].get(axis,r.get(axis))==value if axis in ['alpha','beta','width','hold','power']else True)for r in reps(d)])
            policy_mask=np.array([p.get(axis)==value if axis in ['keepFloor','addFloor','mixOverride']else True for p in FINE_POLICIES]);allowed=rep_mask[:,None,None]&policy_mask[None,None,:];score=np.where(allowed[:,:,:,None],score,-np.inf)
            c=choose(d,score,reference,'combined','protected');q,share=vector(c)
            changes=[]
            for o in range(6):
                ids=d.main[PLANS[o]['train']];actual=metrics(d.y[ids],q[ids],d.weight[ids]);expected=tables[o][c['representationId'],c['familyId'],c['policyId']];np.testing.assert_allclose([actual[k]for k in KEYS],expected,atol=1e-12)
                changes.append(float(d.weight[ids][np.any(q[ids]!=refq[ids],axis=1)].sum()/d.weight[ids].sum()))
            profiles.append(dict(axis=axis,value=value,choice=c,meanChangedPhysicalMass=float(np.mean(changes)),maxPercentagePointChange=float(np.max(np.abs(share[d.main]-refs[d.main]))*100)))
    write(ROOT/'conditional-profiles.json',profiles)
    # Clarify protocol prose without changing its frozen code or any outcomes.
    write(ROOT/'semantics-correction.json',dict(correction='The explicit strong-prototype mixOverride applies to ALL old composite slots, including Hybrid. A missing new family trigger ALONE does not remove an old Hybrid. The producer source always implemented this, so no predictions, selection, or metrics changed.',source='anchored.py predictions(), mixed=baseline[:,hybrid/rcmix/lnmix].any(1)'))
    lines=['# 保留成熟判断，用感知百分数修正','','原349条、原歌曲／物理源分组；新全局权重和门控完整联合选择。','',
        '| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        if name.endswith('/macro')or name.endswith('/balanced'):continue
        m=r['metrics'];lines.append('|'+name+'|'+'|'.join(f'{100*m[k]:.2f}%'for k in KEYS)+'|')
    lines+=['','共同评分候选（仅INNER）：','```json',json.dumps(ref,ensure_ascii=False,indent=2),'```','',
        '原有整谱输出是冻结先验，不是新的逐段标签。低占比剔除、高占比补充、强原型替换Mix、两大家族Hybrid判定都直接读取新积分百分数。全部关闭时精确回到已发布v1.2.2。','',
        '这批规则没有TRAIN数值拟合：每外折在其完整TRAIN上选全局参数，等价于汇总原3INNER验证库存；最终OUTER仍独立于本折选参。报告是重复使用原资料的回顾性检验。INNER保护不保证OUTER无退步。','',
        'mixOverride也可以替换旧Hybrid；没有新家族触发不单独取消旧Hybrid。此处澄清原协议一句过宽的描述，代码／结果没有改变。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[:23]),flush=True)


if __name__=='__main__':run()
