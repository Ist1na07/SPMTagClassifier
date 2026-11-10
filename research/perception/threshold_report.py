"""Freeze direct-percent INNER choices, then evaluate full held-out inventories."""
from pathlib import Path
import sys,json
import numpy as np
from model import Data,read,metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parent))
from threshold_study import ROOT,choose,reps,KEYS
from threshold import POLICIES,SHRINK,train_thresholds,predict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def run():
    read(ROOT/'complete.json');d=Data();scores=[]
    for o in range(6):
        with np.load(ROOT/str(o)/'inner-scores.npz')as z:scores.append(z['scores'].copy())
    mean=np.mean(scores,axis=0);final={}
    for name in ['all','hard_time','branch_time','difficulty','actions','combined']:
        for criterion in ['balanced','macro']:final[name+'/'+criterion]=choose(d,mean,name,criterion)
    for flag in [False,True]:final['hybrid_'+str(flag)]=choose(d,mean,'all','balanced',flag)
    write(ROOT/'final-inner-selection.json',dict(usesOuterTruth=False,choices=final))
    # No OUTER file read before the shared configuration freeze above.
    baseline=np.array([[c in d.records[i]['legacyOverall']['tags']for c in d.classes]for i in d.main]);results={'shipped_v122':dict(metrics=metrics(d.y[d.main],baseline,d.weight[d.main]),predictions=baseline.tolist())}
    for name in final:
        lookup={r['uid']:r for o in range(6)for r in read(ROOT/str(o)/'outer-predictions.json')[name]}
        if set(lookup)!=set(d.manifest['mainUids']):raise ValueError('Incomplete OOF inventory')
        q=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main]);m=metrics(d.y[d.main],q,d.weight[d.main])
        results[name]=dict(metrics=m,predictions=q.tolist(),correctedRecordSets=int(np.sum(np.any(baseline!=d.y[d.main],axis=1)&np.all(q==d.y[d.main],axis=1))),regressedRecordSets=int(np.sum(np.all(baseline==d.y[d.main],axis=1)&np.any(q!=d.y[d.main],axis=1))),
            errors={c:{label:[d.records[d.main[i]]['uid']for i in np.flatnonzero(mask)]for label,mask in [('fp',(d.y[d.main,j]==0)&q[:,j]),('fn',(d.y[d.main,j]==1)&(~q[:,j]))]}for j,c in enumerate(d.classes)})
    write(ROOT/'analysis.json',dict(results=results,finalSelection=final,classes=d.classes,adopted=None))
    # Actual held-out functions on a few informative conditional coordinates.
    # No exponential extra fits: each chosen threshold partition is exact.
    candidates=[];ref=final['combined/balanced']
    for axis,values in [('alpha',[1.,2.]),('beta',[.5,1.]),('power',[1.,2.]),('strength',[0.,.001,.01,.1]),('reference',[.1,.25,.4])]:
        for value in values:
            a=mean.copy();allowed_rep=np.array([not r['hard']and d.manifest['weights'][r['weightId']]['alpha']>0 and d.manifest['weights'][r['weightId']]['beta']>0 and (d.manifest['weights'][r['weightId']].get(axis,r.get(axis))==value if axis in ['alpha','beta','power']else True)for r in reps(d)])
            allowed_shrink=np.array([h.get(axis)==value if axis in ['strength','reference']else True for h in SHRINK]);mask=allowed_rep[:,None,None]&allowed_shrink[None,None,:]
            a=np.where(mask[:,:,:,None],a,-np.inf);c=choose(d,a,'combined','balanced');candidates.append(dict(axis=axis,value=value,choice=c))
    def replay(c,o):
        r=c['representation'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);preds=[];ids=[];intervals=[]
        for s in PLANS[o]['inner']:
            tau,iv,_=train_thresholds(d,share,family,d.main[s['train']],[c['policy']],[c['shrink']]);v=d.main[s['validation']];preds.append(predict(d,share[v],family[v],tau,[c['policy']])[0,0]);ids.extend(v);intervals.append(iv[0,0])
        q=np.concatenate(preds);m=metrics(d.y[ids],q,d.weight[ids]);expected=scores[o][c['representationId'],c['policyId'],c['shrinkId']]
        np.testing.assert_allclose([m[k]for k in KEYS],expected,atol=1e-12)
        return q,np.array(ids),intervals,share
    refs=[replay(ref,o)for o in range(6)];profiles=[]
    for row in candidates:
        changes=[];percentage=[]
        for o in range(6):
            q,ids,ivs,share=replay(row['choice'],o);p,other,_,oldshare=refs[o];np.testing.assert_array_equal(ids,other);changes.append(float(d.weight[ids][np.any(q!=p,axis=1)].sum()/d.weight[ids].sum()));percentage.append(float(np.max(np.abs(share[ids]-oldshare[ids]))*100))
        profiles.append(dict(row,meanChangedPhysicalMass=float(np.mean(changes)),maxPercentagePointChange=max(percentage)))
    write(ROOT/'conditional-profiles.json',profiles)
    lines=['# 直接用感知百分数门控整谱tag','','原349条标注、6×3歌曲分组，全部新权重／门控／收缩参数联合选择。','',
        '| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        if name.endswith('/macro'):continue
        m=r['metrics'];lines.append('|'+name+'|'+'|'.join(f'{100*m[k]:.2f}%'for k in KEYS)+'|')
    lines+=['','仅INNER选择的共同评分候选：','```json',json.dumps(ref,ensure_ascii=False,indent=2),'```','',
        '每个TRAIN门限对应一个实际分区区间。reference即使在strength=0也会选取平地区间中的不同代表，故可能改变未见谱预测；没有把正则化选出的点叫作数据识别的唯一井。条件剖面同时记录标签向量与百分数变化。','',
        '局部支持在此对照中冻结；这不是局部分类改进。SPM组件与局部反传由单独匹配实验检验。全部指标均为同批反复研究资料上的回顾性对照。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[:22]),flush=True)


if __name__=='__main__':run()
