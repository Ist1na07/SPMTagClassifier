"""Common selection, exact selector checks and complete extended-grid report."""
from pathlib import Path
import sys,argparse,json
import numpy as np
from model import read,metrics
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parent))
from extended import ExtendedData
from compact import CompactData
from anchored import base,predictions
from anchored_extended_study import FINE,FAMILY
from fast_select import select
from threshold_study import KEYS,reps
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def run(model):
    root=Path('research/perception/results')/('anchored-r2'if model=='detail'else'anchored-compact-r2');read(root/'complete.json');d=ExtendedData()if model=='detail'else CompactData();b=base(d);tables=[];refs=[];checked=[]
    for o in range(6):
        with np.load(root/str(o)/'inner-scores.npz')as z:table=z['scores'].copy()
        s=read(root/str(o)/'selection.json');ref=np.array([s['baseline'][k]for k in KEYS]);tables.append(table);refs.append(ref)
        for name in ['all/protected','combined/protected','actions/protected','all/balanced','all/macro']:
            arm,criterion=name.split('/');c=select(d,table,ref,arm,criterion);old=s['choices'][name]
            if tuple(c[k]for k in ['representationId','familyId','policyId'])!=tuple(old[k]for k in ['representationId','familyId','policyId']):raise ValueError('Selector algebra changed a saved choice')
            checked.append(dict(outer=o,name=name,sameChoice=True))
    mean=np.mean(tables,axis=0);reference=np.mean(refs,axis=0);final={}
    for name in ['all','hard_time','branch_time','difficulty','actions','combined']:
        for criterion in ['protected','balanced','macro']:final[name+'/'+criterion]=select(d,mean,reference,name,criterion)
    for flag in [False,True]:final['hybrid_'+str(flag)]=select(d,mean,reference,'all','protected',flag)
    write(root/'final-inner-selection.json',dict(usesOuterTruth=False,choices=final));write(root/'selector-equivalence.json',checked)
    # All common choices frozen BEFORE consuming this study's OUTER outputs.
    result={'shipped_v122':dict(metrics=metrics(d.y[d.main],b[d.main],d.weight[d.main]),predictions=b[d.main].tolist())}
    for name in final:
        lookup={r['uid']:r for o in range(6)for r in read(root/str(o)/'outer-predictions.json')[name]}
        if set(lookup)!=set(d.manifest['mainUids']):raise ValueError('Incomplete OOF identities')
        q=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main]);m=metrics(d.y[d.main],q,d.weight[d.main]);m['perClass']=dict(zip(d.classes,m['perClass']))
        result[name]=dict(metrics=m,predictions=q.tolist(),correctedRecordSets=int(np.sum(np.any(b[d.main]!=d.y[d.main],axis=1)&np.all(q==d.y[d.main],axis=1))),regressedRecordSets=int(np.sum(np.all(b[d.main]==d.y[d.main],axis=1)&np.any(q!=d.y[d.main],axis=1))),
            errors={c:{label:[d.records[d.main[i]]['uid']for i in np.flatnonzero(mask)]for label,mask in [('fp',(d.y[d.main,j]==0)&q[:,j]),('fn',(d.y[d.main,j]==1)&(~q[:,j]))]}for j,c in enumerate(d.classes)})
    write(root/'analysis.json',dict(results=result,finalSelection=final,classes=d.classes,adopted=None))
    # Profiles condition on active coordinates, reselect ALL others, and replay
    # actual prediction vectors and the full-cohort percentages.
    ref=final['combined/protected'];profiles=[]
    def vector(c):
        r=c['representation'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);return predictions(d,share,family,b,[c['policy']],[c['family']])[0,0],share
    rq,rs=vector(ref)
    axes=[('alpha',[1.,2.,3.,4.]),('beta',[.5,1.,1.5,2.]),('width',[500.,1000.,2000.]),('hold',[0.,1.]),('power',[1.,2.]),
        ('keepFloor',[0.,.03,.07,.15,.25,.35,.5]),('addFloor',[.25,.35,.4,.5,.65,.8,.95,1.01]),('mixOverride',[.35,.5,.65,.8,.95,1.01]),('minRc',[.03,.05,.07,.1,.18,.2,.3,.32,.4]),('minLn',[.03,.05,.07,.1,.18,.2,.3,.32,.4])]
    rr=reps(d)
    for axis,values in axes:
        for value in values:
            rm=np.array([not r['hard']and d.manifest['weights'][r['weightId']]['alpha']>0 and d.manifest['weights'][r['weightId']]['beta']>0 and (d.manifest['weights'][r['weightId']].get(axis,r.get(axis))==value if axis in ['alpha','beta','width','hold','power']else True)for r in rr])
            fm=np.array([p['enabled']and p.get(axis)==value if axis in ['minRc','minLn']else True for p in FAMILY]);pm=np.array([p.get(axis)==value if axis in ['keepFloor','addFloor','mixOverride']else True for p in FINE]);mask=rm[:,None,None]&fm[None,:,None]&pm[None,None,:]
            score=np.where(mask[:,:,:,None],mean,-np.inf);c=select(d,score,reference,'combined','protected')
            if c is None:profiles.append(dict(axis=axis,value=value,feasible=False));continue
            q,share=vector(c);changes=[]
            for o in range(6):
                ids=d.main[PLANS[o]['train']];m=metrics(d.y[ids],q[ids],d.weight[ids]);expected=tables[o][c['representationId'],c['familyId'],c['policyId']];np.testing.assert_allclose([m[k]for k in KEYS],expected,atol=1e-12)
                changes.append(float(d.weight[ids][np.any(q[ids]!=rq[ids],axis=1)].sum()/d.weight[ids].sum()))
            profiles.append(dict(axis=axis,value=value,feasible=True,choice=c,meanChangedPhysicalMass=float(np.mean(changes)),maxPercentagePointChange=float(np.max(np.abs(share[d.main]-rs[d.main]))*100)))
    write(root/'conditional-profiles.json',profiles)
    replay={};queries={}
    for name in ['actions/protected','combined/protected']:
        q,share=vector(final[name]);replay[name]=metrics(d.y[d.main],q[d.main],d.weight[d.main]);queries[name]=[dict(uid=r['uid'],predicted=[c for c,v in zip(d.classes,q[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
    write(root/'final-common-replay.json',dict(scope='Full-study deterministic common-profile replay, NOT a fresh or OOF estimate',metrics=replay,queries=queries))
    lines=[f'# {model}：扩大活跃边界后的保守百分数聚合','','同一原349条／源歌曲组。每外折全新规则参数联合选择；局部专家未改变。','',
        '| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in result.items():
        if name.endswith('/macro')or name.endswith('/balanced'):continue
        m=r['metrics'];lines.append('|'+name+'|'+'|'.join(f'{100*m[k]:.2f}%'for k in KEYS)+'|')
    lines+=['','仅INNER选择的共同评分配置：','```json',json.dumps(ref,ensure_ascii=False,indent=2),'```','',
        '扩大alpha/beta、时间窗与门控边界是同一模型的完整笛卡尔比较。近最优0.1个百分点内优先线性物理解释和较少修正，是明确先验，不是把平地伪装成数据井。条件剖面记录实际标签向量和百分数的变化；未通过保护的固定坐标明确标为不可行。','',
        '常规输入不需再次评分；公开评分输入复用同次SPMRating原始曲线。局部反传是独立对照，不能把此处整谱增益算成局部分段准确率提高。所有结果是同批反复研究资料上的回顾性检验。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[:24]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['detail','compact'],default='detail');run(p.parse_args().model)
