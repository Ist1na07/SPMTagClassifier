"""Read the categorical-stage experiment and semantic deployment subsets."""
from pathlib import Path
import sys,json
import numpy as np
from model import Data,read,metrics
sys.path.insert(0,str(Path(__file__).resolve().parent))
from study import objective,key,PLANS,NAMES
from hierarchy import fit,probabilities,decode
ROOT=Path('research/perception/results/hierarchy-r2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def eligible(d,row,name):
    c=row['config'];w=d.manifest['weights'][c['weightId']]
    if name=='public_rating':return c['arm']=='combined'
    if name=='component_rating':return c['components']and w['alpha']>0 and w['beta']>0
    if name=='physical_only':return c['arm']=='actions'
    return True


def run():
    read(ROOT/'complete.json');d=Data();tables=[read(ROOT/str(o)/'inner-grid.json')for o in range(6)];sels=[read(ROOT/str(o)/'selection.json')for o in range(6)];names=list(sels[0]['choices']);pooled=[]
    protocol=dict(semanticSubsets=dict(public_rating='soft branches,alpha>0,beta>0,no private components',component_rating='soft branches,alpha>0,beta>0,conditional component features',physical_only='soft branches,alpha0,beta>0,no rating or components'),
        reason='User percentage must combine confidence,difficulty,actions when rating exists; hard-time winners are controls rather than the requested product. No-rating profile must not consume missing rating as zero. Restriction precedes consuming this study OUTER predictions.',
        selection='Same balanced/macro INNER criteria, same original folds; no new threshold or hyperparameters. Supplemental eligible per-outer models fit original TRAIN only after INNER selection.',
        sourceSnapshot=Path(__file__).read_text(encoding='utf8'))
    p=ROOT/'consumer-protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Frozen hierarchy consumer changed')
    if not p.exists():write(p,protocol)
    for ci in range(len(tables[0])):
        r=tables[0][ci];m={k:float(np.mean([t[ci]['metrics'][k]for t in tables]))for k in ['macroF1','macroRecall','macroPrecision','microF1','sampleF1','exactMatch']};pooled.append(dict(configId=ci,fitId=r['fitId'],config=r['config'],metrics=m))
    final={}
    for name in names:
        arm,criterion=name.split('/');indices={r['configId']for r in tables[0]if (arm=='all')or(arm in NAMES and r['config']['arm']==arm)or(arm=='no_rating'and r['config']['arm']in ['hard_time','branch_time','actions'])or(arm in ['conditional','nonconditional']and r['config']['conditional']==(arm=='conditional'))or(arm in ['no_fallback','with_fallback']and r['config']['fallback']==(arm=='with_fallback'))}
        final[name]=max((r for r in pooled if r['configId']in indices),key=lambda r:key(r,criterion))
    supplemental={}
    for name in ['public_rating','component_rating','physical_only']:
        rs=[r for r in pooled if eligible(d,r,name)];final[name]=max(rs,key=key)
        supplemental[name]=[max([r for r in table if eligible(d,r,name)],key=key)for table in tables]
    write(ROOT/'final-inner-selection.json',dict(usesOuterTruth=False,choices=final));write(ROOT/'semantic-inner-selection.json',dict(usesOuterTruth=False,choices=supplemental))
    # No OUTER table was read before freezing all selections above.
    baseline=np.array([[c in d.records[i]['legacyOverall']['tags']for c in d.classes]for i in d.main]);results={'shipped_v122':dict(metrics=metrics(d.y[d.main],baseline,d.weight[d.main]),predictions=baseline.tolist())};supp_models={}
    def consume(name,lookup):
        if set(lookup)!=set(d.manifest['mainUids']):raise ValueError('Whole OOF inventory mismatch')
        q=np.array([[c in lookup[d.records[i]['uid']]['predicted']for c in d.classes]for i in d.main]);m=metrics(d.y[d.main],q,d.weight[d.main])
        return dict(metrics=m,predictions=q.tolist(),correctedRecordSets=int(np.sum(np.any(baseline!=d.y[d.main],axis=1)&np.all(q==d.y[d.main],axis=1))),regressedRecordSets=int(np.sum(np.all(baseline==d.y[d.main],axis=1)&np.any(q!=d.y[d.main],axis=1))),
            errors={c:{label:[d.records[d.main[i]]['uid']for i in np.flatnonzero(mask)]for label,mask in [('fp',(d.y[d.main,j]==0)&q[:,j]),('fn',(d.y[d.main,j]==1)&(~q[:,j]))]}for j,c in enumerate(d.classes)})
    for name in names:
        lookup={r['uid']:r for o in range(6)for r in read(ROOT/str(o)/'outer-predictions.json')[name]};results[name]=consume(name,lookup)
    for name,choices in supplemental.items():
        lookup={};models=[];queries=[]
        for o,row in enumerate(choices):
            c=row['config'];raw,av,share,family=d.representation(c['weightId'],c['power'],c['hard'],c['components']);m,t=fit(d,raw,av,share,family,d.main[PLANS[o]['train']],c);fine,slot=probabilities(d,raw,av,share,family,m);q=decode(d,fine,slot,c['fallback']);models.append(dict(outer=o,model=m,receipt=t))
            lookup.update({d.records[i]['uid']:dict(predicted=[c for c,v in zip(d.classes,q[i])if v])for i in d.main[PLANS[o]['test']]})
            groups={d.records[i]['group']for i in d.main[PLANS[o]['train']]}
            queries.extend(dict(outer=o,uid=r['uid'],heldOutSourceGroup=r['group']not in groups,predicted=[c for c,v in zip(d.classes,q[i])if v],percentages=dict(zip(d.fine,(100*share[i]).tolist())))for i,r in enumerate(d.records)if r['dataset']in ['query','derived'])
        supp_models[name]=models;write(ROOT/(name+'-queries.json'),queries);results[name]=consume(name,lookup)
    write(ROOT/'semantic-outer-models.json',supp_models)
    profiles=[];reference=final['component_rating'];refid=reference['configId']
    for axis in ['weightId','power','ridge','balance','conditional','fallback']:
        for value in sorted({r['config'][axis]for r in pooled}):
            rows=[r for r in pooled if eligible(d,r,'component_rating')and r['config'][axis]==value]
            if not rows:continue
            c=max(rows,key=key);changes=[]
            for o in range(6):
                with np.load(ROOT/str(o)/'inner-vectors.npz')as z:
                    ids=z['validationIds'];p=z['predictions'][refid];q=z['predictions'][c['configId']];changes.append(float(d.weight[ids][np.any(p!=q,axis=1)].sum()/d.weight[ids].sum()))
            profiles.append(dict(axis=axis,value=value,choice=c,objectiveDeficit=objective(reference['metrics'])-objective(c['metrics']),meanChangedPhysicalMass=float(np.mean(changes))))
    write(ROOT/'conditional-profiles.json',profiles);write(ROOT/'analysis.json',dict(results=results,classes=d.classes,finalSelection=final,adopted=None))
    lines=['# 分离整谱Mix与局部成分：配对结果','','同一349条整谱、原歌曲分组、回顾性嵌套比较；局部头仍冻结。','',
        '| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |','|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        if name.endswith('/macro'):continue
        m=r['metrics'];lines.append('|'+name+'|'+'|'.join(f'{100*m[k]:.2f}%'for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch'])+'|')
    lines+=['','共同component_rating候选（按INNER选择，不是OUTER最优）：','```json',json.dumps(final['component_rating'],ensure_ascii=False,indent=2),'```','',
        '完整保留不可支持目标为误判；拟合时其截断NLL常数与零梯度分离。详细数值与错误身份见analysis.json。物理／公开评分／组件三个消费入口分别调参，缺失组件不补零。反传与实际消费者验收仍待完成。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines[:28]),flush=True)


if __name__=='__main__':run()
