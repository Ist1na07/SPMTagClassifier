"""Select partitions from INNER only, then consume held-out predictions."""
import itertools
from pathlib import Path
import numpy as np
import bilateral_partition_study as study
from bilateral_partition_model import Packed, frames, segment_accuracy, read
from rc_action_runtime import sufficient, aggregate, policy_predictions
from rc_action_report import unpack
from landscape_audit import multilabel_metrics
from whole_reference import write
from cut_consistency_model import mil

ROOT = study.ROOT


def choose(candidates, policy, reference):
    eligible = [r for r in candidates if r['headId']==0 or r['config']['conceptWeight']>0]
    if policy=='overlay_guarded':
        eligible = [r for r in eligible if r['macroF1']>=reference-1e-12]
        key=lambda h:(h['segmentAccuracy'],h['macroF1'],h['macroRecall'],h['headId']==0,-abs(h['partition']-4),-h['partition'],-h['headId'],-h['policyId'])
    elif policy=='macro':
        key=lambda h:(h['macroF1'],h['macroRecall'],h['segmentAccuracy'],h['headId']==0,-abs(h['partition']-4),-h['partition'],-h['headId'],-h['policyId'])
    elif policy=='frozen':
        eligible=[r for r in candidates if r['headId']==0]
        key=lambda h:(h['macroF1'],h['macroRecall'],h['segmentAccuracy'],-abs(h['partition']-4),-h['partition'],-h['policyId'])
    else:raise ValueError('Unknown declared selection')
    return max(eligible,key=key)


def run():
    read(ROOT/'complete.json');protocol=read(ROOT/'protocol.json')
    all_whole=[];all_local=[];all_concepts=[];choices=[];fit_cache={};new_fits=[]
    # All decisions use inner scores and manual inner annotations ONLY.
    for variant,kind,outer in itertools.product(['original','physical'],['bias','hand','bilateral','both'],range(6)):
        candidates=[]
        for partition in range(9):
            folder=ROOT/variant/str(partition)/kind/str(outer)
            candidates.extend(dict(r,partition=partition)for r in read(folder/'search.json'))
        reference=max(r['macroF1']for r in candidates if r['headId']==0)
        for policy in ['frozen','macro','overlay_guarded']:
            choice=choose(candidates,policy,reference)
            choices.append(dict(variant=variant,kind=kind,outer=outer,selection=policy,choice=choice,reference=reference,usesOuterTruth=False))
    write(ROOT/'joint-inner-selection.json',choices)
    for selected in choices:
        variant,kind,outer,policy=[selected[k]for k in ['variant','kind','outer','selection']]
        c=selected['choice'];partition=c['partition'];head=c['headId'];plan=study.PLANS[outer]
        study._VARIANT,study._PARTITION,study._OUTER=variant,partition,outer
        study.prepare_world(variant,partition);world=study.base._WORLD
        key=variant,partition,kind,outer,head
        if key not in fit_cache:
            saved=read(ROOT/variant/str(partition)/kind/str(outer)/'outer-models.json')
            if str(head)in saved:model=saved[str(head)]['model']
            elif head==0:model=None
            else:
                packed=Packed(world,[world.main[i]for i in plan['train']],[world.manual[i]for i in plan['strongTrain']],kind=kind)
                model,trace=study.fit(packed,c['config']);new_fits.append(dict(key=list(key),model=model,fit=trace))
            fit_cache[key]=model
        model=fit_cache[key];name=variant+'/'+kind+'/'+policy
        for i in plan['test']:
            record=world.main[i];result=aggregate(sufficient(frames(world,record['uid'],model)),c['policy'])
            all_whole.append(dict(name=name,uid=record['uid'],outer=outer,predicted=result['tags']))
        for i,accuracy in zip(plan['strongTest'],segment_accuracy(world,[world.manual[i]for i in plan['strongTest']],model)):
            all_local.append(dict(name=name,uid=world.manual[i]['uid'],outer=outer,accuracy=accuracy,
                frames=frames(world,world.manual[i]['uid'],model)))
        for uid in ['concept/H01','concept/H02','concept/H03','concept/H04','local/F01','local/F02']:
            all_concepts.append(dict(name=name,uid=uid,outer=outer,frames=frames(world,uid,model)))
    write(ROOT/'selected-outer-models.json',[dict(key=list(k),model=v)for k,v in fit_cache.items()])
    write(ROOT/'additional-selected-fits.json',new_fits)
    write(ROOT/'joint-outer-predictions.json',all_whole);write(ROOT/'joint-outer-local.json',all_local)
    write(ROOT/'joint-concepts.json',all_concepts)
    study.prepare_world('original',4);world=study.base._WORLD
    uids=[r['uid']for r in world.main];y=mil.encode(world.main);weights=np.array([r['recordWeight']for r in world.main]);groups=[r['group']for r in world.main]
    # Actual shipped fixed classifier, not the independently retuned baseline.
    original=[r for r in read('research/rework/results/aggregation/predictions.json')if r['variant']=='original_fixed']
    lookup={r['uid']:r for r in original}
    if len(lookup)!=len(original)or set(lookup)!=set(uids):raise ValueError('Fixed baseline inventory changed')
    baseline=np.array([[c in lookup[uid]['predicted']for c in mil.MAP_CLASSES]for uid in uids],int)
    fixed=multilabel_metrics(y,baseline,weights,mil.MAP_CLASSES,groups)
    fixed['localAccuracy']=float(np.mean(segment_accuracy(world,world.manual,None)))
    fixed['techToDenseMs']=sum(max(0,min(s['end'],a['end'])-max(s['start'],a['start']))
        for r in world.manual for a in r['annotations']if 'tech'in a['tags']
        for s in frames(world,r['uid'],None)if s['tag']=='dense chordstream')
    metrics={'original_fixed':fixed}
    for name in dict.fromkeys(r['name']for r in all_whole):
        rows=[r for r in all_whole if r['name']==name];lookup={r['uid']:r for r in rows}
        if len(lookup)!=len(rows)or set(lookup)!=set(uids):raise ValueError('Incomplete joint outer inventory')
        p=np.array([[c in lookup[uid]['predicted']for c in mil.MAP_CLASSES]for uid in uids],int)
        m=multilabel_metrics(y,p,weights,mil.MAP_CLASSES,groups)
        local=[r for r in all_local if r['name']==name]
        if len(local)!=6:raise ValueError('Incomplete joint local corpus')
        m['localAccuracy']=float(np.mean([r['accuracy']for r in local]))
        m['perClassErrors']={c:{label:[uids[i]for i in np.flatnonzero(mask)]for label,mask in
            [('FP',(y[:,j]==0)&(p[:,j]==1)),('FN',(y[:,j]==1)&(p[:,j]==0))]}for j,c in enumerate(mil.MAP_CLASSES)}
        m['techToDenseMs']=sum(max(0,min(s['end'],a['end'])-max(s['start'],a['start']))
            for r in local for a in world.records[r['uid']]['annotations']if 'tech'in a['tags']
            for s in r['frames']if s['tag']=='dense chordstream')
        metrics[name]=m
    # Active-parameter profiles reselect partition and every other coordinate.
    # This is not evidence that a frozen policy identifies unused parameters.
    profiles=[]
    for variant,kind,outer in itertools.product(['original','physical'],['bias','hand','bilateral','both'],range(6)):
        candidates=[];arrays={};by_partition={}
        for partition in range(9):
            folder=ROOT/variant/str(partition)/kind/str(outer)
            by_partition[partition]=[dict(r,partition=partition)for r in read(folder/'search.json')]
            candidates.extend(by_partition[partition])
            with np.load(folder/'inner-bases.npz')as z:arrays[partition]={k:z[k]for k in z.files}
        reference=max(r['macroF1']for r in candidates if r['headId']==0)
        choice=choose(candidates,'overlay_guarded',reference);z=arrays[choice['partition']]
        target=policy_predictions(unpack(z,choice['headId']),[choice['policy']])[0];w=weights[z['validationIds']]
        curves={}
        for coordinate in ['localL2','strongWeight','conceptWeight','effortPower','primaryPower','partition']:
            values=range(9)if coordinate=='partition'else dict.fromkeys(c[coordinate]for c in study.CONFIGS)
            points=[]
            for setting in values:
                eligible=[r for r in candidates if r['headId']and r['config']['conceptWeight']>0 and r['macroF1']>=reference-1e-12 and
                    (r['partition']if coordinate=='partition'else r['config'][coordinate])==setting]
                if not eligible:points.append(dict(value=setting,eligible=False));continue
                best=choose(eligible,'overlay_guarded',reference);b=arrays[best['partition']]
                if not np.array_equal(b['validationIds'],z['validationIds']):raise ValueError('Partition OOF song scope changed')
                p=policy_predictions(unpack(b,best['headId']),[best['policy']])[0]
                points.append(dict(value=setting,eligible=True,selected=best,changedTagSetMass=float(w@np.any(p!=target,axis=1)/w.sum())))
            curves[coordinate]=points
        for coordinate in ['dominantShare','mixShare','secondRel']:
            points=[]
            for setting in dict.fromkeys(p[coordinate]for p in study.base.POST):
                policy_ids=[i for i,p in enumerate(study.base.POST)if p[coordinate]==setting]
                eligible=[]
                for partition,heads in by_partition.items():
                    b=arrays[partition]
                    for h in heads:
                        if h['headId']and h['config']['conceptWeight']<=0:continue
                        valid=[pid for pid in policy_ids if b['macroF1'][h['headId'],pid]>=reference-1e-12]
                        if not valid:continue
                        pid=max(valid,key=lambda i:(b['macroF1'][h['headId'],i],b['macroRecall'][h['headId'],i],-i))
                        eligible.append(dict(h,policyId=pid,policy=study.base.POST[pid],
                            macroF1=float(b['macroF1'][h['headId'],pid]),macroRecall=float(b['macroRecall'][h['headId'],pid])))
                if not eligible:points.append(dict(value=setting,eligible=False));continue
                best=choose(eligible,'overlay_guarded',reference);b=arrays[best['partition']]
                p=policy_predictions(unpack(b,best['headId']),[best['policy']])[0]
                points.append(dict(value=setting,eligible=True,selected=best,
                    changedTagSetMass=float(w@np.any(p!=target,axis=1)/w.sum())))
            curves[coordinate]=points
        profiles.append(dict(variant=variant,kind=kind,outer=outer,choice=choice,profiles=curves))
    write(ROOT/'analysis.json',dict(scope=protocol['scope'],metrics=metrics,choices=choices,parameterProfiles=profiles,
        adopted=False,nativeConsumerRequired=True,wholeProbabilityMetrics='Hard tag decisions only; no fabricated calibration/AP/Brier metrics.'))
    lines=['# 双手共击与时间分段配套对照','',
        '分段、局部头、聚合均从原外折TRAIN内部的三内折联合选择，不从外层成绩选分段。资料已被反复研究，结果按回顾性分组评价报告。','',
        '| 处理 | macro P | macro R | macro F1 | micro F1 | 集合全对 | 局部按谱准确率 | Tech→Dense错误时长ms |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,m in metrics.items():
        values=[m['macro']['precision'],m['macro']['recall'],m['macro']['f1'],m['micro']['f1'],m['exactMatch'],m['localAccuracy']]
        lines.append('| '+name+' | '+' | '.join(f'{100*v:.2f}%'for v in values)+f" | {m['techToDenseMs']:.1f} |")
    lines+=['','逐类误报／漏报谱身份、真实局部帧、机械对照、完整内层选择和条件参数剖面均已保存；这份报表不自动采用新模型。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
