"""Nested capacity selection over complete fixed native-policy candidates.

No new expert fits or raw prediction computations. Each inner training subset
selects a whole pipeline; its validation subset selects the update budget.
"""
import argparse,gzip
from pathlib import Path
import json
from native_search import f1_counts,BASE,select_segments
from native_followup import load_source
from mil import load,encode,MAP_CLASSES,target,np

BUDGETS=[0,1,2,4,8,None]


def changes(config,source_parameters):
    c=config;p=source_parameters[c['source']]
    result={'metric':p['metricStrength']!=0,'release':p['releaseMs']is not None,
            'minLength':c['partition']//3!=1,'prominence':c['partition']%3!=1,
            'lnRatio':c['head']//2!=1,'scorer':c['head']%2!=0,'mode':c['mode']!='normalized'}
    for k in ['burdenGate','effortPower','confidencePower','dominantShare','mixShare','secondRel','twoTagMaxShare','minEvidence','minimumDiversity']:
        result[k]=c[k]!=BASE[k]
    for i in range(2):result['familyThreshold'+str(i)]=c['familyThresholds'][i]!=BASE['familyThresholds'][i]
    return [k for k,v in result.items()if v]


def candidate_metrics(predictions,accuracy,y,weights,ids,strong_ids):
    p=predictions[:,ids,:];truth=y[ids][None,:,:];w=weights[ids][None,:,None]
    macro=f1_counts((p*truth*w).sum(1),(p*(1-truth)*w).sum(1),((1-p)*truth*w).sum(1))
    segment=accuracy[:,strong_ids].mean(1)
    return macro,segment,.5*(macro+segment)


def pick(metric,eligible):
    macro,segment,joint=metric
    return max(eligible,key=lambda i:(joint[i],segment[i],macro[i],-i))


def nested_choice(predictions,accuracy,y,weights,plan,costs):
    minimum=int(costs.min());allowed=[np.flatnonzero(costs<=minimum+b)if b is not None else np.arange(len(costs))for b in BUDGETS]
    inner_map=np.zeros((len(BUDGETS),len(y),y.shape[1]),dtype=np.uint8)
    inner_accuracy=np.zeros((len(BUDGETS),accuracy.shape[1]));choices=[]
    for j,split in enumerate(plan['inner']):
        metric=candidate_metrics(predictions,accuracy,y,weights,split['train'],split['strongTrain'])
        for b,eligible in enumerate(allowed):
            chosen=int(pick(metric,eligible))
            inner_map[b,split['validation'],:]=predictions[chosen,split['validation'],:]
            inner_accuracy[b,split['strongValidation']]=accuracy[chosen,split['strongValidation']]
            choices.append({'inner':j,'budget':BUDGETS[b],'candidate':chosen,'changes':int(costs[chosen]),
                            'trainJoint':float(metric[2][chosen])})
    # Only predictions from the matching inner validation split enter this
    # second-level choice. Outer held-out columns have never been indexed.
    scores=candidate_metrics(inner_map,inner_accuracy,y,weights,plan['train'],plan['strongTrain'])
    at=max(range(len(BUDGETS)),key=lambda b:(scores[2][b],-b))
    final_metric=candidate_metrics(predictions,accuracy,y,weights,plan['train'],plan['strongTrain'])
    chosen=int(pick(final_metric,allowed[at]))
    return {'candidate':chosen,'budget':BUDGETS[at],'minimumRequiredChanges':minimum,'changes':int(costs[chosen]),
            'innerBudgetScores':[{'budget':b,'mapMacroF1':float(scores[0][i]),'segmentAccuracy':float(scores[1][i]),'joint':float(scores[2][i])}for i,b in enumerate(BUDGETS)],
            'innerPolicyChoices':choices},int(pick(final_metric,allowed[0])),int(pick(final_metric,allowed[-1]))


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def run(args):
    import native_metric_search as metric
    metric.configure()
    parent=Path(args.parent);root=Path(args.output);protocol=json.loads((parent/'protocol.json').read_text(encoding='utf8'))
    if not(parent/'complete.json').exists():raise RuntimeError('Parent complete ablations required before capacity study')
    main,manual,_=load();y=encode(main);weights=np.array([r['recordWeight']for r in main]);plans=protocol['plans']
    identity=[{'uid':r['uid'],'group':r['group'],'tags':target(r),'weight':r.get('recordWeight',1)}for r in main+manual]
    if protocol['records']!=identity:raise RuntimeError('Parent record order, labels or group weights changed')
    groups=list(protocol['groups']);output_protocol={'id':'native-nested-capacity-r1','parentProtocol':protocol,
       'scope':'retrospective nested policy-capacity selection; expert coefficients frozen, historical exposure unknown',
       'budgets':BUDGETS,'reference':'logical5+neutral+coverage, original default boundary/head, normalized aggregation, other BASE knobs',
       'selection':'inner training jointly selects all policy parameters per capacity; inner validation selects capacity; outer training reselects whole policy; no outer labels used',
       'groups':groups,'plans':plans,'sourceSnapshot':{'policy_capacity.py':Path(__file__).read_text(encoding='utf8')}}
    if(root/'protocol.json').exists()and json.loads((root/'protocol.json').read_text(encoding='utf8'))!=output_protocol:raise RuntimeError('Capacity protocol changed')
    write(root/'protocol.json',output_protocol);all_maps=[];all_segments=[];variants=[]
    for group in groups:
        configs=json.loads((parent/group/'candidates.json').read_text(encoding='utf8'))
        historical_choices=json.loads((parent/group/'selection.json').read_text(encoding='utf8'))['selections']
        with np.load(parent/group/'candidate-predictions.npz')as p:predictions=p['map'];accuracy=p['segmentAccuracy']
        costs=np.array([len(changes(c,protocol['sourceParameters']))for c in configs]);selections=[]
        for o,plan in enumerate(plans):
            choice,minimal,unrestricted=nested_choice(predictions,accuracy,y,weights,plan,costs)
            assert unrestricted==historical_choices[o]['candidate'],'Unrestricted control must exactly reproduce the parent final-task selection'
            selected={'outer':o,**choice,'config':configs[choice['candidate']]};selections.append(selected)
            write(root/group/'selection.json',{'selectionUsesOuterLabels':False,'selections':selections})
            cases=[(group,choice['candidate'])]
            if group=='metric_repaired':cases.extend([('capacity_minimum_control',minimal),('capacity_unrestricted_control',unrestricted)])
            for name,i in cases:
                if name not in variants:variants.append(name)
                c=configs[i]
                for k in plan['test']:
                    r=main[k];all_maps.append({'variant':name,'outer':o,'uid':r['uid'],'group':r['group'],'truth':target(r),'weight':r['recordWeight'],
                      'predicted':[label for label,b in zip(MAP_CLASSES,predictions[i,k])if b]})
                for k in plan['strongTest']:
                    r=manual[k];value=load_source(c['source'],[r])[0]
                    all_segments.extend({'variant':name,'outer':o,'uid':r['uid'],**s}for s in select_segments(value,c))
            print('capacity',group,o,'budget',choice['budget'],'changes',choice['changes'],flush=True)
        write(root/group/'costs.json',{'costs':costs.tolist(),'meaning':'number of changed mechanism coordinates; per-ablation minimum subtracted for budget'})
    write(root/'outer-predictions.json',all_maps);write(root/'outer-segments.json',all_segments)
    write(root/'complete.json',{'variants':variants,'outerFits':0,'scope':output_protocol['scope']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',default='research/rework/results/native-metric-r1');p.add_argument('--output',default='research/rework/results/native-capacity-r1');run(p.parse_args())
