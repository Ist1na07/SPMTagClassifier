"""Strictly nested ablations on the union of all existing valid candidates.

No new fits or forward passes. Shared candidate identity and every duplicate
prediction are checked; all actual candidate scores are saved by train scope.
"""
from mil import np,json,Path,load,encode,target,MAP_CLASSES
import argparse,gzip
import native_metric_search as metric
import native_followup as source
from native_search import f1_counts,select_segments
from policy_capacity import changes,BUDGETS,candidate_metrics


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


def scope_metrics(packed,accuracy,y,weights,ids,strong,block=2048):
    macro=np.zeros(len(packed));segment=accuracy[:,strong].mean(1)
    truth=y[ids][None,:,:];weight=weights[ids][None,:,None]
    for lo in range(0,len(packed),block):
        p=np.unpackbits(packed[lo:lo+block][:,ids,:],axis=-1,count=y.shape[1],bitorder='little')
        macro[lo:lo+len(p)]=f1_counts((p*truth*weight).sum(1),(p*(1-truth)*weight).sum(1),((1-p)*truth*weight).sum(1))
    return macro,segment,.5*(macro+segment)


def best(scores,ids):
    macro,segment,joint=scores
    order=np.lexsort((ids,-macro[ids],-segment[ids],-joint[ids]))
    return int(ids[order[0]])


def build_union(parent,root,nmap,nmanual):
    complete=read(parent/'complete.json');names=[n for n in complete['variants']if n!='metric_original_control']
    configs=[];index={};mappings={};origins=[]
    for group,name in enumerate(names):
        local=[]
        for row,c in enumerate(read(parent/name/'candidates.json')):
            key=json.dumps(c,sort_keys=True)
            if key not in index:index[key]=len(configs);configs.append(c);origins.append([group,row])
            local.append(index[key])
        mappings[name]=local
    packed=np.zeros((len(configs),nmap,(len(MAP_CLASSES)+7)//8),dtype=np.uint8)
    accuracy=np.zeros((len(configs),nmanual));seen=np.zeros(len(configs),dtype=bool);duplicates=0
    for name in names:
        with np.load(parent/name/'candidate-predictions.npz')as saved:
            p=np.packbits(saved['map'],axis=-1,bitorder='little');a=saved['segmentAccuracy']
        for row,idx in enumerate(mappings[name]):
            if seen[idx]:
                if not np.array_equal(packed[idx],p[row])or not np.array_equal(accuracy[idx],a[row]):raise RuntimeError('Identical candidate has inconsistent actual predictions: '+name+' '+str(row))
                duplicates+=1
            else:packed[idx]=p[row];accuracy[idx]=a[row];seen[idx]=True
    assert seen.all()
    root.mkdir(parents=True,exist_ok=True)
    with gzip.open(root/'candidates.json.gz','wt',encoding='utf8')as f:json.dump(configs,f,ensure_ascii=False,separators=(',',':'))
    np.savez_compressed(root/'candidate-predictions.npz',packedMap=packed,segmentAccuracy=accuracy)
    np.savez_compressed(root/'candidate-origins.npz',origins=np.array(origins,dtype=np.int32))
    write(root/'pool-audit.json',{'uniqueCandidates':len(configs),'duplicateCandidatesChecked':duplicates,'originGroupOrder':names,
      'encoding':{'classOrder':MAP_CLASSES,'bitorder':'little','count':len(MAP_CLASSES)},
      'sourceGroupToUnionIndex':mappings,'meaning':'all duplicate packed map predictions and every annotated-chart metric are bitwise equal'})
    return configs,packed,accuracy


def nested_select(scores,packed,accuracy,y,weights,plan,ids,costs):
    minimum=int(costs[ids].min());allowed=[ids[costs[ids]<=minimum+b]if b is not None else ids for b in BUDGETS]
    val_map=np.zeros((len(BUDGETS),len(y),y.shape[1]),dtype=np.uint8);val_accuracy=np.zeros((len(BUDGETS),accuracy.shape[1]));selections=[]
    for j,split in enumerate(plan['inner']):
        for b,eligible in enumerate(allowed):
            choice=best(scores['inner'][j],eligible)
            val_map[b,split['validation']]=np.unpackbits(packed[choice,split['validation']],axis=-1,count=y.shape[1],bitorder='little')
            val_accuracy[b,split['strongValidation']]=accuracy[choice,split['strongValidation']]
            selections.append({'inner':j,'budget':BUDGETS[b],'candidate':choice,'changes':int(costs[choice]),'trainingJoint':float(scores['inner'][j][2][choice])})
    validation=candidate_metrics(val_map,val_accuracy,y,weights,plan['train'],plan['strongTrain'])
    budget=max(range(len(BUDGETS)),key=lambda b:(validation[2][b],-b));choice=best(scores['outer'],allowed[budget])
    return {'candidate':choice,'budget':BUDGETS[budget],'changes':int(costs[choice]),'minimumRequiredChanges':minimum,
      'innerPolicyChoices':selections,'innerBudgetScores':[{'budget':b,'mapMacroF1':float(validation[0][j]),'segmentAccuracy':float(validation[1][j]),'joint':float(validation[2][j])}for j,b in enumerate(BUDGETS)]},best(scores['outer'],ids),best(scores['outer'],allowed[0])


def run(args):
    metric.configure();parent=Path(args.parent);root=Path(args.output);protocol=read(parent/'protocol.json')
    if not(parent/'complete.json').exists():raise RuntimeError('Complete parent ablations required')
    main,manual,_=load();identity=[{'uid':r['uid'],'group':r['group'],'tags':target(r),'weight':r.get('recordWeight',1)}for r in main+manual]
    if identity!=protocol['records']:raise RuntimeError('Parent record scope differs')
    output={'id':'native-nested-capacity-union-r1','scope':'retrospective shared-union candidate ablations and nested capacity selection; historical experts frozen, exposure unknown',
      'parentProtocol':protocol,'plans':protocol['plans'],'budgets':BUDGETS,'primary':'pool_repaired_nested',
      'candidateMethod':'label-independent deduplicated union of ALL existing nonhistorical candidate configs; each ablation filters this same pool; every remaining config jointly scored',
      'selection':'inner train chooses complete policy per capacity; pooled inner validation chooses capacity; outer train reselects policy; unrestricted and fixed-minimum controls prespecified',
      'sourceSnapshot':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['native_union.py','policy_capacity.py']}}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=output:raise RuntimeError('Union protocol changed')
    write(root/'protocol.json',output)
    if(root/'candidate-predictions.npz').exists():
        with gzip.open(root/'candidates.json.gz','rt',encoding='utf8')as f:configs=json.load(f)
        with np.load(root/'candidate-predictions.npz')as saved:packed=saved['packedMap'];accuracy=saved['segmentAccuracy']
    else:configs,packed,accuracy=build_union(parent,root,len(main),len(manual))
    print('shared union',len(configs),'candidates',flush=True)
    y=encode(main);weights=np.array([r['recordWeight']for r in main]);plans=protocol['plans'];scores=[]
    for o,plan in enumerate(plans):
        scopes={};pairs=[('outer',plan)]+[(str(j),s)for j,s in enumerate(plan['inner'])]
        for name,split in pairs:
            path=root/'scope-scores'/f'{o}-{name}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                with np.load(path)as saved:scopes[name]=(saved['macro'],saved['segment'],saved['joint'])
            else:
                scopes[name]=scope_metrics(packed,accuracy,y,weights,split['train'],split['strongTrain'])
                np.savez_compressed(path,macro=scopes[name][0],segment=scopes[name][1],joint=scopes[name][2])
            print('union scored',o,name,flush=True)
        scores.append({'outer':scopes['outer'],'inner':[scopes[str(j)]for j in range(3)]})
    costs=np.array([len(changes(c,protocol['sourceParameters']))for c in configs]);maps=[];segments=[];variants=[];sizes={}
    for group,spec in protocol['groups'].items():
        if group=='metric_original_control':continue
        ids=np.array([i for i,c in enumerate(configs)if c['source']in spec['sources']and all(c[k]==v for k,v in spec.get('fixed',{}).items())],dtype=int)
        if not len(ids):raise RuntimeError('Empty shared-pool ablation')
        name=group.replace('metric_','pool_');sizes[name]=len(ids);selection=[]
        for o,plan in enumerate(plans):
            choice,unrestricted,minimum=nested_select(scores[o],packed,accuracy,y,weights,plan,ids,costs)
            selection.append({'outer':o,**choice,'config':configs[choice['candidate']],'unrestrictedCandidate':unrestricted,'minimumCandidate':minimum})
            write(root/name/'selection.json',{'selectionUsesOuterLabels':False,'selections':selection})
            cases=[(name+'_nested',choice['candidate']),(name+'_unrestricted',unrestricted)]
            if group=='metric_repaired':cases.append(('pool_minimum_control',minimum))
            for variant,i in cases:
                if variant not in variants:variants.append(variant)
                c=configs[i];p=np.unpackbits(packed[i,plan['test']],axis=-1,count=len(MAP_CLASSES),bitorder='little')
                for j,k in enumerate(plan['test']):
                    r=main[k];maps.append({'variant':variant,'outer':o,'uid':r['uid'],'group':r['group'],'truth':target(r),'weight':r['recordWeight'],
                      'predicted':[label for label,b in zip(MAP_CLASSES,p[j])if b]})
                for k in plan['strongTest']:
                    r=manual[k];value=source.load_source(c['source'],[r])[0]
                    segments.extend({'variant':variant,'outer':o,'uid':r['uid'],**s}for s in select_segments(value,c))
            print('union selected',name,o,'budget',choice['budget'],'changes',choice['changes'],flush=True)
        write(root/name/'eligible-candidates.json',ids.tolist())
    write(root/'candidate-counts.json',sizes);write(root/'costs.json',costs.tolist());write(root/'outer-predictions.json',maps);write(root/'outer-segments.json',segments)
    write(root/'complete.json',{'variants':variants,'uniqueCandidates':len(configs),'outerFits':0,'scope':output['scope']})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',default='research/rework/results/native-metric-r1');p.add_argument('--output',default='research/rework/results/native-union-r1');run(p.parse_args())
