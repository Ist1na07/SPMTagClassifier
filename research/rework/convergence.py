"""Training-only refinement of the bounded r2 optimizer.

No labels or scores from validation/test enter the stopping decision.  A small
gradient is sufficient; otherwise require two consecutive stable training
objective AND prediction checks.  The latter is numerical stability, not a
claim of a global or differentiable stationary optimum (max-MIL is nonsmooth).
"""
from mil import *

POLICY={'id':'train-stability-r3','gradientMax':1e-5,'chunkIterations':250,
        'maxChunks':16,'lbfgsGradientTolerance':1e-7,'lbfgsChangeTolerance':1e-12,
        'relativeLossChange':1e-8,'probabilityMeanChange':1e-5,
        'probabilityMaxChange':1e-3,'consecutiveStableChecks':2}

def snapshot(model,data,config,objective=None):
    model.zero_grad(set_to_none=True)
    loss=(objective or model.objective)(data,config);loss.backward()
    gradient=max(float(p.grad.abs().max())for p in model.parameters()if p.grad is not None)
    with torch.no_grad():p,z,_=model(data,config)
    return float(loss.detach()),gradient,p.detach().clone(),torch.sigmoid(z).detach().clone()

def polish(model,data,config,policy=POLICY,checkpoint=None,objective=None):
    objective=objective or model.objective
    before=snapshot(model,data,config,objective);history=[];stable=0;status='budget_exhausted'
    if before[1]<=policy['gradientMax']:
        return {'policy':policy,'status':'small_gradient','chunks':[],
                'loss':before[0],'gradientMax':before[1]}
    optimizer=torch.optim.LBFGS(model.parameters(),lr=1,max_iter=policy['chunkIterations'],
        tolerance_grad=policy['lbfgsGradientTolerance'],
        tolerance_change=policy['lbfgsChangeTolerance'],line_search_fn='strong_wolfe')
    for chunk in range(policy['maxChunks']):
        calls=0
        def closure():
            nonlocal calls
            calls+=1;optimizer.zero_grad();loss=objective(data,config);loss.backward();return loss
        optimizer.step(closure);after=snapshot(model,data,config,objective)
        dp=(after[2]-before[2]).abs();dz=(after[3]-before[3]).abs()
        row={'chunk':chunk+1,'evaluations':calls,'loss':after[0],'gradientMax':after[1],
             'relativeLossChange':abs(after[0]-before[0])/max(1.,abs(before[0])),
             'segmentProbabilityMeanChange':float(dp.mean()),'segmentProbabilityMaxChange':float(dp.max()),
             'mapProbabilityMeanChange':float(dz.mean()),'mapProbabilityMaxChange':float(dz.max()),
             'trainingPrimaryFlipFraction':float((after[2].argmax(1)!=before[2].argmax(1)).double().mean())}
        if after[0]>before[0]+1e-9:
            raise RuntimeError('LBFGS increased training objective; do not silently accept the fit')
        history.append(row)
        stable_now=(row['relativeLossChange']<=policy['relativeLossChange'] and
                    max(row['segmentProbabilityMeanChange'],row['mapProbabilityMeanChange'])<=policy['probabilityMeanChange'] and
                    max(row['segmentProbabilityMaxChange'],row['mapProbabilityMaxChange'])<=policy['probabilityMaxChange'])
        stable=stable+1 if stable_now else 0
        if checkpoint:checkpoint(model,history)
        before=after
        if after[1]<=policy['gradientMax']:status='small_gradient';break
        if stable>=policy['consecutiveStableChecks']:status='train_prediction_stable';break
    return {'policy':policy,'status':status,'chunks':history,'loss':before[0],'gradientMax':before[1]}

def bounded_stage(model,data,config,objective):
    parameters=[p for p in model.parameters()if p.requires_grad]
    optimizer=torch.optim.Adam(parameters,lr=.04);previous=None;stable=0
    for step in range(300):
        optimizer.zero_grad();loss=objective(data,config);loss.backward();optimizer.step()
        value=float(loss.detach());stable=stable+1 if previous is not None and abs(value-previous)<1e-7 else 0
        previous=value
        if stable>=15:break
    lb=torch.optim.LBFGS(parameters,lr=1,max_iter=100,tolerance_grad=1e-6,tolerance_change=1e-10,line_search_fn='strong_wolfe');calls=0
    def closure():
        nonlocal calls
        calls+=1;lb.zero_grad();loss=objective(data,config);loss.backward();return loss
    lb.step(closure);loss,gradient,_,_=snapshot(model,data,config,objective)
    return {'adamSteps':step+1,'lbfgsEvaluations':calls,'loss':loss,'gradientMax':gradient}

def control_fit(rows,variant,config):
    torch.manual_seed(20261030);data=Packed(rows,variant);model=Model(data.x.shape[1],True)
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('local.'))
    local_base=bounded_stage(model,data,config,model.local_control_objective)
    local_refinement=polish(model,data,config,objective=model.local_control_objective)
    for name,p in model.named_parameters():p.requires_grad_(not name.startswith('local.'))
    base=bounded_stage(model,data,config,model.objective)
    base['localStage']={'bounded':local_base,'refinement':local_refinement}
    return model,data,base

def expanded_budget_policy(previous,current):
    """The training stop criteria are unchanged; only the safety cap grows."""
    return (current['maxChunks']>previous['maxChunks']and
      {k:v for k,v in previous.items()if k not in ['id','maxChunks']}==
      {k:v for k,v in current.items()if k not in ['id','maxChunks']})

def refined_fit(rows,variant,config,state_path=None,policy=POLICY,continue_exhausted=False):
    """Keep a recoverable train-only model, not just validation predictions."""
    path=Path(state_path)if state_path else None
    identity={'variant':variant,'config':config,'trainUids':[r['uid']for r in rows]}
    if path and path.exists():
        state=torch.load(path,weights_only=True)
        if state['identity']!=identity:
            raise RuntimeError('Refinement checkpoint identity/policy differs')
        if state['policy']!=policy:
            if not(continue_exhausted and state.get('complete')and state.get('refinement',{}).get('status')=='budget_exhausted'and expanded_budget_policy(state['policy'],policy)):
                raise RuntimeError('Refinement checkpoint identity/policy differs')
            state['complete']=False
            state['base']['continuedFrom']={'policy':state['policy'],'refinement':state['refinement']}
        data=Packed(rows,variant,np.array(state['mean']),np.array(state['scale']))
        model=Model(data.x.shape[1],VARIANTS[variant][1]);model.load_state_dict(state['state'])
        if variant in ['frozen_prior','strong_only_prior','copied_prior']:
            for p in model.local.parameters():p.requires_grad_(False)
        base=state['base']
        if state.get('complete'):
            return model,data,{**base,'refinement':state['refinement'],
                               'loss':state['refinement']['loss'],'gradientMax':state['refinement']['gradientMax']}
    else:
        model,data,base=(control_fit(rows,variant,config)if variant in ['strong_only_prior','copied_prior']else fit(rows,variant,config));state={}
    def save(model,history,complete=False,result=None):
        if path is None:return
        path.parent.mkdir(parents=True,exist_ok=True)
        value={'identity':identity,'policy':policy,'state':model.state_dict(),'restarts':restarts,
               'mean':data.mean.tolist(),'scale':data.scale.tolist(),'base':base,
               'complete':complete,'refinement':result,'lastChunkHistory':history}
        temporary=path.with_suffix('.tmp');torch.save(value,temporary);temporary.replace(path)
    # A resumed incomplete polish restarts LBFGS history, never the fitted model.
    # Record that event; exact uninterrupted evidence remains separately intact.
    restarts=state.get('restarts',0)+(1 if state else 0)
    save(model,[])
    result=polish(model,data,config,policy=policy,checkpoint=lambda m,h:save(m,h))
    result['checkpointRestarts']=restarts
    save(model,result['chunks'],True,result)
    return model,data,{**base,'refinement':result,'loss':result['loss'],'gradientMax':result['gradientMax']}

def main_run():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--keys')
    parser.add_argument('--output',default='research/rework/results/optimizer-stability-pilot-extended.json')
    args=parser.parse_args()
    # The same pre-existing ten cases were chosen solely by TRAINING gradient.
    audit=json.loads(Path('research/rework/results/optimizer-audit.json').read_text(encoding='utf8'))
    protocol=json.loads(Path('research/rework/results/mil-stage1-r2/protocol.json').read_text(encoding='utf8'))
    from search import configs
    main,manual,_=load();results=[]
    for case in audit['cases']:
        if args.keys and case['fitKey']not in args.keys.split(','):continue
        v,o,i,c=case['fitKey'].split('/');o,i,c=map(int,(o,i,c));split=protocol['plans'][o]['inner'][i]
        rows=[main[j]for j in split['train']]+[manual[j]for j in split['strongTrain']]
        cfg=configs(v)[c];start=time.perf_counter()
        model,data,trace=refined_fit(rows,v,cfg,Path('.cache/rework-convergence-pilot')/(case['fitKey'].replace('/','-')+'.pt'))
        result=trace['refinement']
        results.append({'fitKey':case['fitKey'],'original':trace,'refinement':result,'seconds':time.perf_counter()-start})
        dump(args.output,{'scope':'training-only; same gradient-selected cases; no validation/test scoring','policy':POLICY,'cases':results})
        print(case['fitKey'],result['status'],result['gradientMax'],'chunks',len(result['chunks']),flush=True)

if __name__=='__main__':main_run()
