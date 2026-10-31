"""Does release geometry help predict explicit whole-chart Release impressions?

Only explicit known LN fine-tag records enter this diagnostic; Rice, Mix and
wildcard-only records are not trivial/uncertain negative examples. No whole-map
label is copied to individual segments. This is NOT a local Release benchmark.
"""
from mil import *
import gzip
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,precision_score,recall_score
from scipy.special import expit
from scipy.optimize import minimize

OUT=Path('research/rework/results/release-impression')
GRID={'C':[.001,.01,.1,1.,10.,100.],'classWeight':[None,'balanced'],'effortPower':[0.,1.]}
VARIANTS=['legacy56','corrected56','legacy_no_tail','geometry','mechanics','legacy_plus_geometry',
 'legacy_replaced_tail','legacy_plus_mechanics','legacy_plus_both','geometry_no_isolation',
 'geometry_no_hold','geometry_no_order']
LN={'coordination','density','inverse','release','technical'}
GEOMETRY=json.loads(Path('.cache/rework-release-geometry-r2-schema.json').read_text(encoding='utf8'))['features']
MECHANICS=json.loads(Path('.cache/rework-mechanics-schema.json').read_text(encoding='utf8'))['features']

def load_extra(path):
    with Path(path).open(encoding='utf8')as f:return {r['uid']:r['features']for r in map(json.loads,f)}

def feature_keys(variant):
    original=['original:'+k for k in SCHEMA['original']];geometry=['geometry:'+k for k in GEOMETRY];mechanics=['mechanics:'+k for k in MECHANICS]
    no_tail=[k for k in original if k!='original:tailIndependence']
    sets={'legacy56':original,'corrected56':['corrected:'+k for k in SCHEMA['corrected']],
      'legacy_no_tail':no_tail,'geometry':geometry,'mechanics':mechanics,
      'legacy_plus_geometry':original+geometry,'legacy_replaced_tail':no_tail+geometry,
      'legacy_plus_mechanics':original+mechanics,'legacy_plus_both':original+geometry+mechanics}
    for name,dropped in [('isolation',{'headIsolation','independentHeldRelease'}),
                         ('hold',{'heldOtherAtTail','independentHeldRelease'}),('order',{'sameHandOrder'})]:
        sets['geometry_no_'+name]=['geometry:'+k for k in GEOMETRY if k not in dropped]
    if len(no_tail)!=55:raise RuntimeError('Tail feature identity changed')
    return sets[variant]

def weighted_pool(x,weight):
    weight=weight/weight.sum();mean=np.sum(x*weight[:,None],axis=0)
    sd=np.sqrt(np.sum((x-mean)**2*weight[:,None],axis=0));q=[]
    for column in x.T:
        order=np.argsort(column,kind='stable');i=np.searchsorted(np.cumsum(weight[order]),.9)
        q.append(column[order[min(i,len(order)-1)]])
    return np.r_[mean,sd,q]

def binary_objective(theta,z,y,weight,C):
    """Exact existing weighted logistic objective, including unpenalized bias."""
    design=np.c_[z,np.ones(len(z))];margin=design@theta;p=expit(margin);mass=weight.sum()
    penalty=np.r_[np.full(z.shape[1],1/C),0.]
    loss=(np.dot(weight,np.logaddexp(0,margin)-y*margin)+.5*np.dot(penalty,theta*theta))/mass
    gradient=(design.T@(weight*(p-y))+penalty*theta)/mass
    hessian=((design.T*(weight*p*(1-p)))@design+np.diag(penalty))/mass
    return loss,gradient,hessian

def fit_model(x,y,w,cfg):
    scaler=StandardScaler().fit(x,sample_weight=w);z=scaler.transform(x);effective=w.copy()
    if cfg['classWeight']=='balanced':effective*=np.array([w.sum()/(2*w[y==k].sum())for k in y])
    model=LogisticRegression(C=cfg['C'],max_iter=2000,tol=1e-8).fit(z,y,sample_weight=effective)
    def gradient():
        p=model.predict_proba(z)[:,1];residual=effective*(p-y)
        return max(np.max(np.abs((z.T@residual+model.coef_[0]/cfg['C'])/effective.sum())),abs(residual.sum()/effective.sum()))
    g=gradient()
    if g>1e-5:
        model.set_params(warm_start=True,max_iter=10000).fit(z,y,sample_weight=effective);g=gradient()
    recovery=None
    if g>1e-5:
        before=g;initial=np.r_[model.coef_[0],model.intercept_[0]]
        def objective(t):return binary_objective(t,z,y,effective,cfg['C'])[:2]
        result=minimize(objective,initial,jac=True,hess=lambda t:binary_objective(t,z,y,effective,cfg['C'])[2],
                        method='trust-exact',options={'gtol':1e-8,'maxiter':200})
        model.coef_=result.x[:-1][None,:];model.intercept_=result.x[-1:];g=gradient()
        recovery={'method':'trust-exact warm continuation of identical convex objective','gradientBefore':before,
                  'gradientAfter':float(g),'iterations':int(result.nit),'message':str(result.message)}
    if g>1e-5:raise RuntimeError('Binary training gradient remains too large: '+str(g))
    return scaler,model,{'iterations':int(model.n_iter_.max()),'gradientMax':float(g),**({'recovery':recovery}if recovery else{})}

def threshold_scores(y,p,w):
    unique=np.unique(p);thresholds=np.r_[0.,(unique[:-1]+unique[1:])/2,np.nextafter(1.,2.)]
    pred=p[None,:]>=thresholds[:,None];positive=y==1
    tp=(pred*w*positive).sum(1);fp=(pred*w*~positive).sum(1)
    fn=((~pred)*w*positive).sum(1);tn=((~pred)*w*~positive).sum(1)
    fpos=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=2*tp+fp+fn>0)
    fneg=np.divide(2*tn,2*tn+fp+fn,out=np.zeros_like(tp),where=2*tn+fp+fn>0)
    return thresholds,(fpos+fneg)/2,fpos

def metrics(y,probability,predicted,w):
    return {'macroF1':float(f1_score(y,predicted,labels=[0,1],average='macro',sample_weight=w,zero_division=0)),
      'releaseF1':float(f1_score(y,predicted,sample_weight=w,zero_division=0)),
      'precision':float(precision_score(y,predicted,sample_weight=w,zero_division=0)),
      'recall':float(recall_score(y,predicted,sample_weight=w,zero_division=0)),
      'averagePrecision':float(average_precision_score(y,probability,sample_weight=w))}

def reuse_lineage(previous,configs):
    """Partial outputs can omit ancestor scores while retaining their traces.

    Merge valid same-feature ancestors before continuing; deduplicate actual
    config/fold fits and keep each saved outer model tied to its own choice.
    """
    layers=[];seen=set();path=Path(previous);reference=None
    while path.resolve()not in seen:
        seen.add(path.resolve());protocol=json.loads((path/'protocol.json').read_text(encoding='utf8'))
        if reference is None:reference=protocol
        if any(protocol['sourceSnapshot'][p]!=reference['sourceSnapshot'][p]for p in ['release-geometry.cjs','mechanics-probe.cjs']):break
        if any(protocol[k]!=reference[k]for k in ['records','plans','featureSets']):raise RuntimeError('Reuse lineage data scope differs')
        layers.append((path,protocol));ancestor=protocol.get('extension',{}).get('source')
        if not ancestor:break
        path=Path(ancestor.replace('\\','/'))
    search={};traces={};choices={};models={};predictions=[]
    key=lambda r:(r['variant'],r['outer'],json.dumps(r['config'],sort_keys=True))
    for path,protocol in layers:
        for row in json.loads((path/'search.json').read_text(encoding='utf8')):search.setdefault(key(row),row)
        old_configs=[dict(zip(protocol['grid'],v))for v in itertools.product(*protocol['grid'].values())]
        for row in json.loads((path/'fits.json').read_text(encoding='utf8')):
            cfg=old_configs[row['configId']];identity=(row['variant'],row['outer'],row['inner'],json.dumps(cfg,sort_keys=True))
            traces.setdefault(identity,{**row,'configId':configs.index(cfg),'reusedFrom':str(path)})
        for row in json.loads((path/'selection.json').read_text(encoding='utf8'))['selections']:
            identity=(row['variant'],row['outer']);model=path/'models'/f'{identity[0]}-{identity[1]}.json'
            if model.exists()and identity not in choices:choices[identity]=row;models[identity]=model
    for identity,choice in choices.items():
        for path,protocol in layers:
            source_choices={(r['variant'],r['outer']):r for r in json.loads((path/'selection.json').read_text(encoding='utf8'))['selections']}
            matched=source_choices.get(identity)
            if matched and matched['config']==choice['config']and matched['threshold']==choice['threshold']:
                saved=[r for r in json.loads((path/'outer-predictions.json').read_text(encoding='utf8'))if(r['variant'],r['outer'])==identity]
                if saved:predictions.extend(saved);break
    return search,choices,predictions,list(traces.values()),models,[str(p)for p,_ in layers]

def main_run():
    import argparse,shutil
    global OUT,GRID
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(OUT));parser.add_argument('--c-grid');reuse=parser.add_mutually_exclusive_group();reuse.add_argument('--extend-from');reuse.add_argument('--repair-from')
    args=parser.parse_args();OUT=Path(args.output);prior=args.repair_from or args.extend_from;previous=Path(prior)if prior else None
    if previous:GRID={**GRID,'C':[.00001,.0001,.001,.01,.1,1.,10.,100.,1000.]}
    if args.c_grid:
        values=[float(v)for v in args.c_grid.split(',')]
        if values!=sorted(set(values))or any(not math.isfinite(v)or v<=0 for v in values):raise ValueError('C grid must be positive finite unique ascending values')
        GRID={**GRID,'C':values}
    main,_,_=load();records=[r for r in main if set(target(r))&LN]
    geometry=load_extra('.cache/rework-release-geometry-r2.jsonl');mechanics=load_extra('.cache/rework-mechanics.jsonl')
    for r in records:
        for extra,key in [(geometry,'geometry'),(mechanics,'mechanics')]:
            if len(extra[r['uid']])!=len(r['segments']):raise RuntimeError('Representation partition mismatch')
            for s,v in zip(r['segments'],extra[r['uid']]):s[key]=v
    y=np.array(['release'in target(r)for r in records],int);w=np.array([r['recordWeight']for r in records])
    folds=grouped_folds(records,6);plans=[]
    for test in folds:
        train=[i for i in range(len(records))if i not in set(test)];inner=grouped_folds([records[i]for i in train],3)
        plans.append({'train':train,'test':test,'innerValidation':[[train[j]for j in fold]for fold in inner]})
    configs=[dict(zip(GRID,v))for v in itertools.product(*GRID.values())]
    protocol={'id':'release-impression-order-repair-r3'if args.repair_from else'release-impression-r2'if previous else'release-impression-r1','task':'explicit whole-chart Release impression versus other explicit known LN fine labels',
      'scope':'fixed original automatic intervals; no local Release truth inferred from map labels; no deployment model selected',
      'records':[{'uid':r['uid'],'group':r['group'],'truth':int(y[i]),'tags':target(r),'weight':r['recordWeight']}for i,r in enumerate(records)],
      'groups':len({r['group']for r in records}),'positiveGroups':len({r['group']for i,r in enumerate(records)if y[i]}),
      'exclusions':'Rice-only, Mix-only, wildcard-only, unlabelled and manual records excluded; explicit fine tags need not exhaust physical subordinate attributes',
      'featureSets':{v:feature_keys(v)for v in VARIANTS},'grid':GRID,
      'pooling':'same duration * modern effort^a for each representation; weighted mean/sd/q90, no class prediction used for pooling',
      'thresholds':'all distinct inner prediction partitions; exact binary macro-F1 tuning crossed with every C/classWeight/effort configuration',
      'selection':'inner pooled binary macro-F1, then Release F1, AP, smaller C, deterministic order',
      'plans':plans,'sourceBaseCommit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{name:(Path(__file__).parent/name).read_text(encoding='utf8')for name in ['release-study.py','release-extract.cjs','release-geometry.cjs','mechanics-probe.cjs','mil.py']}}
    protocol['geometryRepairSpecification']=json.loads(Path('.cache/rework-release-geometry-r2-schema.json').read_text(encoding='utf8'))
    if args.c_grid:protocol['id']='release-impression-expanded-r4'
    previous_search={};previous_choices={};previous_predictions=[];previous_traces=[]
    if previous:
        old=json.loads((previous/'protocol.json').read_text(encoding='utf8'))
        if not args.repair_from:
            for producer in ['release-geometry.cjs','mechanics-probe.cjs']:
                if old['sourceSnapshot'][producer]!=protocol['sourceSnapshot'][producer]:raise RuntimeError('Feature producer differs; use explicit coordinate repair rather than fit reuse')
        for key in ['records','plans','featureSets']:
            if old[key]!=protocol[key]:raise RuntimeError('Cannot reuse changed '+key)
        for key in ['classWeight','effortPower']:
            if old['grid'][key]!=GRID[key]:raise RuntimeError('Cannot reuse changed '+key)
        previous_search,previous_choices,previous_predictions,previous_traces,previous_models,lineage=reuse_lineage(previous,configs)
        protocol['extension']={'source':str(previous),'reason':'inner choices hit both C boundaries; expand the common C range for EVERY ablation, not just the promising representation',
          'reuse':'same data, folds, pooling, training objective and threshold search; retain all valid existing inner scores/fits and unchanged outer selections',
          'descriptionErratum':'r1 protocol text said legacy burden; executed source and cached segment effort used modern effort. This correction changes no feature or fit.'}
        protocol['reuseLineage']=lineage
        if args.repair_from:
            valid={v for v in VARIANTS if 'geometry:sameHandOrder'not in feature_keys(v)}
            previous_search={k:v for k,v in previous_search.items()if k[0]in valid}
            previous_choices={k:v for k,v in previous_choices.items()if k[0]in valid}
            previous_traces=[r for r in previous_traces if r['variant']in valid]
            protocol['extension']={'source':str(previous),'reason':'actual six-key mirror audit exposed arbitrary tied-release endpoint order; repair the only affected coordinate',
              'unchangedVariants':sorted(valid),'reuse':'only coordinate-excluding representations retain their valid fits and outer outputs; every affected representation is fully refitted over the same expanded common grid and retuned threshold space'}
        elif args.c_grid:
            protocol['extension']['reason']='r3 selected the common C lower bound in12/72 choices and upper bound in6/72; expand both ends for all12 ablations, retaining every unchanged valid fit'
            if not(previous/'complete.json').exists():
                protocol['id']='release-impression-gradient-recovery-r5'
                protocol['extension']['reason']='resume all persisted valid fits and choices of the partial common-grid study; warm exact-Hessian continuation only when the original optimizer leaves gradient above1e-5, identical scalar objective and stop threshold'
    if(OUT/'protocol.json').exists()and json.loads((OUT/'protocol.json').read_text(encoding='utf8'))!=protocol:raise RuntimeError('Release diagnostic protocol changed')
    dump(OUT/'protocol.json',protocol);matrices={}
    for variant in VARIANTS:
        keys=feature_keys(variant);matrices[variant]={}
        for power in GRID['effortPower']:
            rows=[]
            for r in records:
                x=np.array([[s[group][(SCHEMA[group]if group in SCHEMA else GEOMETRY if group=='geometry'else MECHANICS).index(name)]for group,name in (k.split(':',1)for k in keys)]for s in r['segments']])
                duration=np.array([s['end']-s['start']for s in r['segments']]);burden=np.array([s['effort']for s in r['segments']])
                rows.append(weighted_pool(x,duration*np.maximum(burden,1e-4)**power))
            matrices[variant][power]=np.array(rows)
    allpred=[];choices=[];search=[];traces=list(previous_traces)
    for variant in VARIANTS:
      for outer,plan in enumerate(plans):
        best=None
        for cid,cfg in enumerate(configs):
            old_row=previous_search.get((variant,outer,json.dumps(cfg,sort_keys=True)))
            if old_row:
                row={**old_row,'configId':cid,'reusedFrom':str(previous)};search.append(row)
                rank=(row['macroF1'],row['releaseF1'],row['averagePrecision'],-cfg['C'])
                if best is None or rank>best[0]:best=(rank,row)
                continue
            x=matrices[variant][cfg['effortPower']];ids=[];probability=[]
            for inner,val in enumerate(plan['innerValidation']):
                train=[i for i in plan['train']if i not in set(val)]
                scaler,model,trace=fit_model(x[train],y[train],w[train],cfg)
                traces.append({'variant':variant,'outer':outer,'inner':inner,'configId':cid,**trace})
                ids.extend(val);probability.extend(model.predict_proba(scaler.transform(x[val]))[:,1])
            probability=np.array(probability);ts,macro,positive=threshold_scores(y[ids],probability,w[ids]);ap=float(average_precision_score(y[ids],probability,sample_weight=w[ids]))
            at=max(range(len(ts)),key=lambda j:(macro[j],positive[j],-abs(ts[j]-.5),-ts[j]))
            row={'variant':variant,'outer':outer,'configId':cid,'config':cfg,'threshold':float(ts[at]),'macroF1':float(macro[at]),'releaseF1':float(positive[at]),'averagePrecision':ap,'thresholdPartitions':len(ts)}
            search.append(row);rank=(macro[at],positive[at],ap,-cfg['C'])
            if best is None or rank>best[0]:best=(rank,row)
        choice=best[1];choices.append(choice);dump(OUT/'selection.json',{'outerOpenedForCurrentSelection':False,'selections':choices})
        old_choice=previous_choices.get((variant,outer))
        if old_choice and old_choice['config']==choice['config']and old_choice['threshold']==choice['threshold']:
            path=OUT/'models'/f'{variant}-{outer}.json';path.parent.mkdir(parents=True,exist_ok=True)
            previous_model=previous_models[(variant,outer)];old_model=json.loads(previous_model.read_text(encoding='utf8'))
            train,test=plan['train'],plan['test']
            if old_model['trainUids']!=[records[i]['uid']for i in train]or old_model['testUids']!=[records[i]['uid']for i in test]:raise RuntimeError('Prior outer model training/test scope differs')
            saved=[r for r in previous_predictions if r['variant']==variant and r['outer']==outer]
            if len(saved)==len(test)and {r['uid']for r in saved}=={records[i]['uid']for i in test}:
                allpred.extend({**r,'reusedFrom':str(previous)}for r in saved)
            elif not saved:
                # A partial run can have committed the valid selected model
                # before the per-variant prediction file was published.
                x=matrices[variant][choice['config']['effortPower']]
                z=(x[test]-np.array(old_model['mean']))/np.array(old_model['scale'])
                probability=expit(z@np.array(old_model['W'])+old_model['b'])
                allpred.extend({'variant':variant,'outer':outer,'uid':records[i]['uid'],'group':records[i]['group'],'truth':int(y[i]),'weight':w[i],
                  'probability':float(p),'predicted':int(p>=choice['threshold']),'reusedModelFrom':str(previous)}for i,p in zip(test,probability))
            else:raise RuntimeError('Partial or repeated saved outer predictions')
            shutil.copyfile(previous_model,path)
            print('release impression reused outer',variant,outer,flush=True);continue
        cfg=choice['config'];x=matrices[variant][cfg['effortPower']];train,test=plan['train'],plan['test']
        scaler,model,trace=fit_model(x[train],y[train],w[train],cfg);p=model.predict_proba(scaler.transform(x[test]))[:,1]
        dump(OUT/'models'/f'{variant}-{outer}.json',{'config':cfg,'threshold':choice['threshold'],'features':feature_keys(variant),'pool':['weightedMean','weightedSd','weightedQ90'],
          'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),'W':model.coef_[0].tolist(),'b':float(model.intercept_[0]),'fit':trace,
          'trainUids':[records[i]['uid']for i in train],'testUids':[records[i]['uid']for i in test]})
        for i,prob in zip(test,p):allpred.append({'variant':variant,'outer':outer,'uid':records[i]['uid'],'group':records[i]['group'],'truth':int(y[i]),'weight':w[i],'probability':float(prob),'predicted':int(prob>=choice['threshold'])})
        print('release impression',variant,outer,flush=True)
      dump(OUT/'outer-predictions.json',allpred);dump(OUT/'search.json',search);dump(OUT/'fits.json',traces)
    result={}
    for variant in VARIANTS:
        rows=[r for r in allpred if r['variant']==variant]
        result[variant]=metrics(np.array([r['truth']for r in rows]),[r['probability']for r in rows],[r['predicted']for r in rows],np.array([r['weight']for r in rows]))
    original={r['uid']:r for r in json.loads(Path('.cache/rework-legacy-full.json').read_text(encoding='utf8'))}
    pred=np.array(['release'in original[r['uid']]['overall']['tags']for r in records],int)
    result['historical_fixed']=metrics(y,pred,pred,w)
    result['historical_fixed'].update(averagePrecision=None,scoreKind='hard overall decision only; not a continuous ranking score')
    dump(OUT/'metrics.json',result);dump(OUT/'complete.json',{'variants':VARIANTS,'innerFits':len(traces),
      'reusedInnerFits':sum('reusedFrom'in r for r in traces),'newInnerFits':sum('reusedFrom'not in r for r in traces),'scope':protocol['scope']})
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main_run()
