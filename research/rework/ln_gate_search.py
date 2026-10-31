"""Fully reselect heads and final policies after correcting the extra gate.

All accepted auxiliary head fits are identical to the completed LN study.
Only final consumer/selection changes; no claim of new expert training evidence.
"""
import argparse,concurrent.futures,subprocess
import ln_adapter_search as s
from ln_gate import pack_gated
from search import plan


MODES=['ln_protected','off']
PARENT=s.Path('research/rework/results/ln-adapter-r1')


def policy_sets():
    base=s.policies();off={}
    for c in base:
        c={**c,'burdenGate':0.};off.setdefault(s.json.dumps(c,sort_keys=True),c)
    return {'ln_protected':base,'off':list(off.values())}


def job(arguments):
    variant,o,output,checkpoints=arguments;root=s.Path(output);p=s._PLANS[o];posts=policy_sets();heads=1 if variant=='no_update'else 31
    if all((root/mode/variant/str(o)/'complete.json').exists()for mode in MODES):return {'variant':variant,'outer':o,'resumed':True}
    validation=[s._MAIN[i]for i in p['train']]+[s._MANUAL[i]for i in p['strongTrain']];position={r['uid']:i for i,r in enumerate(validation)}
    nmap=len(p['train']);nstrong=len(p['strongTrain']);traces=[]
    pred={mode:s.np.zeros((heads*len(post),nmap,len(s.MAP_CLASSES)),dtype=s.np.uint8)for mode,post in posts.items()}
    acc={mode:s.np.zeros((len(pred[mode]),nstrong))for mode in MODES}
    local_jobs={}
    for mode,post in posts.items():
        groups=s.defaultdict(list)
        for j,c in enumerate(post):groups[(c['lnRatio'],c['scorer'],c['burdenGate'])].append(j)
        local_jobs[mode]=groups
    for head in range(heads):
        probabilities=[None]*len(validation)
        for inner,split in enumerate(p['inner']):
            records=[s._MAIN[i]for i in split['validation']]+[s._MANUAL[i]for i in split['strongValidation']]
            if head:
                path=s.Path(checkpoints)/variant/str(o)/str(inner)/(str(head-1)+'.json.gz')
                if not path.exists():raise RuntimeError('Original unchanged auxiliary fit missing; no silent refit')
                model,trace=s.fit_or_load(path,[s._MAIN[i]for i in split['train']],[s._MANUAL[i]for i in split['strongTrain']],variant,s.CONFIGS[head-1])
                local=s.adapter.predict(model,records);traces.append({'head':head,'inner':inner,'configId':head-1,'source':str(path),'fit':trace})
            else:local=[s._LN[r['uid']]for r in records]
            for r,value in zip(records,local):probabilities[position[r['uid']]]=value
        for ratio,scorer in s.itertools.product([.35,.45,.55],['model','blend']):
            values=s.cases(validation,probabilities,ratio,scorer)
            for mode,post in posts.items():
                for gate in ([0.]if mode=='off'else s.GRID['burdenGate']):
                    indices=local_jobs[mode][(ratio,scorer,gate)];c=post[indices[0]];data=pack_gated(values,c,mode);cache={}
                    local_accuracy=s.np.array([s.segment_accuracy(r,rows)for r,rows in zip(validation[nmap:],data['chunks'][nmap:])])
                    for j in indices:
                        c=post[j];key=tuple(c[k]for k in ['mode','effortPower','confidencePower'])
                        if key not in cache:cache[key]=s.summaries(data,c)
                        at=head*len(post)+j;pred[mode][at]=s.decisions(cache[key],c)[:nmap];acc[mode][at]=local_accuracy
        print('gate head',variant,o,head,'/',heads-1,flush=True)
    y=s.encode(validation[:nmap]);w=s.np.array([r['recordWeight']for r in validation[:nmap]])
    for mode,post in posts.items():
        directory=root/mode/variant/str(o);predictions=pred[mode]
        macro=s.f1_counts((predictions*y[None,:,:]*w[None,:,None]).sum(1),(predictions*(1-y[None,:,:])*w[None,:,None]).sum(1),((1-predictions)*y[None,:,:]*w[None,:,None]).sum(1))
        segment=acc[mode].mean(1);joint=.5*(macro+segment);at=max(range(len(joint)),key=lambda i:(joint[i],segment[i],macro[i],-i))
        head,policy=divmod(at,len(post));choice=post[policy]
        s.write(directory/'selection.json',{'selectionUsesOuterLabels':False,'head':head,'fitConfig':s.CONFIGS[head-1]if head else None,'policyId':policy,'policy':choice,'extraGate':mode,
          'inner':{'joint':float(joint[at]),'mapMacroF1':float(macro[at]),'segmentAccuracy':float(segment[at])}})
        s.write(directory/'search.json.gz',[{'head':i//len(post),'policy':i%len(post),'joint':float(joint[i]),'mapMacroF1':float(macro[i]),'segmentAccuracy':float(segment[i])}for i in range(len(joint))])
        s.np.savez_compressed(directory/'inner-predictions.npz',map=predictions,segmentAccuracy=acc[mode]);s.write(directory/'inner-expert-reuse.json.gz',traces)
        weak=[s._MAIN[i]for i in p['train']];manual=[s._MANUAL[i]for i in p['strongTrain']]
        if head:
            old=s.Path(checkpoints)/'outer'/variant/str(o)/(str(head-1)+'.json.gz');path=old if old.exists()else root/'new-outer-fits'/variant/str(o)/(str(head-1)+'.json.gz')
            existed=path.exists();model,trace=s.fit_or_load(path,weak,manual,variant,s.CONFIGS[head-1]);fit_source=str(path)
        else:model=s.zero_model(variant);trace={'status':'frozen_archived_prior'};existed=True;fit_source=None
        profile={'model':model,'fit':trace,'policy':choice,'extraGate':mode,'trainUids':[r['uid']for r in weak+manual],'expertFitReused':existed,'expertFitSource':fit_source}
        s.write(directory/'outer-model.json',profile)
        records=[s._MAIN[i]for i in p['test']]+[s._MANUAL[i]for i in p['strongTest']];local=s.adapter.predict(model,records)
        data=pack_gated(s.cases(records,local,choice['lnRatio'],choice['scorer']),choice,mode);prediction=s.decisions(s.summaries(data,choice),choice);rows=[];segments=[]
        for i,r in enumerate(records):
            if r['dataset']=='manual':segments.extend({'variant':variant,'outer':o,'uid':r['uid'],**x}for x in data['chunks'][i])
            else:rows.append({'variant':variant,'outer':o,'uid':r['uid'],'group':r['group'],'truth':s.target(r),'weight':r['recordWeight'],'predicted':[c for c,b in zip(s.MAP_CLASSES,prediction[i])if b]})
        s.write(directory/'outer-predictions.json',rows);s.write(directory/'outer-segments.json',segments)
        s.write(directory/'complete.json',{'innerExpertFitsReused':len(traces),'jointCandidates':len(joint),'newOuterFit':not existed})
    return {'variant':variant,'outer':o,'modes':MODES}


def run(args):
    main,manual,_=s.load();ps=plan(main,manual);root=s.Path(args.output);parent_root=s.Path(args.parent);parent=s.read(parent_root/'protocol.json')
    protocol={'id':'ln-extra-gate-r1','scope':'fixed original raw boundaries; archived Break/family logic unchanged; extra head-burden gate protected for LN or disabled; all head/consumer parameters reselected using matching inner scopes; archived prior exposure unknown',
      'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'primary':{'extraGate':'ln_protected','variant':'geometry3'},
      'parentProtocol':parent,'sourceSnapshot':{n:s.Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['ln_gate_search.py','ln_gate.py','ln_adapter_search.py','ln_adapter.py','family.py','native_search.py']},
      'postPolicies':policy_sets(),'fitGrid':s.FIT_GRID,'plans':ps,'variants':s.VARIANTS,'training':'unchanged auxiliary LN fits reused, not retrained on the gate outcome; no local proxy head pruning; exact zero head in each representation'}
    if(root/'protocol.json').exists()and s.read(root/'protocol.json')!=protocol:raise RuntimeError('Gate protocol changed')
    s.write(root/'protocol.json',protocol)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=s.init,initargs=(str(root),args.checkpoints,ps))as pool:
        for r in pool.map(job,[(v,o,str(root),args.checkpoints)for v in s.VARIANTS for o in range(6)]):print('gate outer',r,flush=True)
    for mode in MODES:
        for filename in ['outer-predictions.json','outer-segments.json']:
            rows=[]
            for variant in s.VARIANTS:
                for o in range(6):rows.extend(s.read(root/mode/variant/str(o)/filename))
            s.write(root/mode/filename,rows)
        s.write(root/mode/'protocol.json',{'id':'ln-extra-gate-'+mode+'-r1','scope':protocol['scope'],'primary':'geometry3','extraGate':mode,'parent':str(root/'protocol.json')})
        s.write(root/mode/'complete.json',{'variants':s.VARIANTS,'outerSelections':48,'jointPolicies':len(protocol['postPolicies'][mode]),'uniqueInnerExpertFitsReused':3780})
    s.write(root/'complete.json',{'modes':MODES,'variants':s.VARIANTS,'uniqueInnerExpertFitsReused':3780,'outerSelections':96,'parentAllGateControl':str(parent_root)})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4);p.add_argument('--output',default='research/rework/results/ln-extra-gate-r1')
    p.add_argument('--checkpoints',default='.cache/rework-ln-adapter-r1-fits');p.add_argument('--parent',default=str(PARENT));run(p.parse_args())
