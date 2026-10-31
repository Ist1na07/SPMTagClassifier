"""Retune four deterministic support treatments over unchanged expert fits.

All effective training configs are reconsidered. Reusing a coefficient fit does
not imply reusing its old validation scores or final model selection.
"""
from coarse_model import np,json,Path,SPECS,materialize,training_rows
from coarse_data import load_headers,plans
from coarse_search import configurations,read,write,source_metric,segment_metric,fit_or_load
from coarse_supported import predict_modes,prepare_prediction
import argparse,subprocess

MODES=['none','jack','ln','both']


def evaluate(args):
    weak,manual,_=load_headers();ps=plans(weak,manual);root=Path(args.output);source=Path(args.fits);outer_cache=Path(args.outer_fits)
    variant=args.variant;spec=SPECS[variant];cfgs=configurations(variant);criterion='4k'if variant=='weak4_only'else'shared'
    protocol={'id':'coarse-physical-support-r1','scope':'unchanged raw auxiliary-trained expert plus deterministic sufficient physical templates; all expert configs reselected on final supported outputs',
      'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'variant':variant,'spec':spec,'configs':cfgs,'modes':MODES,'primary':'both',
      'criterion':criterion,'plans':ps,'expertFits':str(source),'training':'unchanged original coarse auxiliary objective; existing fit reused only with exact config/spec/trainUids and accepted optimizer status',
      'sourceSnapshot':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['coarse_support_search.py','coarse_supported.py','physical-support.cjs','coarse-runtime.cjs','coarse_model.py','coarse_data.py']},
      'records':[{'uid':r['uid'],'group':r['group'],'keys':r['keys'],'truth':r['coarseTags'],'weight':r['recordWeight']}for r in weak],
      'manual':[{'uid':r['uid'],'annotations':r['annotations']}for r in manual]}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise RuntimeError('Support study protocol changed')
    write(root/'protocol.json',protocol);all_maps=[];all_segments=[];outer_fits=[]
    for o,plan in enumerate(ps):
        for inner,split in enumerate(plan['inner']):
            done=root/'inner'/str(o)/str(inner)/'complete.json'
            if done.exists():continue
            val=materialize([weak[i]for i in split['validation']]);strong=materialize([manual[i]for i in split['strongValidation']])
            n=len(val);ns=len(strong);p={mode:np.zeros((len(cfgs),n),dtype=np.uint8)for mode in MODES};a={mode:np.zeros((len(cfgs),ns))for mode in MODES};traces=[]
            records=val+strong;packed={}
            expected=[r['uid']for r in training_rows([weak[i]for i in split['train']]+[manual[i]for i in split['strongTrain']],spec)]
            for cid,cfg in enumerate(cfgs):
                path=source/variant/str(o)/str(inner)/(str(cid)+'.json.gz')
                saved=read(path)
                if saved['config']!=cfg or saved['spec']!=spec or saved['trainUids']!=expected:raise RuntimeError('Expert fit scope/config differs')
                if saved['fit']['status']not in ['small_projected_gradient','train_prediction_stable']:raise RuntimeError('Expert training stop unaccepted')
                key=(cfg['windowMs'],cfg['burdenGate'])
                if key not in packed:packed[key]=prepare_prediction(saved['model'],records)
                prediction=predict_modes(saved['model'],records,prepared=packed[key])
                for mode,result in prediction.items():
                    p[mode][cid]=np.where(result['mass'][:n]>0,result['composition'][:n].argmax(1),5)
                    a[mode][cid]=[segment_metric(r,local,cfg)['accuracy']for r,local in zip(strong,result['local'][n:])]
                traces.append({'configId':cid,'acceptedExpertStatus':saved['fit']['status'],'source':str(path),
                               'structuralWindows':{mode:result['structuralWindows']for mode,result in prediction.items()}})
            for mode in MODES:
                directory=root/mode/str(o)/str(inner);directory.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(directory/'inner-predictions.npz',map=p[mode],segmentAccuracy=a[mode])
                write(directory/'complete.json',{'mapUids':[r['uid']for r in val],'strongUids':[r['uid']for r in strong],'configurationEvaluations':len(cfgs)})
            write(done.with_name('expert-reuse.json'),traces);write(done,{'expertFitsReused':len(cfgs),'modesEvaluated':MODES})
            print('supported inner',variant,o,inner,'reused expert fits',len(cfgs),flush=True)
        train=materialize([weak[i]for i in plan['train']]+[manual[i]for i in plan['strongTrain']]);test=materialize([weak[i]for i in plan['test']]+[manual[i]for i in plan['strongTest']]);nmap=len(plan['test'])
        for mode in MODES:
            # Config count is that of the raw variant; the actual mode directory
            # stores its own independently recomputed validation predictions.
            temporary=root/mode;choice,search=choose(temporary,o,weak,manual,plan,cfgs,criterion)
            directory=root/mode/str(o);write(directory/'selection.json',{'selectionUsesOuterLabels':False,**choice});write(directory/'search.json.gz',search)
            cid=choice['configId'];cfg=choice['config'];old=source/'outer'/variant/str(o)/(str(cid)+'.json.gz')
            path=old if old.exists()else outer_cache/variant/str(o)/(str(cid)+'.json.gz')
            existed=path.exists();model,trace=fit_or_load(path,train,spec,cfg)
            if trace['status']not in ['small_projected_gradient','train_prediction_stable']:raise RuntimeError('Selected outer expert unaccepted')
            outer_fits.append({'mode':mode,'outer':o,'source':str(path),'reused':existed,'fit':trace})
            profile={'model':model,'postprocess':{'physicalSupport':mode},'fit':trace,'trainingDefinition':protocol['training'],'selection':choice}
            write(directory/'outer-model.json',profile);prediction=predict_modes(model,test,[mode])[mode]
            for j,r in enumerate(test):
                if j<nmap:
                    q=prediction['composition'][j];all_maps.append({'variant':mode,'outer':o,'uid':r['uid'],'keys':r['keys'],'group':r['group'],'truth':r['coarseTags'][0],
                      'predicted':model['classes'][int(q.argmax())]if q.sum()>0 else None,'composition':q.tolist(),'weight':r['recordWeight']})
                else:
                    for w,p,weight in zip(r['windows'][str(cfg['windowMs'])],prediction['local'][j],prediction['weights'][j]):
                        all_segments.append({'variant':mode,'outer':o,'uid':r['uid'],'start':w['start'],'end':w['end'],'tag':(model['classes']+['break'])[int(p.argmax())],
                          'probability':p.tolist(),'aggregationWeight':float(weight)})
            print('supported outer',variant,mode,o,'config',cid,'new raw fit',not existed,flush=True)
    write(root/'outer-predictions.json',all_maps);write(root/'outer-segments.json',all_segments);write(root/'outer-fits.json',outer_fits)
    write(root/'complete.json',{'variants':MODES,'rawVariant':variant,'uniqueInnerExpertFitsReused':len(cfgs)*18,'innerConfigurationEvaluations':len(cfgs)*18*len(MODES),
      'newOuterExpertFits':sum(not r['reused']for r in outer_fits),'outerSelections':len(outer_fits),'scope':protocol['scope']})


def choose(root,o,weak,manual,plan,configs,criterion):
    ps=[];accuracies=[];validation=[]
    for j,s in enumerate(plan['inner']):
        directory=root/str(o)/str(j);info=read(directory/'complete.json')
        if info['mapUids']!=[weak[i]['uid']for i in s['validation']]:raise RuntimeError('Support validation ordering differs')
        with np.load(directory/'inner-predictions.npz')as saved:ps.append(saved['map']);accuracies.append(saved['segmentAccuracy'])
        validation.extend(weak[i]for i in s['validation'])
    p=np.concatenate(ps,axis=1);a=np.concatenate(accuracies,axis=1);search=[]
    for cid,cfg in enumerate(configs):
        four=source_metric(validation,p[cid],4);seven=source_metric(validation,p[cid],7);segment=float(a[cid].mean())
        score=four['macroF1']if criterion=='4k'else .25*four['macroF1']+.25*seven['macroF1']+.5*segment
        search.append({'configId':cid,'config':cfg,'score':score,'map4':four,'map7':seven,'segmentChartAccuracy':segment})
    if criterion=='4k':at=max(range(len(configs)),key=lambda i:(search[i]['score'],search[i]['map4']['accuracy'],-i))
    else:at=max(range(len(configs)),key=lambda i:(search[i]['score'],search[i]['segmentChartAccuracy'],search[i]['map4']['macroF1'],search[i]['map7']['macroF1'],-i))
    return search[at],search


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',default='weak4_only');p.add_argument('--fits',default='.cache/rework-coarse-shared-r3-fits')
    p.add_argument('--output',default='research/rework/results/coarse-supported-4k-r1');p.add_argument('--outer-fits',default='.cache/rework-coarse-supported-r1-outer');evaluate(p.parse_args())
