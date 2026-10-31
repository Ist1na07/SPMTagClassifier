"""All-data joint choice for the opt-in LN candidate; stable 7K stays mature."""
import argparse,subprocess
from ln_adapter_search import read,write,Path,np,load,encode,MAP_CLASSES,CONFIGS,fit_or_load,zero_model
from native_search import f1_counts


def run(args):
    source=Path(args.source);root=Path(args.output);parent=read(source/'protocol.json');read(source/'complete.json')
    weak,manual,_=load();posts=parent['postPolicies']['ln_protected'];variant='geometry3';count=31*len(posts)
    tp=np.zeros((count,len(MAP_CLASSES)));fp=tp.copy();fn=tp.copy();accuracy=np.zeros(count);seen={r['uid']:0 for r in weak};strong_seen={r['uid']:0 for r in manual}
    protocol={'id':'ln-final-inner-selection-r1','source':str(source),'primary':{'variant':variant,'extraGate':'ln_protected'},
      'purpose':'opt-in research candidate only; main outer performance does not justify replacing stable7K',
      'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'source':str(source),
      'selection':'all31heads cross512policies, repeated inner held-out predictions; record weights/5, manual charts/5; no outer predictions, scores or selected outer profiles read',
      'sourceSnapshot':Path(__file__).read_text(encoding='utf8'),'parentProtocol':parent}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise RuntimeError('Final LN selection protocol changed')
    write(root/'protocol.json',protocol)
    for o,plan in enumerate(parent['plans']):
        rows=[weak[i]for i in plan['train']];strong=[manual[i]for i in plan['strongTrain']]
        for r in rows:seen[r['uid']]+=1
        for r in strong:strong_seen[r['uid']]+=1
        y=encode(rows);w=np.array([r['recordWeight']/5 for r in rows])
        with np.load(source/'ln_protected'/variant/str(o)/'inner-predictions.npz')as saved:
            p=saved['map'];a=saved['segmentAccuracy'];assert p.shape==(count,len(rows),len(MAP_CLASSES))and a.shape==(count,len(strong))
            for start in range(0,count,128):
                end=min(count,start+128);b=p[start:end]
                tp[start:end]+=(b*y[None]*w[None,:,None]).sum(1);fp[start:end]+=(b*(1-y[None])*w[None,:,None]).sum(1);fn[start:end]+=((1-b)*y[None]*w[None,:,None]).sum(1)
            accuracy+=a.sum(1)/5/len(manual)
    if set(seen.values())!={5}or set(strong_seen.values())!={5}:raise RuntimeError('Each record must have five actual held-out inner predictions')
    macro=f1_counts(tp,fp,fn);joint=.5*(macro+accuracy);at=max(range(count),key=lambda i:(joint[i],accuracy[i],macro[i],-i));head,pid=divmod(at,len(posts));policy=posts[pid]
    choice={'head':head,'policyId':pid,'policy':policy,'fitConfig':CONFIGS[head-1]if head else None,
      'inner':{'joint':float(joint[at]),'mapMacroF1':float(macro[at]),'segmentAccuracy':float(accuracy[at])},'readsOuterPredictions':False}
    write(root/'selection.json',choice);write(root/'search.json.gz',[{'head':i//len(posts),'policy':i%len(posts),'joint':float(joint[i]),'mapMacroF1':float(macro[i]),'segmentAccuracy':float(accuracy[i])}for i in range(count)])
    if head:model,trace=fit_or_load(root/'raw-fit.json.gz',weak,manual,variant,CONFIGS[head-1])
    else:model=zero_model(variant);trace={'status':'frozen_archived_prior'}
    write(root/'profile.json',{'model':model,'fit':trace,'policy':policy,'extraGate':'ln_protected','trainingUids':[r['uid']for r in weak+manual],
      'evidence':{'historicalPriorExposure':'unknown','allTrainingFit':True,'defaultReplacementAccepted':False,'meaning':'all-data fit for explicit opt-in use, not a new independent test score'}})
    write(root/'complete.json',{'selection':choice,'uniqueAllDataLearnedFits':int(bool(head)),'fitStatus':trace['status']});print(choice,trace['status'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='research/rework/results/ln-extra-gate-r1');p.add_argument('--output',default='research/rework/results/ln-final-r1');run(p.parse_args())
