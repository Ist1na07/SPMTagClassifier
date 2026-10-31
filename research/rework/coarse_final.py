"""Select final configurations from repeated inner predictions, then fit once.

Never reads outer predictions, outer metrics or selected outer profiles.
The resulting all-training models do not supply independent test scores.
"""
import argparse,subprocess
from collections import Counter
from coarse_data import load_headers,plans
from coarse_search import read,write,configurations,source_metric,fit_or_load
from coarse_model import np,Path,SPECS,materialize,training_rows


TARGETS={'4k':{'variant':'weak4_only','mode':'both','criterion':'4k'},
         '6k':{'variant':'full53','mode':'both','criterion':'shared'}}


def choose(root,target,weak,manual,ps):
    variant=target['variant'];mode=target['mode'];cfgs=configurations(variant);validation=[];predictions=[];accuracies=[]
    map_counts=Counter();manual_counts=Counter()
    for o,plan in enumerate(ps):
        for i,split in enumerate(plan['inner']):
            directory=root/variant/mode/str(o)/str(i);info=read(directory/'complete.json')
            expected=[weak[j]['uid']for j in split['validation']];strong=[manual[j]['uid']for j in split['strongValidation']]
            if info['mapUids']!=expected or info['strongUids']!=strong:raise RuntimeError('Inner prediction scope differs')
            with np.load(directory/'inner-predictions.npz')as saved:
                p=saved['map'];a=saved['segmentAccuracy']
                if p.shape!=(len(cfgs),len(expected))or a.shape!=(len(cfgs),len(strong)):raise RuntimeError('Inner matrix shape differs')
                predictions.append(p);accuracies.append(a)
            validation.extend({**weak[j],'recordWeight':weak[j]['recordWeight']/5}for j in split['validation'])
            map_counts.update(expected);manual_counts.update(strong)
    if set(map_counts)!={r['uid']for r in weak}or set(map_counts.values())!={5}:raise RuntimeError('Expected five held-out inner predictions per weak record')
    if set(manual_counts)!={r['uid']for r in manual}or set(manual_counts.values())!={5}:raise RuntimeError('Expected five held-out inner predictions per manual record')
    p=np.concatenate(predictions,axis=1);a=np.concatenate(accuracies,axis=1);search=[]
    for cid,cfg in enumerate(cfgs):
        four=source_metric(validation,p[cid],4);seven=source_metric(validation,p[cid],7);segment=float(a[cid].mean())
        score=four['macroF1']if target['criterion']=='4k'else .25*four['macroF1']+.25*seven['macroF1']+.5*segment
        search.append({'configId':cid,'config':cfg,'score':score,'map4':four,'map7':seven,'segmentChartAccuracy':segment})
    if target['criterion']=='4k':at=max(range(len(cfgs)),key=lambda i:(search[i]['score'],search[i]['map4']['accuracy'],-i))
    else:at=max(range(len(cfgs)),key=lambda i:(search[i]['score'],search[i]['segmentChartAccuracy'],search[i]['map4']['macroF1'],search[i]['map7']['macroF1'],-i))
    return search[at],search,{'weakRecords':len(map_counts),'manualRecords':len(manual_counts),'heldOutInnerPredictionsPerRecord':5,'weakPredictionRows':len(validation),
      'scope':'repeated inner CV for all-data configuration choice; roughly5/9 training fraction per expert, not new nested outer performance or a five-model vote'}


def run(args):
    source=Path(args.source);complete=read(source/'complete.json');root=Path(args.output);weak,manual,_=load_headers();ps=plans(weak,manual)
    if complete['uniqueInnerExpertFitsReused']!=26892:raise RuntimeError('Full declared supported ablation study missing')
    protocol={'id':'coarse-final-inner-selection-r1','targets':TARGETS,'source':str(source),'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['coarse_final.py','coarse_model.py','coarse_data.py']},
      'selection':'all effective configs; accumulate original per-record physical weights/5 on five actual held-out inner predictions; no outer model/score/prediction read; no outer-ablation winner',
      'records':[{'uid':r['uid'],'group':r['group'],'keys':r['keys'],'tags':r['coarseTags'],'weight':r['recordWeight']}for r in weak],
      'manual':[{'uid':r['uid'],'annotations':r['annotations']}for r in manual],'plans':ps}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise RuntimeError('Final selection protocol changed')
    write(root/'protocol.json',protocol)
    headers=weak+manual;records=None;selected={}
    for name,target in TARGETS.items():
        choice,search,scope=choose(source,target,weak,manual,ps);directory=root/name
        write(directory/'selection.json',{'target':target,'readsOuterPredictions':False,'innerPredictionScope':scope,**choice});write(directory/'search.json',search)
        if records is None:records=materialize(headers)
        model,trace=fit_or_load(directory/'raw-fit.json.gz',records,SPECS[target['variant']],choice['config'])
        if trace['status']not in ['small_projected_gradient','train_prediction_stable']:raise RuntimeError('All-training optimizer has not met declared stopping condition')
        actual=training_rows(headers,SPECS[target['variant']])
        profile={'model':model,'postprocess':{'physicalSupport':target['mode']},'fit':trace,'selection':choice,'innerPredictionScope':scope,
          'trainingUids':[r['uid']for r in actual],'evidence':{'allTrainingFit':True,'humanSixKeyAccuracy':None,'reportedAccuracySource':'matching nested outer study; this final fit is not an independent test',
          'fineClassesValidated':False,'taxonomy':'five supported coarse morphologies; 4K source stamina means jump/handstream chordstream bucket'}}
        write(directory/'profile.json',profile);selected[name]={'target':target,'configId':choice['configId'],'config':choice['config'],'fit':trace['status'],'profile':str(directory/'profile.json')}
        print('final all-training coarse profile',name,selected[name],flush=True)
    write(root/'complete.json',{'profiles':selected,'selectedOnlyFromInnerPredictions':True,'newNestedTestScores':False})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='research/rework/results/coarse-supported-all-r1')
    p.add_argument('--output',default='research/rework/results/coarse-final-r1');run(p.parse_args())
