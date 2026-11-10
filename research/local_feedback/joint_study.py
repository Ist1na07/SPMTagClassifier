"""One bounded joint-type study; real error passages are an explicit endpoint."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,time,itertools,argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from joint_activity import Data,Kernel,fit,PLANS,read,write,local,summary,parent
from competition import local_metrics
from reanalyze_allowed import corrected
from human_regression import coverage
from rc_action_runtime import smooth
import checkpoint_io

ROOT=Path('research/local_feedback/results/joint-r7-physical');CACHE=Path('.cache/local-feedback-joint-r7-physical');D=None
ARMS=['all','joint','local_only','whole_only','no_extra','route_only','type_only','no_guidance','no_context','product']


def configurations():
    out=[]
    for mode in ['both','route_only','type_only','no_guidance','no_context']:
        tasks=[(1.,e,w)for e,w in itertools.product([0.,1.,4.],[0.,1.,3.]if mode=='both'else[1.,3.])]
        if mode=='both':tasks.append((0.,0.,1.))
        for (strong,extra,whole),ridge in itertools.product(tasks,[3.,.3,.03]):
            out.append(dict(mode=mode,widthId=1,context=mode!='no_context',obligations=mode!='no_guidance',trainRoute=mode!='type_only',trainType=mode!='route_only',physicalEligibility=True,strongWeight=strong,extraWeight=extra,wholeWeight=whole,ridge=ridge))
    return out


def categories():
    out={'B01':['break'],'B02':['release'],'C16':['speed'],'C24':['tech'],'CTX-L':['dense chordstream','fast chordstream'],'CTX-H':['break'],'LN-L':['release'],'LN-H':['release'],'I01':['ln wildcard'],'I02':['ln wildcard']}
    for rnd in ['r4','r5','r6']:
        for name,a in read('research/frontier/annotations-'+rnd+'.json')['cases'].items():out[name]=[a['localCategory']]
    return out


def cases(d,frames,observations):
    tags=categories();out=[]
    for e in observations:
        a,b=d.records[e['index']]['offset'];v=coverage(smooth(frames[a:b]),e['start'],e['end']);allowed=tags[e['id']];out.append(dict(id=e['id'],uid=d.records[e['index']]['uid'],allowedPrimary=allowed,coverage=v,accuracy=sum(v.get(t,0.)for t in allowed),primary=max(v,key=v.get)if v else None,group=e['group']))
    return out


def local_summary(manual,real):
    values=[r['accuracy']for r in manual]+[r['accuracy']for r in real]
    return dict(accuracy=float(np.mean(values)),manualAccuracy=float(np.mean([r['accuracy']for r in manual])),realAccuracy=float(np.mean([r['accuracy']for r in real]))if real else None,observations=len(values),manualObservations=len(manual),realObservations=len(real),techToDenseMs=sum(r['techToDenseMs']for r in manual),techCsConfusionMs=sum(r['techCsConfusionMs']for r in manual))


def mean_consumers(values):
    return{k:float(np.mean([v[k]for v in values]))for k in ['macroPrecision','macroRecall','macroF1','microF1','sampleF1','exactMatch']}


def score(row):
    m=mean_consumers(list(row['consumers'].values()));return .5*row['local']['accuracy']+.25*m['macroF1']+.125*m['microF1']+.125*m['sampleF1']


def eligible(c,arm):
    if arm in ['route_only','type_only','no_guidance','no_context']:return c['mode']==arm
    if c['mode']!='both':return False
    if arm=='all':return True
    if arm=='whole_only':return c['strongWeight']==0
    if arm=='local_only':return c['strongWeight']>0 and c['wholeWeight']==0
    if arm=='product':return c['strongWeight']>0
    if arm=='joint':return c['strongWeight']>0 and c['wholeWeight']>0
    if arm=='no_extra':return c['strongWeight']>0 and c['wholeWeight']>0 and c['extraWeight']==0
    raise ValueError(arm)


def select(rows,baseline,arm):
    good=[r for r in rows if eligible(r['config'],arm)and r['local']['accuracy']>baseline['local']['accuracy']+1e-12
        and r['local']['manualAccuracy']>=baseline['local']['manualAccuracy']-.01
        and r['local']['techCsConfusionMs']<=baseline['local']['techCsConfusionMs']+1e-8
        and all(r['consumers'][n][k]>=baseline['consumers'][n][k]-.01 for n in baseline['consumers']for k in ['macroPrecision','macroRecall','macroF1','microF1'])
        and(arm!='product'or r['mechanismPass'])]
    peak=max([score(baseline)]+[score(r)for r in good]);near=[r for r in good if score(r)>=peak-.002]
    if not near:return baseline
    return min(near,key=lambda r:(-r['config']['ridge'],-score(r),r['config']['extraWeight'],r['configId']))


def initialize():
    global D
    D=Data()
    # Only the fixed 2-s architecture and its two declared feature ablations
    # are needed. Retain both guide/context controls, drop unrelated copies.
    D.inputs={key:v for key,v in D.inputs.items()if key[0]==1}


def scope_baseline(d,plan):
    ids=np.concatenate([d.main[s['validation']]for s in plan['inner']]);manual=[];real=[];frames=[f for _,f in d.frame_records]
    for split in plan['inner']:
        manual.extend(corrected(local_metrics(d,d.manual[split['strongValidation']])));groups={d.records[i]['group']for i in d.main[split['validation']]};real.extend(cases(d,frames,[e for e in d.extra if e['group']in groups]))
    return dict(configId=-1,config={},local=local_summary(manual,real),consumers={n:parent.metrics(d.y[ids],p[ids],d.weight[ids])for n,p in d.fixed.items()},mechanismPass=False)


def job(o):
    d=D;folder=ROOT/str(o);cache=CACHE/str(o);cache.mkdir(parents=True,exist_ok=True)
    if(folder/'inner-complete.json').exists():return o,'resumed'
    begin=time.monotonic();plan=PLANS[o];va=np.concatenate([d.main[s['validation']]for s in plan['inner']]);rows=[];kernels={};warm={};previous=None
    for ci,c in enumerate(configurations()):
        key=c['mode'];path=cache/f'config-{ci}.json'
        if key!=previous:kernels={};warm={};previous=key
        if checkpoint_io.exists(path):record=checkpoint_io.read(path)
        else:
            inner=[];pred={n:[]for n in d.consumer_names};manual=[];real=[];mechanisms=[]
            for j,split in enumerate(plan['inner']):
                if j not in kernels:kernels[j]=Kernel(d,d.main[split['train']],d.manual[split['strongTrain']],c)
                k=kernels[j];k.set_config(c);wk=(j,c['strongWeight'],c['extraWeight'],c['wholeWeight']);seed=warm.get(wk);seed_source='same-scope stronger-ridge continuation'if seed is not None else'random'
                previous_path=Path('.cache/local-feedback-joint-r7')/str(o)/f'config-{ci}.json'
                if checkpoint_io.exists(previous_path):
                    older=checkpoint_io.read(previous_path);old_config=dict(c);old_config.pop('physicalEligibility')
                    if older['row']['config']!=old_config:raise ValueError('Old joint warm seed scope/config mismatch')
                    seed=np.array(older['inner'][j]['theta']);seed_source='same-scope/config previous unconstrained physical model, INITIALIZATION ONLY'
                theta,receipt=fit(k,warm=seed);receipt['secondStartSource']=seed_source;warm[wk]=theta
                frames,p,support=k.frames(theta);manual.extend(local(d,d.manual[split['strongValidation']],frames));groups={d.records[i]['group']for i in d.main[split['validation']]};real.extend(cases(d,frames,[e for e in d.extra if e['group']in groups]));mechanisms.append(cases(d,frames,[e for e in d.extra if e['group']is None]))
                for name in pred:pred[name].append(d.consume(support,p,name)[d.main[split['validation']]])
                inner.append(dict(theta=theta.tolist(),receipt=receipt))
            physics=all(r['primary']in r['allowedPrimary']for scope in mechanisms for r in scope if r['id']in ['B01','B02','C16'])
            row=dict(configId=ci,config=c,local=local_summary(manual,real),consumers={n:parent.metrics(d.y[va],np.concatenate(q),d.weight[va])for n,q in pred.items()},mechanismPass=physics)
            record=dict(row=row,inner=inner,manual=manual,real=real,mechanisms=mechanisms,predictions={n:np.concatenate(q).tolist()for n,q in pred.items()});checkpoint_io.write(path,record)
        for j,state in enumerate(record['inner']):warm[j,c['strongWeight'],c['extraWeight'],c['wholeWeight']]=np.array(state['theta'])
        rows.append(record['row'])
        if ci%6==0:print('joint INNER',o,ci,'/',len(configurations()),round(time.monotonic()-begin,1),'seconds',flush=True)
    base=scope_baseline(d,plan);choices={arm:select(rows,base,arm)for arm in ARMS};write(folder/'inner-grid.json',rows);write(folder/'selection.json',dict(usesOuterTruth=False,baseline=base,choices=choices));write(folder/'inner-complete.json',dict(configs=len(rows),innerFits=len(rows)*3,seconds=time.monotonic()-begin));return o,read(folder/'inner-complete.json')


def declare():
    protocol=dict(id='joint-r7-physical',researchBaseline='v1.1.0',productControl='v1.2.3',configs=configurations(),plans=PLANS,
        hypothesis='R6 cannot correct wrong within-family rankings. Jointly learn3-way activity and two real-expert fixed-affinity-budget conditional type heads; no trainable whole classifier. TRAIN-only input whitening, Helmert contrasts, explicit ridge, no teacher duplication. Fixed2s physical input/500ms output architecture motivated by short-prefix and release-context probes; window is a bounded modeling choice, not claimed optimally tuned.',
        localEndpoint='Report original6manual and8actively queried real passages separately, plus their equal-passage macro. Real source E/F/G/I groups are TRAIN-only for fitting and validation-only for local selection; no source title/BeatmapID/SetID overlap with originalmanual6. I01/I02 primary LNWildcard per corrected user answers, component grades retained in source records. Real active cases are not a fresh random benchmark. Pure F02 Speed remains primary, Tech component not invented as a numeric target.',
        supervision='Original manual allowed-type segments use exact overlap. Active source/constructed answers are burden-normalized target-interval bags; their label is NOT copied to each child. Constructed constraints are reported separately, not validation accuracy. No extra means no extra TRAIN loss; all arms share the same held-out source endpoint. Product selection additionally requires TRAIN-dev B01Break/B02Release/C16Speed plurality, so product selection is reported separately from the matched ablation choices.',
        selection='Score.5combinedLocal+.25meanMacroF1+.125meanMicroF1+.125meanSampleF1. Positive combined local gain, originalmanual within1pp, no Tech/CS confusion regression, every actualconsumer macroP/R/F1/micro within1pp on INNER. Near.002 prefer stronger ridge. Baseline if none. Primary candidates cannot be declared better by comparing all-TRAIN replay. Common parameters saved before OUTER. Product adoption additionally requires real-case, render, native parity and final three-consumer acceptance.',
        comparisons='102 configurations x18=1836INNER fits, two starts each. Full architecture27local/joint+3whole-only. Each freeze-route, freeze-types, no-guidance, no-context control independently retunes ridge3/.3/.03,extra0/1/4,whole1/3. No-extra and local-only reselect active grid. Both starts original coordinate zero/random or exact-scope stronger-ridge continuation. Every accepted original gradient<=1e-5.',
        numerical='Compiled existing Numba, no new install. Same fixed-budget equation/Jacobian as R2; safeguarded Newton must contract actual residual byhalf or bisect. Independent original bisection oracle on saturated scales.001..500 maxdifference2.56e-13. Accepted all-TRAIN state exact objective replay gradient1.59e-7; projection speed only, no model change. Dense BFGS preconditioner not selected after cost comparison. Family bag objective is auxiliary; actual MAP physical coverage in validation, fine supports have real integral gradient. Native SPM curves reused, no scorer calls.',
        physicalRepair='Actual native adapter on the no-LN CTX-H map exposed impossible LN/Hybrid from an unrestricted counterfactual router. Rice requires a physical head in the local window; LN requires LN heads, union-hold occupancy or LN tails. Break always available. No tuned threshold: exact positive physical activity. Mask impossible family logits before softmax, and normalize fine TRAIN costs over physically available candidates. All variants share these semantic constraints. Old partial R7 fits are retained and used only as matched same-scope/config second-start initializers; they are not accepted as new results. Refit every affected configuration. The original v1.2.3 control is unchanged.',
        resourceScope='The saved unrestricted INNER trial is a cost/reference receipt, not evidence of physical-model performance. Current physical eligibility has a separate combined-term finite-difference and impossible-LN extreme-bias test. No physical-model outcomes have been read before this declaration.',
        resource=read('research/local_feedback/results/joint-compiled-inner-resource-r7.json'),
        sources={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['joint_activity.py','fast_projection.py','joint_study.py','checkpoint_io.py','test_joint.py']})
    path=ROOT/'protocol.json'
    if path.exists()and read(path)!=protocol:raise ValueError('Frozen joint protocol changed')
    if not path.exists():write(path,protocol)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=6);parser.add_argument('--declare-only',action='store_true');args=parser.parse_args();declare()
    if not args.declare_only:
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize)as pool:
            for f in as_completed([pool.submit(job,o)for o in range(6)]):print('joint INNER complete',f.result(),flush=True)
        write(ROOT/'inner-complete.json',dict(scopes=6,configs=len(configurations()),innerFits=18*len(configurations())))
