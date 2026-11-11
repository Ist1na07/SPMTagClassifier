"""Actual hard-output joint tuning of three original, effective parameters.

No high-dimensional refit. Shared LN model gain, existing LN wildcard base,
and original late easyAmbiguous confidence threshold. Original W/b, input
features, all other rules and integral gates remain frozen.
"""
from joint_core import *
from local_study import PLANS,mean_metrics,subset_metric
import itertools

ROOT=Path('research/core_retune/results/native-parameters-r1')


def run():
    d=Data();w=Whole(d);m=w.manifest;records={r['uid']:r for r in m['records']};main=m['mainUids'];manual=m['manualUids'];configs=[dict(gain=g,wildBase=b,confidenceFloor=f)for g,b,f in itertools.product([.35,.5,.7,1.,1.4,2.,2.8],[.13,.195,.26,.325,.39],[.25,.35,.5,.65])]
    default=configs.index(dict(gain=1.4,wildBase=.26,confidenceFloor=.5));protocol=dict(id='native-parameters-r1',researchBaseline='v1.1.0',productControl='v1.2.4',configs=configs,defaultId=default,plans=PLANS,
      selection='Local: .75 original-six per-chart accuracy + .25 real-eight per-case accuracy when present. Joint: .5 local score + .25 mean three-consumer macroF1 + .125 mean macroPrecision + .125 mean macroRecall. Pooled INNER predictions; fewer Tech/CS errors then nearer original parameters break ties. No metric floor excludes candidates. Four independently selected arms: gain fixed/free x local/joint. Save common INNER choice before OUTER evaluation. Historical data exposure acknowledged.',
      scope='Natural parents only, fixed original W/b and baseline aggregate prior. Exact original score normalization, wildcard demotion, native late confidence gate and smoothing. No smooth32 surrogate or approximated gradient. All three integral paths use original physical fields; verified full349 baseline parity. Local source labels included only in their source groups.',
      source={p:Path(__file__).with_name(p).read_text(encoding='utf8')for p in ['core.py','joint_core.py','native_parameter_study.py']})
    if(ROOT/'protocol.json').exists()and read(ROOT/'protocol.json')!=protocol:raise ValueError('Frozen protocol changed')
    write(ROOT/'protocol.json',protocol)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'));from model import metrics
    y=np.array([[c in r['tags']for c in m['classes']]for r in w.records]);all_local=[];all_real=[];all_pred=[]
    for ci,c in enumerate(configs):
        frames=d.frames({},gains={'ln':c['gain']},wild_scales={'ln':c['wildBase']/.26},confidence_floor=c['confidenceFloor']);all_local.append(d.local(frames)['details']);all_real.append(d.real(frames)['details']);pred=w.evaluate(frames,return_vectors=True);all_pred.append(pred)
        if ci%20==0:print('native hard parameter outputs',ci,'/',len(configs),flush=True)
    write(ROOT/'outputs-complete.json',dict(configurations=len(configs),nativeParents=len(d.parents),source=protocol['scope']))

    def evaluate(ci,splits):
        ls=[];ids=[]
        for split in splits:
            groups={records[main[i]]['group']for i in split['validation']};ms={manual[i]for i in split['strongValidation']};es={e['id']for e in d.extra if records[e['uid']]['group']in groups};loc=[r for r in all_local[ci]if r['uid']in ms];real=[r for r in all_real[ci]if r['id']in es];ls.append(dict(manual=dict(details=loc),real=dict(details=real)));ids.extend(w.lookup[main[i]]for i in split['validation'])
        local=mean_metrics(ls);consumers={n:metrics(y[ids],all_pred[ci][n][ids],w.weight[ids])for n in w.consumers};avg={k:float(np.mean([v[k]for v in consumers.values()]))for k in ['macroF1','macroPrecision','macroRecall']};score=.5*local['objective']+.25*avg['macroF1']+.125*avg['macroPrecision']+.125*avg['macroRecall'];return dict(local=local,consumers=consumers,jointScore=score)

    def key(row,arm):
        c=row['config'];metric=row['metrics'];distance=abs(np.log(c['gain']/1.4))+abs(np.log(c['wildBase']/.26))+abs(c['confidenceFloor']-.5);return(metric['jointScore']if arm.endswith('joint')else metric['local']['objective'],-metric['local']['techCsConfusionMs'],-distance,-row['id'])

    arms=['fixed/local','fixed/joint','free/local','free/joint'];selections=[];grids=[]
    for outer,plan in enumerate(PLANS):
        rows=[dict(id=ci,config=c,metrics=evaluate(ci,plan['inner']))for ci,c in enumerate(configs)];choices={arm:max([r for r in rows if arm.startswith('free')or r['config']['gain']==1.4],key=lambda r:key(r,arm))for arm in arms};selections.append(choices);grids.append(rows);write(ROOT/str(outer)/'inner-grid.json',rows);write(ROOT/str(outer)/'selection.json',dict(usesOuterTruth=False,choices=choices))
    pooled=[]
    for ci,c in enumerate(configs):
        met=[g[ci]['metrics']for g in grids];pooled.append(dict(id=ci,config=c,metrics=dict(jointScore=float(np.mean([r['jointScore']for r in met])),local={k:float(np.mean([r['local'][k]for r in met]))for k in ['objective','techCsConfusionMs']})))
    common={arm:max([r for r in pooled if arm.startswith('free')or r['config']['gain']==1.4],key=lambda r:key(r,arm))for arm in arms};write(ROOT/'common-selection.json',dict(usesOuterTruth=False,choices=common,grid=pooled));write(ROOT/'inner-complete.json',dict(configurations=len(configs),innerScopes=18))
    result={}
    for arm in arms+['baseline']:
        local=[];real=[];outer_pred={n:np.zeros_like(y)for n in w.consumers};chosen=[]
        for outer,plan in enumerate(PLANS):
            ci=default if arm=='baseline'else selections[outer][arm]['id'];ms={manual[i]for i in plan['strongTest']};groups={records[main[i]]['group']for i in plan['test']};es={e['id']for e in d.extra if records[e['uid']]['group']in groups};local.extend(r for r in all_local[ci]if r['uid']in ms);real.extend(r for r in all_real[ci]if r['id']in es);ids=[w.lookup[main[i]]for i in plan['test']]
            for name in w.consumers:outer_pred[name][ids]=all_pred[ci][name][ids]
            chosen.append(configs[ci])
        l=mean_metrics([dict(manual=dict(details=local),real=dict(details=real))]);ids=w.main;result[arm]=dict(local=l,localDetails=local,realDetails=real,consumers={n:metrics(y[ids],p[ids],w.weight[ids])for n,p in outer_pred.items()},chosen=chosen)
    write(ROOT/'oof.json',dict(scope=protocol['scope'],metrics=result));print('NATIVE PARAMETER COMPLETE', {k:dict(local=v['local'],F1={n:r['macroF1']for n,r in v['consumers'].items()})for k,v in result.items()},flush=True)


if __name__=='__main__':run()
