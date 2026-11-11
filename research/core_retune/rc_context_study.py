"""Isolate short RC evidence from harmful re-routing/rest changes.

One actual core input parameter: convex blend of parent/local vectors. No
extra learned head, invented class, fixed-affinity budget or LN modification.
"""
from core import *
from local_study import PLANS,subset_metric,mean_metrics

ROOT=Path('research/core_retune/results/rc-context-source-r1')


def run():
    d=Data('.cache/core-retune-rc-context-r1.jsonl.gz',frozen_routes=True);m=read('.cache/local-feedback-activity-r6b/manifest.json');records={r['uid']:r for r in m['records']};main=m['mainUids'];manual=m['manualUids'];configs=[dict(width=w,alpha=a)for w in [1000,2000,4000]for a in [.25,.5,.75,1.]]
    protocol=dict(id='rc-context-source-r1',configs=configs,plans=PLANS,baseline='v1.2.4',scope='Source-group local mechanism ablation, no whole feedback yet. Frozen old rest/family and LN readout; original RC softmax W/b evaluated on blended parent/local original 56D input. Original rule attributes and wildcard support unchanged. No label boundaries.',selection='Same .75 manual accuracy + .25 real local accuracy; all INNER choices saved before OUTER; fewer Tech/CS errors breaks objective ties, then smaller alpha, then ID. Baseline eligible.',source={p:Path(__file__).with_name(p).read_text(encoding='utf8')for p in ['core.py','extract_rc_context.cjs','rc_context_study.py']})
    if(ROOT/'protocol.json').exists()and read(ROOT/'protocol.json')!=protocol:raise ValueError('Frozen protocol changed')
    write(ROOT/'protocol.json',protocol);frames={-1:d.baseline};b=d.blocks['rc'];model=d.models['rc'];W=np.array(model['W']).reshape(len(model['classes']),-1)
    for ci,c in enumerate(configs):
        x=d.x.copy()
        for j,p in enumerate(d.parents):
            if p['baseline']['family']=='rc':x[j]=(1-c['alpha'])*x[j]+c['alpha']*np.array(p['localVectors'][str(c['width'])])
        z=(x-np.array(model['mean']))/np.array(model['std']);delta=(z-b['z'])@W.T;frames[ci]=d.frames({'rc':delta})
    # Splitting an old parent alone must not change its actual label coverage.
    old=Data();oldscore=old.local(old.baseline);newscore=d.local(d.baseline)
    if abs(oldscore['accuracy']-newscore['accuracy'])>1e-12 or oldscore['techCsConfusionMs']!=newscore['techCsConfusionMs']:raise ValueError('Child baseline changed')
    def validation(split):
        groups={records[main[i]]['group']for i in split['validation']};return[manual[i]for i in split['strongValidation']],[e['id']for e in d.extra if records[e['uid']]['group']in groups]
    def key(r):return(r['metrics']['objective'],-r['metrics']['techCsConfusionMs'],-r['config']['alpha']if r['config']else 0,-r['id'])
    pooled=[]
    for outer,plan in enumerate(PLANS):
        rows=[]
        for ci in [-1]+list(range(len(configs))):
            ls=[subset_metric(d,frames[ci],*validation(split))for split in plan['inner']];rows.append(dict(id=ci,config=configs[ci]if ci>=0 else None,metrics=mean_metrics(ls),details=ls))
        choices={arm:max([r for r in rows if arm=='all'or r['id']==-1 or r['config']['width']==int(arm)],key=key)for arm in ['all','1000','2000','4000']};write(ROOT/str(outer)/'inner-grid.json',rows);write(ROOT/str(outer)/'selection.json',dict(usesOuterTruth=False,choices=choices));pooled.append(rows)
    common=[]
    for j in range(len(configs)+1):
        rows=[p[j]for p in pooled];common.append(dict(id=rows[0]['id'],config=rows[0]['config'],metrics=mean_metrics([v for r in rows for v in r['details']])))
    write(ROOT/'common-selection.json',dict(usesOuterTruth=False,choice=max(common,key=key),grid=common))
    outcome={arm:[]for arm in ['all','1000','2000','4000']};baseline=[]
    for outer,plan in enumerate(PLANS):
        split=dict(validation=plan['test'],strongValidation=plan['strongTest']);ms,es=validation(split);baseline.append(subset_metric(d,d.baseline,ms,es));choices=read(ROOT/str(outer)/'selection.json')['choices'];row={}
        for arm,c in choices.items():row[arm]=subset_metric(d,frames[c['id']],ms,es);outcome[arm].append(row[arm])
        write(ROOT/str(outer)/'outer.json',row)
    result=dict(scope='Historical source-group nested local selection; unchanged original RC core, one blend parameter. Not complete product acceptance.',baseline=dict(metrics=mean_metrics(baseline),details=baseline),arms={arm:dict(metrics=mean_metrics(v),details=v)for arm,v in outcome.items()});write(ROOT/'oof.json',result);print('RC CONTEXT RESULT',result['baseline']['metrics'],{k:v['metrics']for k,v in result['arms'].items()},flush=True)


if __name__=='__main__':run()
