"""Retuned mechanism ablation: atypical LN fit is not evidence of physical rest.

Only exempt LN Wildcard from the confidence-dependent late Break disjunct.
Keep explicit gap/sparse/early pressure gates and all other original behavior.
Original 140-config arm is reused from exact saved INNER metrics; repaired arm
retunes the same complete grid. OUTER predictions replay exact hard outputs.
"""
from joint_core import *
from local_study import PLANS,mean_metrics
ROOT=Path('research/core_retune/results/wildcard-gate-r1')
OLD=Path('research/core_retune/results/native-parameters-r1')


def run():
    original=read(OLD/'protocol.json');configs=original['configs'];default=original['defaultId'];d=Data();w=Whole(d);m=w.manifest;records={r['uid']:r for r in m['records']};main=m['mainUids'];manual=m['manualUids'];protocol=dict(id='wildcard-rest-exemption-r1',configs=configs,plans=PLANS,scope='Matched complete retuning on/off LN-Wildcard confidence-only Break exemption. Same native parameters and selection as native-parameters-r1; original saved INNER tables reused, no original folds altered. Explicit physical rest/sparse gates remain. Historical data, not blind.',source={p:Path(__file__).with_name(p).read_text(encoding='utf8')for p in ['core.py','wildcard_gate_study.py']})
    if(ROOT/'protocol.json').exists()and read(ROOT/'protocol.json')!=protocol:raise ValueError('Frozen mechanism protocol changed')
    write(ROOT/'protocol.json',protocol);sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'));from model import metrics
    y=np.array([[c in r['tags']for c in m['classes']]for r in w.records]);local=[];real=[];pred=[]
    def frames(c,exempt):return d.frames({},gains={'ln':c['gain']},wild_scales={'ln':c['wildBase']/.26},confidence_floor=c['confidenceFloor'],wildcard_rest_exemption=exempt)
    for ci,c in enumerate(configs):
        fs=frames(c,True);local.append(d.local(fs)['details']);real.append(d.real(fs)['details']);pred.append(w.evaluate(fs,return_vectors=True))
        if ci%20==0:print('wildcard mechanism outputs',ci,'/',len(configs),flush=True)
    def evaluate(ci,splits):
        ls=[];ids=[]
        for split in splits:
            groups={records[main[i]]['group']for i in split['validation']};ms={manual[i]for i in split['strongValidation']};es={e['id']for e in d.extra if records[e['uid']]['group']in groups};ls.append(dict(manual=dict(details=[r for r in local[ci]if r['uid']in ms]),real=dict(details=[r for r in real[ci]if r['id']in es])));ids.extend(w.lookup[main[i]]for i in split['validation'])
        l=mean_metrics(ls);cons={n:metrics(y[ids],pred[ci][n][ids],w.weight[ids])for n in w.consumers};avg={k:float(np.mean([v[k]for v in cons.values()]))for k in ['macroF1','macroPrecision','macroRecall']};return dict(local=l,consumers=cons,jointScore=.5*l['objective']+.25*avg['macroF1']+.125*avg['macroPrecision']+.125*avg['macroRecall'])
    def key(r,supervision):
        c=r['config'];mm=r['metrics'];distance=abs(np.log(c['gain']/1.4))+abs(np.log(c['wildBase']/.26))+abs(c['confidenceFloor']-.5);return(mm['jointScore']if supervision=='joint'else mm['local']['objective'],-mm['local']['techCsConfusionMs'],-distance,-r['id'])
    selections=[];grids=[]
    for outer,plan in enumerate(PLANS):
        rows=[dict(id=ci,config=c,metrics=evaluate(ci,plan['inner']))for ci,c in enumerate(configs)];old=read(OLD/str(outer)/'inner-grid.json');choice={}
        for mode,table in [('original',old),('exempt',rows)]:
            for gain in ['fixed','free']:
                for supervision in ['local','joint']:
                    eligible=[r for r in table if gain=='free'or r['config']['gain']==1.4];choice[mode+'/'+gain+'/'+supervision]=max(eligible,key=lambda r:key(r,supervision))
        selections.append(choice);grids.append(rows);write(ROOT/str(outer)/'inner-grid.json',rows);write(ROOT/str(outer)/'selection.json',dict(usesOuterTruth=False,choices=choice))
    pooled=[]
    for ci,c in enumerate(configs):
        mm=[g[ci]['metrics']for g in grids];pooled.append(dict(id=ci,config=c,metrics=dict(jointScore=float(np.mean([r['jointScore']for r in mm])),local={k:float(np.mean([r['local'][k]for r in mm]))for k in ['objective','techCsConfusionMs']})))
    write(ROOT/'common-inner.json',dict(usesOuterTruth=False,choices={gain+'/'+supervision:max([r for r in pooled if gain=='free'or r['config']['gain']==1.4],key=lambda r:key(r,supervision))for gain in ['fixed','free']for supervision in ['local','joint']}))
    outcomes={}
    for name in selections[0]:
        mode,gain,supervision=name.split('/');ps={n:np.zeros_like(y)for n in w.consumers};ls=[];rs=[];choices=[];oldcache={}
        for outer,plan in enumerate(PLANS):
            ci=selections[outer][name]['id'];c=configs[ci];choices.append(c)
            if mode=='exempt':ll=local[ci];rr=real[ci];vectors=pred[ci]
            else:
                if ci not in oldcache:
                    fs=frames(c,False);oldcache[ci]=(d.local(fs)['details'],d.real(fs)['details'],w.evaluate(fs,return_vectors=True))
                ll,rr,vectors=oldcache[ci]
            ids=[w.lookup[main[i]]for i in plan['test']];groups={records[main[i]]['group']for i in plan['test']};ms={manual[i]for i in plan['strongTest']};es={e['id']for e in d.extra if records[e['uid']]['group']in groups};ls.extend(r for r in ll if r['uid']in ms);rs.extend(r for r in rr if r['id']in es)
            for n in w.consumers:ps[n][ids]=vectors[n][ids]
        outcomes[name]=dict(local=mean_metrics([dict(manual=dict(details=ls),real=dict(details=rs))]),manual=ls,real=rs,consumers={n:metrics(y[w.main],v[w.main],w.weight[w.main])for n,v in ps.items()},choices=choices)
    # Unretuned mechanism isolates the logical change from tuning outcomes.
    fs=frames(configs[default],True);fixed=dict(local=d.local(fs),real=d.real(fs),consumers=w.evaluate(fs));write(ROOT/'oof.json',dict(scope=protocol['scope'],metrics=outcomes,unretunedMechanism=fixed));print('WILDCARD MECHANISM COMPLETE', {n:dict(local=r['local'],F1={k:v['macroF1']for k,v in r['consumers'].items()})for n,r in outcomes.items()},flush=True)


if __name__=='__main__':run()
