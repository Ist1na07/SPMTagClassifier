"""Bounded joint retuning of rest gates, with each sensor arm independently selected."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core_retune'))
from joint_core import *

ROOT=Path('research/break_pressure/results/tune-r1')
PLANS=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']


def configs(sensor):
    recipes=[dict(quantile=None,fraction=None,target='none')]
    if sensor:recipes += [dict(quantile=q,fraction=f,target=t)for q in [.5,.75,.9]for f in [.25,.5,.75]for t in ['low','low+late']]
    return[dict(**p,confidenceFloor=c,lowNpsScale=n)for p in recipes for c in [.35,.5,.65]for n in [.75,1.,1.25]]


class Study:
    def __init__(self):
        self.d=Data();self.w=Whole(self.d)
        with gzip.open('.cache/break-pressure-input-r1.jsonl.gz','rt',encoding='utf8')as f:self.rows=[json.loads(s)for s in f]
        self.frames=[s for r in self.rows for s in r['frames']]
        self.baseline=[s for r in self.rows for s in r['baseline']]
        if len(self.frames)!=len(self.d.parents):raise ValueError('Parent alignment')
        self.extra=self.d.extra
        self.cached={}

    def predict(self,c,name):
        out=[]
        for r in self.rows:
            ps=r['parents'];fs=[dict(s)for s in r['frames']];nps=sorted(p['f']['nps']for p in ps if p['extra']['rowCount']);mapn=nps[len(nps)//2]if nps else 0
            alln=sorted(p['f']['nps']for p in ps);median=alln[len(alln)//2]if alln else 0
            protection=[False]*len(ps)
            if c['quantile']is not None:
                entry=next(v for v in r['pressure'][name]if v['quantile']==c['quantile'])
                protection=[v['reference']>0 and v['highMs']>0 and v['highFraction']>=c['fraction']for v in entry['segments']]
            for j,p in enumerate(ps):
                f=p['f'];ex=p['extra'];dur=p['end']-p['start'];n=c['lowNpsScale']
                low=dur>=2500 and f['nps']<mapn and ((f['nps']<18.5*n and ex['segIvLogStd']>.41)or((f['nps']<mapn*.85 or ex['rowCount']*1000/dur<16)and f['chordMean']<1.9 and ex['vibroFraction']>.3 and ex['segRhythmChangeRate']<.1)or(f['nps']<22.5*n and ex['segIvLogStd']>.25 and ex['vibroFraction']<.45 and f['lnWeightRatio']<.5))
                if low and not protection[j]and fs[j]['family']!='break':fs[j].update(tag='break',family='break',confidence=1,secondary=None)
            active=[j for j,s in enumerate(fs)if s['family']!='break'];diff=[p['f']['nps']*(.5+p['f']['chordMean']/3)*(.6+p['f']['jackSpeedBpm']/150)for p in ps];ordered=sorted(diff[j]for j in active);p15=ordered[int(len(ordered)*.15)]if ordered else float('inf')
            for j in active:
                s=fs[j];p=ps[j];n=p['f']['nps'];dur=p['end']-p['start']
                easy=s['tag']!='ln wildcard'and s['confidence']<c['confidenceFloor']and diff[j]<=max(p15,1)and n<mapn
                sparse=dur<20000 and 0<n<median*.4 and n<4
                if(easy and not(c['target']=='low+late'and protection[j]))or sparse:s.update(tag='break',family='break',secondary=None)
            out.extend(fs)
        return out

    def breaks(self,frames,manual):
        tp=fp=fn=0.
        for uid in manual:
            a,b=self.d.offset[uid]
            for ann in self.d.records[uid]['annotations']:
                if not ann['tags']:continue
                truth='break'in ann['tags']
                for s in frames[a:b]:
                    dt=max(0,min(ann['end'],s['end'])-max(ann['start'],s['start']));pred=s['tag']=='break';tp+=dt*(truth and pred);fp+=dt*(not truth and pred);fn+=dt*(truth and not pred)
        p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
        return dict(precision=p,recall=r,f1=2*p*r/max(1e-30,p+r),falseBreakMs=fp,missedBreakMs=fn,trueBreakMs=tp)

    def uncached_score(self,frames,name,ids,manual,real):
        local=self.d.local(frames,manual);b=self.breaks(frames,manual);rr=[r for r in self.d.real(frames)['details']if r['id']in real];ra=float(np.mean([r['accuracy']for r in rr]))if rr else None
        whole=self.w.evaluate(frames,ids)[name];localScore=.3*local['accuracy']+.15*b['f1']+(.15*ra if ra is not None else .15*local['accuracy']);score=localScore+.4*whole['macroF1']
        return dict(score=score,manual=local,breaks=b,real=dict(accuracy=ra,details=rr),whole=whole)

    def score(self,frames,name,ids,manual,real):
        from model import metrics
        key=id(frames)
        if key not in self.cached:
            self.cached[key]=(frames,self.d.local(frames)['details'],self.d.real(frames)['details'],{u:self.breaks(frames,[u])for u in self.d.manual},self.w.evaluate(frames,return_vectors=True))
        _,locals_,reals,breaks_,vectors=self.cached[key];lr=[r for r in locals_ if r['uid']in manual];rr=[r for r in reals if r['id']in real]
        local=dict(accuracy=float(np.mean([r['accuracy']for r in lr])),techToDenseMs=sum(r['techToDenseMs']for r in lr),techCsConfusionMs=sum(r['techCsConfusionMs']for r in lr),details=lr)
        tp=sum(breaks_[u]['trueBreakMs']for u in manual);fp=sum(breaks_[u]['falseBreakMs']for u in manual);fn=sum(breaks_[u]['missedBreakMs']for u in manual);p=tp/max(1,tp+fp);r=tp/max(1,tp+fn)
        b=dict(precision=p,recall=r,f1=2*p*r/max(1e-30,p+r),falseBreakMs=fp,missedBreakMs=fn,trueBreakMs=tp)
        ra=float(np.mean([r['accuracy']for r in rr]))if rr else None;classes=self.w.manifest['classes'];y=np.array([[c in r['tags']for c in classes]for r in self.w.records]);whole=metrics(y[ids],vectors[name][ids],self.w.weight[ids]);score=.3*local['accuracy']+.15*b['f1']+(.15*ra if ra is not None else .15*local['accuracy'])+.4*whole['macroF1']
        return dict(score=score,manual=local,breaks=b,real=dict(accuracy=ra,details=rr),whole=whole)

    def scope(self,s,test=False):
        ids=[self.w.main[i]for i in s['validation'if test else'train']];groups={self.w.records[i]['group']for i in ids};manual=[self.d.manual[i]for i in s['strongValidation'if test else'strongTrain']];real=[e['id']for e in self.extra if self.d.records[e['uid']]['group']in groups]
        return ids,manual,real


def run():
    st=Study();base=dict(quantile=None,fraction=None,target='none',confidenceFloor=.5,lowNpsScale=1.)
    check=st.predict(base,'actions')
    if[(s['tag'],s['family'])for s in check]!=[(s['tag'],s['family'])for s in st.baseline]:raise ValueError('Exact current Break baseline mismatch')
    expected=read('research/core_retune/results/deployment-refit-r1/config-6-fixed-prior-product.report.json')
    for name,v in st.w.evaluate(check).items():
        if abs(v['macroF1']-expected['whole'][name]['macroF1'])>1e-12:raise ValueError('Consumer baseline mismatch')
    args=('detail',st.w.main,st.d.manual,[e['id']for e in st.extra])
    if st.score(check,*args)!=st.uncached_score(check,*args):raise ValueError('Cached metric equality failed')
    protocol=dict(scope='Retrospective bounded gate tuning; fixed current local W/b and fixed reference prior. All declared configs scanned independently for no-sensor/sustained-demand arms. Source groups for inner/outer, complete-data final replay separate.',grids={name:configs(name!='actions')for name in ['actions','detail','compact']},plans=PLANS,selection='0.30 manual per-chart accuracy +0.15 correct-denominator Break F1 +0.15 real local accuracy (manual substituted when no real case) +0.40 consumer macroF1; tie: closer to old knobs then stricter pressure coverage. No all-metric veto.',source=Path(__file__).read_text(encoding='utf8'))
    if(ROOT/'protocol.json').exists():
        prior=read(ROOT/'protocol.json')
        if{k:v for k,v in prior.items()if k!='source'}!={k:v for k,v in protocol.items()if k!='source'}:raise ValueError('Preserve protocol')
        write(ROOT/'execution-repair.json',dict(reason='Cache fixed per-config local and whole predictions, instead of recomputing every inner subset. No grid/selection/prediction changes. Baseline cached versus uncached full metric objects exactly equal.',source=protocol['source']))
    else:write(ROOT/'protocol.json',protocol)
    results={};selected={}
    for name in ['actions','detail','compact']:
        cfgs=configs(name!='actions');predictions=[st.predict(c,name)for c in cfgs];grids=[]
        for o,p in enumerate(PLANS):
            rows=[]
            for ci,(c,frames)in enumerate(zip(cfgs,predictions)):
                details=[st.score(frames,name,*st.scope(s,True))for s in p['inner']];rows.append(dict(id=ci,config=c,score=float(np.mean([r['score']for r in details])),details=details))
            choices={}
            def key(r):
                c=r['config'];return(r['score'],-abs(c['lowNpsScale']-1)-abs(c['confidenceFloor']-.5),c['fraction']or 0,-r['id'])
            for arm in ['off','pressure']if name!='actions'else['off']:
                eligible=[r for r in rows if(r['config']['quantile']is not None)==(arm=='pressure')];winner=max(eligible,key=key);choices[arm]=dict(id=winner['id'],config=winner['config'],score=winner['score'])
            write(ROOT/name/str(o)/'inner.json',dict(configs=rows,choices=choices));grids.append(rows)
            print('complete INNER',name,o,flush=True)
        # Freeze shared INNER choice before consuming OUTER scores.
        choices={}
        for arm in ['off','pressure']if name!='actions'else['off']:
            eligible=[i for i,c in enumerate(cfgs)if(c['quantile']is not None)==(arm=='pressure')]
            ci=max(eligible,key=lambda i:(np.mean([rows[i]['score']for rows in grids]),-abs(cfgs[i]['lowNpsScale']-1)-abs(cfgs[i]['confidenceFloor']-.5),cfgs[i]['fraction']or 0,-i));choices[arm]=dict(id=ci,config=cfgs[ci]);
        write(ROOT/name/'selection.json',choices);selected[name]=choices;result={}
        for arm in choices:
            pooled=list(st.baseline);details=[]
            for o,p in enumerate(PLANS):
                choice=read(ROOT/name/str(o)/'inner.json')['choices'][arm];frames=predictions[choice['id']];split=dict(train=p['train'],validation=p['test'],strongTrain=p['strongTrain'],strongValidation=p['strongTest']);ids,ms,es=st.scope(split,True);details.append(st.score(frames,name,ids,ms,es));uids={st.w.records[i]['uid']for i in ids}|set(ms)|{e['uid']for e in st.extra if e['id']in es}
                for uid in uids:
                    a,b=st.d.offset[uid];pooled[a:b]=frames[a:b]
            result[arm]=dict(sourceOOF=st.score(pooled,name,st.w.main,st.d.manual,[e['id']for e in st.extra]),outerDetails=details,final=st.score(predictions[choices[arm]['id']],name,st.w.main,st.d.manual,[e['id']for e in st.extra]))
        result['baseline']=st.score(st.baseline,name,st.w.main,st.d.manual,[e['id']for e in st.extra]);results[name]=result
        write(ROOT/name/'report.json',result);print(name,{k:{'manual':v['final']['manual']['accuracy'],'breaks':v['final']['breaks'],'wholeF':v['final']['whole']['macroF1']}for k,v in result.items()if k!='baseline'},flush=True)
    write(ROOT/'report.json',dict(results=results,selected=selected));write(ROOT/'complete.json',dict(actions=9,detail=171,compact=171,innerScopes=18,noFits='Exact gate-grid replay; no numerical optimizer or local W/b refit.'))


if __name__=='__main__':run()
