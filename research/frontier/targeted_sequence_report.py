"""Consume the predeclared product criterion; do not fit or move thresholds."""
from pathlib import Path
import argparse,itertools
import numpy as np
from targeted_sequence_model import World,FEATURES,frames
from targeted_sequence_study import ROOT,CONFIGS,FIXED,POST,PLANS,choose
from pressure_data import mil,read
from rc_action_runtime import sufficient,policy_predictions,macro_scores
from landscape_audit import multilabel_metrics
from whole_reference import write

NAMES=['shipped_fixed','frozen','macro','experience','experience_fixed','experience_no_error']
AXES=['localL2','strongWeight','errorBoost','effortPower','primaryPower']


def declare():
    value=dict(selection='Exact producer criterion, joint contexts/representations/hyperparameters/all245post; OUTER truth never selects an arm. Mean6INNER arrays select one common deployment configuration.',
        local='Displayed accuracy and pressure weighted symmetric Tech<->CS confusion; original raw accuracy separately retained. No invented ground-truth confidence. All2contexts x5representations independently retuned.',
        profiles='For each actual value of each coordinate, reselect all others+arm+post using original criterion and guards. Active models only; frozen has inactive training parameters. Replay actual INNER whole tag vectors to distinguish same scalar metrics from same predictions.',
        audit='Retain projected gradients/multistart spread/embedded-parent TRAIN comparisons; numerical stationarity is not a global optimum proof.',
        acceptance='No automatic adoption. Compare OOF P/R/F1, exact sets, original6manual, Tech/Dense error milliseconds and query diagnostics; native consumer and actual renderer required for adopted code.',
        sourceSnapshot=Path(__file__).read_text(encoding='utf8'))
    f=ROOT/'consumer-protocol.json'
    if f.exists()and read(f)!=value:raise ValueError('Frozen targeted consumer changed')
    if not f.exists():write(f,value)


def run():
    complete=read(ROOT/'complete.json');base=World('native');uids=[r['uid']for r in base.main]
    y=mil.encode(base.main);weight=np.array([r['recordWeight']for r in base.main]);groups=[r['group']for r in base.main]
    selections={n:[]for n in NAMES};final_candidates={n:[]for n in NAMES};arms={};profiles=[{}for _ in PLANS];audit=[]
    for context,kind in itertools.product(['native','local2s'],FEATURES):
        tables=[];arm=context+'/'+kind;arms[arm]={n:[]for n in NAMES}
        for outer in range(6):
            folder=ROOT/context/kind/str(outer);rows=read(folder/'inner-search.json');tables.append(rows)
            choices=read(folder/'selection.json')['choices'];ref=choices['shipped_fixed']
            for name,c in choices.items():
                c={**c,'context':context,'kind':kind,'outer':outer};selections[name].append(c);arms[arm][name].append(c)
            for axis in AXES:
                for value in dict.fromkeys(c[axis]for c in CONFIGS):
                    eligible=[r for r in rows if r['headId']and r['config'][axis]==value and r['macroF1']>=ref['macroF1']-1e-12 and r['macroRecall']>=ref['macroRecall']-1e-12]
                    if not eligible:continue
                    chosen=choose(eligible,'experience',ref);key=axis+':'+str(value)
                    profiles[outer].setdefault(key,[]).append({**chosen,'context':context,'kind':kind,'outer':outer})
            for r in read(folder/'fits.json'):
                if r['maximumProjectedGradient']>1e-5:raise ValueError('Unaccepted optimizer state')
                losses=[x['loss']for x in r['starts']];best=min(losses)
                for proof in r['nestedTrainProofs']:
                    if best>proof['embeddedLoss']+1e-8:raise ValueError('TRAIN inclusion failure')
                audit.append(dict(context=context,kind=kind,outer=outer,headId=r['headId'],inner=r['inner'],gradient=r['maximumProjectedGradient'],startSpread=max(losses)-best,parentComparisons=len(r['nestedTrainProofs'])))
        pooled=[{**r,**{k:float(np.mean([t[i][k]for t in tables]))for k in ['macroF1','macroRecall','manualAccuracy','pressureConfusion']},'context':context,'kind':kind}for i,r in enumerate(tables[0])]
        ref=next(r for r in pooled if r['headId']==0 and r['policyId']==FIXED)
        for name in NAMES:final_candidates[name].append(ref if name=='shipped_fixed'else choose(pooled,name,ref))
    outer_choices={};final={}
    for name in NAMES:
        outer_choices[name]=[]
        for o in range(6):
            rows=[r for r in selections[name]if r['outer']==o];ref=next(r for r in selections['shipped_fixed']if r['outer']==o)
            outer_choices[name].append(rows[0]if name=='shipped_fixed'else choose(rows,name,ref))
        final[name]=final_candidates[name][0]if name=='shipped_fixed'else choose(final_candidates[name],name,final_candidates['shipped_fixed'][0])
    write(ROOT/'joint-inner-selection.json',dict(usesOuterTruth=False,choices=outer_choices))
    write(ROOT/'final-inner-selection.json',dict(usesOuterTruth=False,selections=final))
    # The complete final selection above is frozen before reading any OUTER prediction.
    def evaluate(name,choices):
        preds={};locals=[];regs=[]
        for c in choices:
            f=ROOT/c['context']/c['kind']/str(c['outer'])
            for r in read(f/'outer-whole.json'):
                if r['name']==name:
                    if r['uid']in preds:raise ValueError('Duplicate OOF identity')
                    preds[r['uid']]=r
            locals.extend(r for r in read(f/'outer-local.json')if r['name']==name)
            regs.extend(r for r in read(f/'regressions.json')if r['name']==name)
        if set(preds)!=set(uids)or len(locals)!=6 or len({r['uid']for r in locals})!=6:raise ValueError('Incomplete OOF inventory')
        p=np.array([[c in preds[u]['predicted']for c in mil.MAP_CLASSES]for u in uids],int);m=multilabel_metrics(y,p,weight,mil.MAP_CLASSES,groups)
        m['perClassErrors']={c:{label:[uids[i]for i in np.flatnonzero(mask)]for label,mask in [('FP',(y[:,j]==0)&(p[:,j]==1)),('FN',(y[:,j]==1)&(p[:,j]==0))]}for j,c in enumerate(mil.MAP_CLASSES)}
        m.update(displayManualAccuracy=float(np.mean([r['accuracy']for r in locals])),rawManualAccuracy=float(np.mean([r['rawAccuracy']for r in locals])),pressureConfusion=float(np.mean([r['pressureConfusion']for r in locals])),techCsConfusionMs=sum(r['confusionMs']for r in locals),
            techToDenseMs=sum(d['end']-d['start']for r in locals for d in r['details']if'tech'in d['truth']and d['predicted']=='dense chordstream'))
        return dict(metrics=m,local=locals,regressions=regs,predictions=p.tolist())
    result={n:evaluate(n,c)for n,c in outer_choices.items()};arm_results={a:{n:evaluate(n,c)['metrics']for n,c in names.items()}for a,names in arms.items()}
    write(ROOT/'analysis.json',dict(complete=complete,joint=result,arms=arm_results,optimization=dict(targetReceipts=len(audit),maxProjectedGradient=max(r['gradient']for r in audit),maxStartLossSpread=max(r['startSpread']for r in audit),nestedComparisons=sum(r['parentComparisons']for r in audit)),adopted=False))
    # Replay conditional-profile *vectors*. Retain scalers to avoid identity
    # recycling in the research design cache when loading saved model objects.
    worlds={};saved_models={};vectors={};retained=[]
    def vector(c):
        key=c['context'],c['kind'],c['outer'],c['headId'],c['policyId']
        if key in vectors:return vectors[key]
        context,kind,outer,hid,pid=key
        if context not in worlds:worlds[context]=World(context)
        w=worlds[context];folder=ROOT/context/kind/str(outer)
        if key[:3]not in saved_models:
            rs=read(folder/'inner-models.json');retained.extend(rs);saved_models[key[:3]]={(r['headId'],r['inner']):r['model']for r in rs}
        stats=[];ids=[]
        for inner,s in enumerate(PLANS[outer]['inner']):
            m=saved_models[key[:3]][hid,inner]if hid else None
            stats.extend(sufficient(frames(w,w.main[i]['uid'],m))for i in s['validation']);ids.extend(s['validation'])
        pred=policy_predictions(stats,[POST[pid]])[0];f,r=macro_scores(y[ids],pred[None],weight[ids])
        if abs(f[0]-c['macroF1'])>1e-11 or abs(r[0]-c['macroRecall'])>1e-11:raise ValueError('Saved profile prediction does not replay')
        vectors[key]=pred,ids;return vectors[key]
    curves=[]
    for outer,points in enumerate(profiles):
        reference=outer_choices['experience'][outer];v,ids=vector(reference);samples=[]
        for axis in AXES:
            for value in dict.fromkeys(c[axis]for c in CONFIGS):
                key=axis+':'+str(value);rows=points.get(key,[])
                if not rows:samples.append(dict(axis=axis,value=value,eligibleActiveModel=False));continue
                c=choose(rows,'experience',next(r for r in selections['shipped_fixed']if r['outer']==outer));p,pids=vector(c)
                if pids!=ids:raise ValueError('Profile validation identity order changed')
                samples.append(dict(axis=axis,value=value,eligibleActiveModel=True,choice=c,changedWholeTagSetMass=float(weight[ids]@np.any(p!=v,axis=1)/weight[ids].sum())))
        curves.append(dict(outer=outer,reference=reference,activeTrainingParameterProfiles=samples))
    write(ROOT/'conditional-profiles.json',curves)
    lines=['# 限时定点：连续行结构与逐手时序','',
        '两种上下文×五种表示，各自完整60训练配置×245聚合；原6×3歌曲分组。体验选择先保护INNER F1及召回，再减少显示中的压力加权Tech/CS错误。','',
        '| 联合选择 | macro P | macro R | macro F1 | micro F1 | 整谱集合全对 | 显示局部 | 原始局部 | Tech→Dense ms | 双向Tech/CS ms |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n,r in result.items():
        m=r['metrics'];xs=[m['macro']['precision'],m['macro']['recall'],m['macro']['f1'],m['micro']['f1'],m['exactMatch'],m['displayManualAccuracy'],m['rawManualAccuracy']]
        lines.append('| '+n+' | '+' | '.join(f'{100*x:.2f}%'for x in xs)+f" | {m['techToDenseMs']:.1f} | {m['techCsConfusionMs']:.1f} |")
    lines+=['','共同INNER部署配置：'+str(final['experience']),'','本报告不自动采用。两条Speed摘录只作回归，旧外层资料已多轮观察，不称新盲测或模型上限。']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)
    write(ROOT/'consumer-complete.json',dict(profiles=len(curves),vectorReplays=len(vectors),noNewFits=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--declare-only',action='store_true');a=p.parse_args();declare()
    if not a.declare_only:run()
