"""Evaluate complete-data parameter sets, alongside their existing source study.

Source-heldout new fits versus a historically exposed fixed expert are not an
equal-exposure winner test. Keep that evidence, but also actually fit the final
parameter sets and replay them. This is development performance, not a new test
set. Frozen classic priors here are explicitly a first-stage comparison; any
shipping candidate still needs actual native aggregation/readout replay.
"""
from joint_core import *

ROOT=Path('research/core_retune/results/deployment-refit-r1')


def delta_from_models(d,models):
    out={}
    for f,m in models.items():
        b=d.blocks[f];native=b['z']@np.array(m['W']).reshape(len(m['classes']),-1).T+np.array(m['b'])
        out[f]=native-b['logits']
    return out


def evaluate(d,w,models):
    changes=delta_from_models(d,models);gains={f:m.get('scoreGain',1.4)for f,m in models.items()}
    frames=d.frames(changes,gains=gains,wildcard_rest_exemption=True)
    return dict(local=d.local(frames),real=d.real(frames),wholeFrozenPrior=w.evaluate(frames))


def run():
    d=Data();w=Whole(d);uids=set(d.manual)|{e['uid']for e in d.extra}
    configs=read('research/core_retune/results/joint-source-r1/protocol.json')['configs']
    common=read('research/core_retune/results/joint-source-r1/common-inner.json')
    # Choose by existing INNER evidence, before any full-data refit/replay.
    choices={}
    for weight in [0.,1.]:
        for free in [False,True]:
            rows=[r for r in common if r['config']['wholeWeight']==weight and r['config']['learnGain']==free]
            chosen=max(rows,key=lambda r:(r['metrics']['score'],r['config']['ridge'],-r['id']))
            choices[('joint'if weight else'local')+('/free'if free else'/fixed')]=dict(id=chosen['id'],config=chosen['config'],innerMetrics=chosen['metrics'])
    protocol=dict(id='complete-data-refit-r1',baseline='v1.1.0',productControl='v1.2.4 plus current Wildcard rest correction',
      scope='Retrospective development replay; every eligible existing local label and all349 whole labels available to final fit. Source-study results remain separately reported, not used as an asymmetric fixed-old veto.',
      configs=configs,choicesBeforeRefit=choices,localUids=sorted(uids),wholeUids=[w.records[i]['uid']for i in w.main],
      evaluation='Exact native family/score/late gate on8483 parents; existing three integral fields and frozen old classic priors. This first-stage metric is not the actual new native classic-prior product.',
      source=Path(__file__).read_text(encoding='utf8'))
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Refit protocol changed')
    write(p,protocol)
    baselineFrames=d.frames({},wildcard_rest_exemption=True)
    result=dict(protocol='protocol.json',baseline=dict(local=d.local(baselineFrames),real=d.real(baselineFrames),wholeFrozenPrior=w.evaluate(baselineFrames)),existingLocal=[],jointGrid=[])
    # Reuse valid full-data fits; don't run those same optimizations again.
    for mode in ['bias','weights']:
        for ridge in [.1,.01,.001]:
            source=Path('research/core_retune/results')/f'feasibility-{mode}-{ridge}.json';r=read(source)
            result['existingLocal'].append(dict(mode=mode,ridge=ridge,fitSource=str(source),**evaluate(d,w,r['models'])))
    for ci,c in enumerate(configs):
        p=ROOT/f'config-{ci}.json'
        if p.exists():row=read(p)
        else:
            models={};receipts={};start=time.monotonic()
            for f in ['rc','ln']:
                k=JointKernel(d,w,f,uids,w.main,ridge=c['ridge'],whole_weight=c['wholeWeight'],learn_gain=c['learnGain'],normalization='smooth32')
                theta,receipts[f]=fit(k);models[f]=k.export(theta)
                if any(not np.isfinite(v['loss'])or v['maximumGradient']>1e-5 for v in receipts[f]):raise ValueError('Unaccepted fit '+str(ci)+' '+f)
            row=dict(id=ci,config=c,models=models,receipts=receipts,seconds=time.monotonic()-start,**evaluate(d,w,models));write(p,row)
        result['jointGrid'].append({k:v for k,v in row.items()if k not in ['models','receipts']})
        print('full refit',ci,c,'manual',row['local']['accuracy'],'real',row['real']['accuracy'],'wholeF',{k:v['macroF1']for k,v in row['wholeFrozenPrior'].items()},flush=True)
    write(ROOT/'report.json',result);write(ROOT/'complete.json',dict(jointConfigs=len(configs),newFamilyFits=len(configs)*2,starts=len(configs)*4,reusedLocalFits=6,choicesBeforeRefit=choices))


if __name__=='__main__':run()
