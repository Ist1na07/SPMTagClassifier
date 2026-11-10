"""Consumer-driven restriction of the completed R2 grid; no new fits or grid.

Prototype degrees are affinities, not calibrated probabilities. Squaring them
before keep/drop gates suppressed all solid tags on 25 formerly tagged inputs
in the Compact common replay. Keep their first-degree meaning, reselect EVERY
other active coordinate jointly, and report this post-hoc restriction openly.
"""
from pathlib import Path
import json
import numpy as np
from extended import ExtendedData
from compact import CompactData
from model import read,metrics
from study import PLANS
from anchored import base,predictions
from threshold_study import reps,KEYS
from fast_select import select

ROOT=Path('research/perception/results/linear-support-r3')
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n',encoding='utf8')

def run():
    ROOT.mkdir(exist_ok=True)
    protocol=dict(reason=__doc__,supportPower=1.,newGrid=False,newFits=0,newScorerCalls=0,
        selection='Same R2 Cartesian tables, all active coordinates reselected on INNER; same safeguards/tolerance/tie prior. Post-hoc consumer restriction, retrospective only.',
        sourceSnapshot=Path(__file__).read_text(encoding='utf8'))
    p=ROOT/'protocol.json'
    if p.exists()and read(p)!=protocol:raise ValueError('Frozen restriction changed')
    write(p,protocol)
    for name,cls,up in [('detail',ExtendedData,'anchored-r2'),('compact',CompactData,'anchored-compact-r2')]:
        d=cls();b=base(d);folder=ROOT/name;folder.mkdir(exist_ok=True);choices=[];refs=[];total=None
        mask=np.array([r['hard']or r['power']==1 for r in reps(d)])
        arms=['hard_time','branch_time','difficulty','actions','combined']
        for o in range(6):
            upstream=Path('research/perception/results')/up/str(o)
            with np.load(upstream/'inner-scores.npz')as z:table=z['scores'].copy()
            if total is None:total=table.copy()
            else:total+=table
            ref=np.array([read(upstream/'selection.json')['baseline'][k]for k in KEYS]);refs.append(ref)
            table[~mask]=-np.inf;choices.append({arm:select(d,table,ref,arm,'protected')for arm in arms})
        total/=6;total[~mask]=-np.inf;final={arm:select(d,total,np.mean(refs,axis=0),arm,'protected')for arm in arms}
        write(folder/'selection.json',dict(usesOuterTruth=False,folds=choices,common=final))
        # Freeze all selections above before consuming held-out truth below.
        output={};perclass={};query={};replay={}
        def vector(c):
            r=c['representation'];_,_,share,family=d.representation(r['weightId'],r['power'],r['hard']);return predictions(d,share,family,b,[c['policy']],[c['family']])[0,0],share
        for arm in arms:
            q=np.zeros_like(b)
            for o in range(6):
                actual,_=vector(choices[o][arm]);ids=d.main[PLANS[o]['test']];q[ids]=actual[ids]
            m=metrics(d.y[d.main],q[d.main],d.weight[d.main]);m['perClass']=dict(zip(d.classes,m['perClass']));output[arm]=m
            perclass[arm]=[dict(uid=d.records[i]['uid'],actual=[c for c,y in zip(d.classes,d.y[i])if y],predicted=[c for c,y in zip(d.classes,q[i])if y])for i in d.main]
            actual,share=vector(final[arm]);replay[arm]=metrics(d.y[d.main],actual[d.main],d.weight[d.main]);query[arm]=[dict(uid=r['uid'],predicted=[c for c,y in zip(d.classes,actual[i])if y],percentages=dict(zip(d.fine,(100*share[i]).tolist())))for i,r in enumerate(d.records)if r['dataset']in ['query','derived']]
        write(folder/'oof.json',dict(baseline=metrics(d.y[d.main],b[d.main],d.weight[d.main]),metrics=output,predictions=perclass))
        write(folder/'common-replay.json',dict(scope='Full-study replay, NOT fresh/OOF',metrics=replay,queries=query))
        print(name,{arm:{k:round(v*100,2)for k,v in m.items()if k in KEYS}for arm,m in output.items()},flush=True)
    write(ROOT/'complete.json',dict(models=2,newFits=0,newScorerCalls=0,gridRestriction='power1 only; all other active coordinates jointly reselected'))
if __name__=='__main__':run()
