"""Parameter-free hand-motion certificate; exact drop-only affinity ablation.

This is an independent correction of the fixed product, not another learned
model or a change to the running joint training objective.
"""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity import Data,read,write,parent
from competition import local_metrics,summary,RC
from reanalyze_allowed import corrected
from joint_study import cases
from joint_activity import local

ROOT=Path('research/local_feedback/results/static-cut-r1')
CS=['dense chordstream','fast chordstream']


def change(frame):
    scores={**frame['scores'],**{c:0. for c in CS}}
    if frame['tag']not in CS:return dict(frame,scores=scores,secondary=None if frame.get('secondary')in CS else frame.get('secondary'))
    ranked=sorted(scores.items(),key=lambda v:-v[1]);tag,value=ranked[0]
    if tag=='wildcard':
        for c,v in ranked[1:]:
            if c!='wildcard'and v>.3:tag,value=c,v;break
    secondary=ranked[1][0]if len(ranked)>1 and ranked[1][1]>=.75*ranked[0][1]and ranked[1][1]>.3 else None
    return dict(frame,scores=scores,tag=tag,confidence=min(1.,value),secondary=secondary)


def run():
    d=Data();rows=[json.loads(s)for s in Path('.cache/local-feedback-static-cut-r1.jsonl').read_text(encoding='utf8').splitlines()]
    if [r['uid']for r in rows]!=[r['uid']for r in d.records]:raise ValueError('Cut guard source inventory mismatch')
    for row,record in zip(rows,d.records):
        if [(f['start'],f['end'],f['parent'])for f in row['frames']]!=[(f['start'],f['end'],f['parent'])for f in record['frames']]:raise ValueError('Cut guard interval mismatch')
    base_probability=np.eye(3)[d.route];base_support=np.sum(base_probability[:,1:,None]*d.expert,axis=1);raw=[f for _,f in d.frame_records];baseline=dict(local=summary(corrected(local_metrics(d,d.manual))),real=cases(d,raw,[e for e in d.extra if e['group']is not None]),consumers={n:parent.metrics(d.y[d.main],q[d.main],d.weight[d.main])for n,q in d.fixed.items()});results=[]
    for kind in ['masks','alternation']:
        for wid,width in enumerate([1000,2000,4000]):
            mask=np.concatenate([np.array(r[kind][wid],bool)for r in rows]);mask&=d.route==1;support=base_support.copy();support[np.ix_(mask,[d.fine.index(c)for c in CS])]=0;frames=[change(f)if active else f for f,active in zip(raw,mask)];loc=local(d,d.manual,frames);consumers={n:parent.metrics(d.y[d.main],d.consume(support,base_probability,n)[d.main],d.weight[d.main])for n in d.consumer_names};changes=[]
            for i in d.main:
                a,b=d.records[i]['offset'];ms=sum(f['end']-f['start']for j,f in enumerate(raw[a:b],a)if mask[j]and f['tag']in CS)
                if ms:changes.append(dict(uid=d.records[i]['uid'],changedPrimaryMs=ms))
            result=dict(kind=kind,width=width,local=summary(loc),manual=loc,real=cases(d,frames,[e for e in d.extra if e['group']is not None]),consumers=consumers,certifiedFrames=int(mask.sum()),changedPrimaryFrames=int(sum(m and f['tag']in CS for f,m in zip(raw,mask))),changedCharts=changes);results.append(result);print(kind,width,result['local'],{n:round(m['macroF1']*100,4)for n,m in consumers.items()},'changedcharts',len(changes),flush=True)
    write(ROOT/'audit.json',dict(scope='Fixed v1.2.3, six hand-rule/window alternatives, no fitted coefficients. Whole support is drop-only: zero certified CS affinities, all other affinities and old family/whole prior unchanged. Thus this does not smuggle a new normalization into the comparison. Descriptive full replay; do not call selection on it blind validation.',baseline=baseline,results=results))


if __name__=='__main__':run()
