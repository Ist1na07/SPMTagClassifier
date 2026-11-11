"""Local errors that changing only core weights cannot represent."""
from core import *


def run():
    d=Data();tech=[];constant=[]
    for uid in d.manual:
        total=sum(a['end']-a['start']for a in d.records[uid]['annotations']if a['tags']);best=0
        for j in range(*d.offset[uid]):
            p=d.parents[j];votes={}
            for a in d.records[uid]['annotations']:
                dt=max(0,min(p['end'],a['end'])-max(p['start'],a['start']))
                for c in a['tags']:votes[c]=votes.get(c,0)+dt
                if dt and 'tech'in a['tags']and'dense chordstream'not in a['tags']and p['baseline']['tag']=='dense chordstream':tech.append(dict(uid=uid,parent=j,start=p['start'],end=p['end'],wrongTechMs=dt))
            best+=max(votes.values(),default=0)
        constant.append(dict(uid=uid,bestConstantParentAccuracy=best/total))
    for e in tech:
        p=d.parents[e['parent']];e['truthDurationMs']={}
        for a in d.records[e['uid']]['annotations']:
            dt=max(0,min(p['end'],a['end'])-max(p['start'],a['start']))
            if dt:e['truthDurationMs']['|'.join(a['tags'])]=e['truthDurationMs'].get('|'.join(a['tags']),0)+dt
    forced=np.zeros_like(d.blocks['ln']['logits']);forced[:,d.blocks['ln']['wild']]=1000;frames=d.frames({'ln':forced});real=[]
    for e in d.extra:
        if e['id']not in ['I01','I02']:continue
        s,t=e['interval'];a,b=d.offset[e['uid']];parts=[]
        for j,w in d.overlap(e['uid'],s,t):parts.append(dict(parent=j,durationMs=w,preLate=d.parents[j]['preLate']['family'],ordinary=d.parents[j]['baseline']['tag'],allModelMassToWildcard=frames[j]['tag'],wildScore=float(d.blocks['ln']['fixed'][j,0])))
        real.append(dict(id=e['id'],expected=e['tags'],parts=parts))
    result=dict(scope='Structural diagnosis, not candidate performance. Parent-constant oracle permits any category and ignores learnability; it only bounds one-label-per-parent readout.',parentConstantOracle=constant,techToDense=tech,wildcardFeasibility=real);write('research/core_retune/results/structural-audit.json',result);print(result)


if __name__=='__main__':run()
