"""Count which actual annotated errors a fixed-routing adapter can change."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import Data,write,RC,LN


def run():
    d=Data();rows=[];totals={};bychart=[]
    for i in d.manual:
        r=d.records[i];total=error=blocked=correct=0.
        for ann in r['annotations']:
            tags=ann['tags']
            if not tags:continue
            for j in range(*r['offset']):
                f=d.frame_records[j][1];dt=max(0.,min(ann['end'],f['end'])-max(ann['start'],f['start']))
                if not dt:continue
                total+=dt;ok=f['tag']in tags;correct+=dt*ok
                if ok:continue
                error+=dt;family=f['family'];candidates=RC if family=='rc'else LN if family=='ln'else []
                reachable=any(c in tags and f['scores'].get(c,0)>0 for c in candidates)
                reason='learnable within family'if reachable else'Break false negative'if family=='break'else'Break false positive'if tags==['break']else'family mismatch'if not any(c in candidates for c in tags)else'zero candidate'
                if not reachable:blocked+=dt
                totals[reason]=totals.get(reason,0.)+dt
                rows.append(dict(uid=r['uid'],frame=j-r['offset'][0],start=max(ann['start'],f['start']),end=min(ann['end'],f['end']),durationMs=dt,actual=tags,predicted=f['tag'],reason=reason))
        bychart.append(dict(uid=r['uid'],labelledMs=total,errorMs=error,blockedErrorMs=blocked,accuracy=correct/total,
            fixedRoutingAccuracyCeiling=1-blocked/total))
    write('research/local_feedback/results/learnability-r2.json',dict(scope='Original6 natural raw frames, actual overlap; a ceiling, not a measured improved classifier',
        totalsMs=totals,charts=bychart,errors=rows,chartMeanFixedRoutingCeiling=float(np.mean([r['fixedRoutingAccuracyCeiling']for r in bychart]))))
    print('error milliseconds',totals,'mean routing ceiling',np.mean([r['fixedRoutingAccuracyCeiling']for r in bychart]),flush=True)


if __name__=='__main__':run()
