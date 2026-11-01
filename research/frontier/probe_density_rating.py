"""Pinned Rating inspection of fixed-OD, no-DT human contrasts; no fitting."""
import json
from pathlib import Path
import numpy as np
from rating_bridge import load_rating,capture_native,RATING_COMMIT


def run():
    root=Path('research/frontier/results/density-probe-r2')
    tag=json.loads((root/'tag-probe.json').read_text(encoding='utf8'))
    a,b=[r for r in tag['records']if r['id'].startswith('CTX')]
    if a['targetHeads']!=b['targetHeads']:raise ValueError('Context target is not physically matched')
    rating,engine=load_rating('.cache/spm-rating-v1.2.0');records=[]
    for name in ['detail','compact']:
        model=rating.load_model(name)
        for r in tag['records']:
            notes=engine.parser.parse_file('artifacts/frontier-annotation-r2/'+r['id']+'.osu',rate=1.)
            e,meta=capture_native(notes,model,engine)
            whole=e.window(float(e.query[0]),float(np.nextafter(e.query[-1],np.inf)))
            target=e.window(*r['targetInterval']);reference=whole['channels']['demand']['rowP90']
            records.append({'id':r['id'],'rating':e.metadata,'probe':meta,'target':target,'whole':whole,
                'centralTarget':e.window(10000,16000)if r['id'].startswith('CTX')else None,
                'illustrativeRelativeDemand':target['channels']['demand']['rowMean']/reference if reference>0 else None})
            print(name,r['id'],'raw',e.metadata['raw_score'],'target D',target['channels']['demand']['rowMean'],flush=True)
    out={'scope':'Concept diagnosis only, no trained gate or accuracy estimate. Relative demand uses head-row, not uniform-time, statistics.',
        'ratingTag':'v1.2.0','ratingCommit':RATING_COMMIT,'models':2,'predictorCalls':len(records),
        'conditions':'7K, OD8, rate1; CTX targets literally identical; existing upstream parameters frozen',
        'records':records,'sourceSnapshots':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['rating_bridge.py','probe_density_rating.py']}}
    (root/'rating-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')


if __name__=='__main__':run()
