"""No fitting: hand off a completed resource candidate to its native consumer."""
from targeted_sequence_model import World,frames,INPUT
import json
from pressure_data import read
from rc_action_runtime import smooth,sufficient,aggregate
from whole_reference import write

def run():
    source=read('research/overlay-quality/targeted-sequence-r1/resource-check.json');m=source['model'];w=World('local2s')
    policy=dict(dominantShare=.52,mixShare=.07,secondRel=.75);profile=dict(context='local2s',model=m,policy=policy)
    ids=['manual/DS','manual/AA','local/F01','local/F02']+[r['uid']for r in w.main[::40]]
    rows=[];sources={r['uid']:r['source']for r in map(json.loads,INPUT.read_text(encoding='utf8').splitlines())}
    for uid in ids:
        f=frames(w,uid,m);rows.append(dict(uid=uid,source=sources[uid],scope='exact-target-interval'if uid.startswith('local/')else'native-full-chart',parents=w.inputs[uid]['parents'],rawExpected=f,expected=smooth(f),overall=aggregate(sufficient(f),policy)))
    write('research/overlay-quality/targeted-sequence-r1/runtime-resource-fixture.json',dict(profile=profile,rows=rows))
    print('Resource-candidate runtime fixture',len(rows))

if __name__=='__main__':run()
