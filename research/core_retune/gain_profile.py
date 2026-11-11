"""Original literal score gain: actual decision and aggregate sensitivity."""
from joint_core import *


def run():
    d=Data();w=Whole(d);reference=read('research/local_feedback/results/product-readout-report.json');baseline=w.evaluate(d.baseline)
    for name,v in baseline.items():
        if abs(v['macroF1']-reference['whole'][name]['macroF1'])>1e-12:raise ValueError('Exact baseline aggregate mismatch '+name)
    rows=[]
    for family in ['rc','ln']:
        for gain in [.35,.7,1.4,2.8,5.6]:
            frames=d.frames({},gains={family:gain});changed=sum(a['tag']!=b['tag']for a,b in zip(d.baseline,frames));row=dict(family=family,gain=gain,changedPrimaryParents=changed,local=d.local(frames),real=d.real(frames),whole=w.evaluate(frames));rows.append(row);print('gain',family,gain,'changed',changed,'local',row['local']['accuracy'],'macroF',[row['whole'][n]['macroF1']for n in w.consumers],flush=True)
    write('research/core_retune/results/gain-profile.json',dict(scope='Full-data response surface, NOT tuned validation or a selected gain. Exact inherited support and fixed aggregate consumer; no grade/label changes.',baseline=baseline,rows=rows))


if __name__=='__main__':run()
