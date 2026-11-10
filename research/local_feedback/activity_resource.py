"""Independent original-kernel oracle and changed execution-cost measurement."""
from pathlib import Path
import sys,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity import *


def run():
    d=Data();s=PLANS[0]['inner'][0];old=read('research/local_feedback/results/activity-resource-r6.json');cfg=old['config'];k=Kernel(d,d.main[s['train']],d.manual[s['strongTrain']],cfg);checks=[];rng=np.random.default_rng(24)
    for task in [0.,1.]:
        k.set_config(dict(cfg,wholeWeight=task))
        for theta in [np.zeros(k.dimension),rng.normal(0,.05,k.dimension),rng.normal(0,.1,k.dimension)]:
            a,g=k.objective(theta);b,h=parent.Kernel.objective(k,theta);loss=abs(a-b);gradient=float(np.max(np.abs(g-h)))
            if loss>1e-10 or gradient>1e-9:raise ValueError('Execution optimization changed the TRAIN objective')
            checks.append(dict(wholeWeight=task,lossDifference=loss,gradientDifference=gradient))
    k.set_config(cfg);theta,receipt=fit(k)
    write('research/local_feedback/results/activity-resource-fast-r6.json',dict(config=cfg,receipt=receipt,oracle=checks,source=Path(__file__).with_name('activity.py').read_text(encoding='utf8')))
    print('activity exact fast resource',receipt['seconds'],'dimension',k.dimension,'oracle gradient',max(v['gradientDifference']for v in checks),flush=True)


if __name__=='__main__':run()
