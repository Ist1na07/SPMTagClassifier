"""Audit already accepted joint states with an independent analytic pullback."""
from pathlib import Path
import sys
import numpy as np
from model import Data,read
from joint import Kernel
from fast_kernel import FastKernel
from study import PLANS
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier'))
from whole_reference import write


def run():
    d=Data();checks=[]
    for o in range(6):
        for ci in [0,5,11,17,23]:
            p=Path(f'.cache/perception-joint-r1/{o}/config-{ci}.json')
            if not p.exists():raise ValueError('Wait for actual accepted config '+str(p))
            r=read(p);s=PLANS[o]['inner'][0];k=FastKernel(d,d.main[s['train']],d.manual[s['strongTrain']],r['config'])
            for j,state in enumerate(r['inner'][0]['receipt']['starts']):
                theta=np.array(state['theta']);a,g=k.objective(theta);b,h=Kernel.objective(k,theta);loss=abs(a-b);gradient=float(np.abs(g-h).max())
                if loss>1e-10 or gradient>1e-9 or abs(a-state['loss'])>1e-9:raise ValueError('Actual accepted state no longer represents the frozen objective')
                checks.append(dict(outer=o,configId=ci,start=j,lossDifference=loss,gradientDifference=gradient,recordedLossDifference=abs(a-state['loss'])))
    write('research/perception/results/joint-r1/independent-gradient-oracle.json',dict(checks=checks,actualAcceptedStates=len(checks),maxLossDifference=max(c['lossDifference']for c in checks),maxGradientDifference=max(c['gradientDifference']for c in checks),
        method='Independent analytic NumPy chain rule vs original Torch autograd; frozen accepted states, no optimizer/model/selection changes',fits=0,sourceSnapshot=Path(__file__).with_name('fast_kernel.py').read_text(encoding='utf8')))
    print('independent accepted-state gradient oracle',len(checks),flush=True)


if __name__=='__main__':run()
