"""Validate an equation-preserving scalar root solver repair, not a new fit."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import competition as c


def run():
    root=Path('research/local_feedback/results/competition-r2');protocol=c.read(root/'protocol.json');old_source=protocol['sources']['competition.py'];current=Path(c.__file__).read_text(encoding='utf8')
    scope={'__file__':c.__file__,'__name__':'old_competition_solver_spec'};exec(compile(old_source,c.__file__,'exec'),scope)
    old=scope['allocate'];new=c.allocate;d=c.Data();checks=[]
    try:
        for o in range(6):
            for ci in [0,7,18]:
                path=Path(f'.cache/local-feedback-competition-r2/{o}/config-{ci}.json')
                if not path.exists():continue
                saved=c.read(path);config=saved['row']['config']
                for j,r in enumerate(saved['inner']):
                    split=c.PLANS[o]['inner'][j];k=c.Kernel(d,d.main[split['train']],d.manual[split['strongTrain']],config);theta=np.array(r['theta'])
                    c.allocate=old;a,g=k.objective(theta);c.allocate=new;b,h=k.objective(theta)
                    loss=abs(a-b);gradient=float(np.max(np.abs(g-h)))
                    if loss>1e-10 or gradient>1e-8:raise ValueError('Cached objective changed, cannot reuse')
                    checks.append(dict(outer=o,configId=ci,inner=j,lossDifference=loss,gradientDifference=gradient))
    finally:c.allocate=new
    with np.load('.cache/local-feedback-projection-failure.npz')as z:
        q=new(z['prior'],z['delta'],z['budget']);err=float(np.max(np.abs(q.sum(1)-z['budget'])))
        worst=dict(prior=z['prior'][616].tolist(),delta=z['delta'][616].tolist(),budget=float(z['budget'][616]))
    if err>2e-13:raise ValueError('Captured line-search projection still fails')
    c.write(root/'numerical-repair.json',dict(reason='Bracketed Newton oscillated across saturated sigmoid plateaus on an actual line-search trial. Pure bisection shrinks the bracket every step; same unique support-budget equation and implicit derivative.',
        checks=checks,maxLossDifference=max(r['lossDifference']for r in checks),maxGradientDifference=max(r['gradientDifference']for r in checks),
        capturedFailureBudgetError=err,capturedFailureRow=worst,originalProtocolPreserved=True,sourceBefore=old_source,sourceAfter=current))
    print('numerical repair accepted',len(checks),'cached states','captured budget error',err,flush=True)


if __name__=='__main__':run()
