"""Same lexicographic selector as R2, without rebuilding recipe dicts per tie."""
import numpy as np


def select(d,scores,baseline,name='all',criterion='protected',family_enabled=None):
    from anchored_extended_study import FINE,FAMILY,NEAR
    from threshold_study import reps,KEYS,eligible
    rr=reps(d);allowed=np.array([eligible(d,r,name)for r in rr])[:,None,None]
    if family_enabled is not None:allowed=allowed&np.array([p['enabled']==family_enabled for p in FAMILY])[None,:,None]
    if criterion=='protected':
        for j in [0,1,2,3]:allowed=allowed&(scores[...,j]>=baseline[j]-1e-12)
    objective=.5*scores[...,2]+.25*scores[...,3]+.25*scores[...,4]if criterion!='macro'else scores[...,2];value=np.where(allowed,objective,-np.inf);maximum=value.max()
    if not np.isfinite(maximum):return None
    at=np.flatnonzero(value.ravel()>=maximum-(NEAR if criterion=='protected'else 1e-14))
    dims=value.shape;r,f,p=np.unravel_index(at,dims)
    if criterion=='protected':
        penalties=[]
        for c in rr:
            w=d.manifest['weights'][c['weightId']];penalties.append([c['hard'],(w['alpha']-1)**2+(w['beta']-1)**2+(c['power']-1)**2,w['width']!=1000.,w['hold']!=1.])
        prior=np.array(penalties)
        rules=np.array([sum([x['keepFloor']>0,x['addFloor']<=1.,x['mixOverride']<=1.])for x in FINE]);families=np.array([x['enabled']for x in FAMILY])
        column=prior[r,0];take=column==column.min();at=at[take];r=r[take];f=f[take];p=p[take]
        # Sequential filtering, recomputing only cheap indexed arrays.
        for j in range(1,4):
            a=prior[r,j];take=a==a.min();at=at[take];r=r[take];f=f[take];p=p[take]
        for getter in [lambda:rules[p]+families[f],lambda:-value.ravel()[at],lambda:-scores[...,1].ravel()[at],lambda:at]:
            a=getter();take=a==a.min();at=at[take];r=r[take];f=f[take];p=p[take]
    else:
        a=scores[...,1].ravel()[at];at=at[a==a.max()];r,f,p=np.unravel_index(at,dims)
    chosen=int(at.min());r,f,p=map(int,np.unravel_index(chosen,dims))
    return dict(representationId=r,familyId=f,policyId=p,representation=rr[r],family=FAMILY[f],policy=FINE[p],metrics=dict(zip(KEYS,scores[r,f,p].tolist())),objectiveDeficitFromMaximum=float(maximum-value.ravel()[chosen]),nearOptimalTolerance=NEAR if criterion=='protected'else 0.)
