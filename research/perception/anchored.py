"""Conservative correction of mature tags by actual perceptual percentages.

No class-specific probability fit: retain/drop/promote thresholds are shared.
The exact shipped tag set is embedded, not approximated by a large logit.
"""
import itertools
import numpy as np

FINE_POLICIES=[dict(keepFloor=k,addFloor=a,mixOverride=m)for k,a,m in itertools.product([0.,.03,.07,.15,.25],[.5,.65,.8,.95,1.01],[.65,.8,.95,1.01])]
FAMILY_POLICIES=[dict(enabled=False,minRc=0.,minLn=0.)]+[dict(enabled=True,minRc=r,minLn=l)for r,l in itertools.product([.1,.18,.2,.3,.32,.4],[.1,.18,.2,.3,.32,.4])]


def base(d):return np.array([[c in r['legacyOverall']['tags']for c in d.classes]for r in d.records])


def predictions(d,share,family,baseline,fine_policies=FINE_POLICIES,family_policies=FAMILY_POLICIES):
    fi=[d.classes.index(c)for c in d.fine];mi=[d.classes.index(c)for c in ['hybrid','rc mix','ln mix']];mixed=baseline[:,mi].any(1)
    fine=[]
    for p in fine_policies:
        q=np.zeros_like(baseline);q[:,fi]=(baseline[:,fi]&(share>=p['keepFloor']))|(share>=p['addFloor'])
        strong=share>=p['mixOverride'];override=mixed&strong.any(1);q[mixed]=baseline[mixed];q[np.flatnonzero(override)[:,None],np.array(fi)[None,:]]=strong[override]
        q[np.flatnonzero(override)[:,None],np.array(mi)[None,:]]=False;fine.append(q)
    fine=np.array(fine);out=np.repeat(fine[None,:,:,:],len(family_policies),axis=0)
    for j,p in enumerate(family_policies):
        if not p['enabled']:continue
        hybrid=(family[:,0]>=p['minRc'])&(family[:,1]>=p['minLn']);out[j,:,hybrid,:]=False;out[j,:,hybrid,d.classes.index('hybrid')]=True
    return out
