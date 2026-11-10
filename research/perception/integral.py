"""Exact piecewise-time integral; no scores or labels are fitted here.

Difficulty uses the upstream right-continuous row field, NOT a row average.
The last sample is extended to the physical endpoint, explicitly a reconstruction.
Action kernels and union holds are constant on the union of their breakpoints.
"""
import numpy as np

COMPONENTS=['Rbar','Pbar','Cbar','Jm']
LOCAL_KEYS=['relativeLogDemand','relativeLogActions','releaseVsPress','coordinationVsPress','jackVsPress','logDemand','releaseIndependence','crossCut']


def union_holds(notes,keys):
    holds=[]
    for col in range(keys):
        intervals=[]
        for n in sorted((n for n in notes if n['col']==col and n['lnEnd'] is not None),key=lambda n:n['t']):
            a,b=n['t'],n['lnEnd']
            if intervals and a<intervals[-1][1]:intervals[-1][1]=max(intervals[-1][1],b)
            else:intervals.append([a,b])
        holds.extend(intervals)
    return np.asarray(holds,dtype=float).reshape(-1,2)


def atoms(row,curve,width=1000.,hold=1.):
    m=row['map'];start=m['firstNote'];end=max(start+1.,m['lastNote']);notes=m['notes'];keys=m['keys']
    heads=np.sort([n['t']for n in notes]);hs=union_holds(notes,keys)
    tails=np.sort(hs[:,1])if len(hs)else np.zeros(0)
    query=np.asarray(curve['query'],float);channels={k:np.asarray(curve[k],float)for k in ['demand']+COMPONENTS}
    if not len(query)or np.any(np.diff(query)<=0)or any(v.shape!=query.shape or not np.isfinite(v).all()or(v<0).any()for v in channels.values()):raise ValueError('Invalid shared rating curve')
    # A rating curve belongs to this physical head-row clock, not an excerpt clock.
    if not np.array_equal(np.unique(heads),query):raise ValueError('Rating/source physical head-row clock mismatch: '+row['uid'])
    frame=np.asarray([[s['start'],s['end']]for s in row['frames']],float)
    points=[np.array([start,end]),query,frame.ravel(),heads-width/2,heads+width/2,tails-width/2,tails+width/2,hs.ravel()]
    grid=np.unique(np.clip(np.concatenate(points),start,end));dt=np.diff(grid);t=(grid[:-1]+grid[1:])/2
    count=lambda times:np.searchsorted(times,t+width/2,side='right')-np.searchsorted(times,t-width/2,side='right')
    held=(np.searchsorted(np.sort(hs[:,0]),t,side='right')-np.searchsorted(np.sort(hs[:,1]),t,side='right'))if len(hs)else np.zeros(len(t))
    action=(1000./width*(count(heads)+count(tails))+hold*held)/keys
    qi=np.clip(np.searchsorted(query,t,side='right')-1,0,len(query)-1)
    d=np.log1p(channels['demand'][qi]);parts={k:np.log1p(v[qi])for k,v in channels.items()}
    values=np.c_[d,np.log1p(action),parts['Rbar']-parts['Pbar'],parts['Cbar']-parts['Pbar'],parts['Jm']-parts['Pbar'],d]
    values[:,:2]-=np.average(values[:,:2],weights=dt,axis=0)
    ix=np.searchsorted(frame[:,0],t,side='right')-1
    valid=(ix>=0)&(t<frame[np.maximum(ix,0),1]);ix=np.where(valid,ix,-1)
    return dict(dt=dt,frame=ix,logDemand=d,logActions=np.log1p(action),values=values,held=held,action=action,
                ratingEndpointExtensionMs=max(0.,end-query[-1]))


def summarize(row,a,alpha,beta):
    weight=a['dt'].copy()
    if alpha:weight*=a['logDemand']**alpha
    if beta:weight*=a['logActions']**beta
    total=weight.sum();n=len(row['frames']);valid=a['frame']>=0;ix=a['frame'][valid];w=weight[valid]
    mass=np.bincount(ix,weights=w,minlength=n)/max(total,1e-30)
    x=np.array([np.bincount(ix,weights=w*a['values'][valid,j],minlength=n)for j in range(a['values'].shape[1])]).T
    x=np.divide(x,np.bincount(ix,weights=w,minlength=n)[:,None],out=np.zeros_like(x),where=mass[:,None]>0)
    geom=np.array([[g['releaseIndependence'],g['crossCut']]for g in row['geometry']])
    return mass,np.c_[x,geom],float(total)


def shares(mass,support):return mass@support
