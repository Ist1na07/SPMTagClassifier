"""Ablate the EXTRA relative head-burden gate; archived gates stay unchanged."""
from native_search import pack,CLASSES,BI


def pack_gated(values,config,treatment='ln_protected'):
    if treatment=='all':return pack(values,config)
    if treatment not in ['ln_protected','off']:raise ValueError('Unknown extra gate treatment')
    if treatment=='off'or not config['burdenGate']:return pack(values,{**config,'burdenGate':0.})
    modified=[]
    for value in values:
        rows=value['partitions'][config['partition']]['cases'][config['head']]['raw']
        if config['smoothing']!='coherent':raise ValueError('Gate study requires unchanged raw coherent intervals')
        active=sorted((s['burden'],s['end']-s['start'])for s in rows if s['burden']>0)
        mass=sum(w for v,w in active);acc=0.;reference=1.
        for v,w in active:
            acc+=w;reference=v
            if acc>=.75*mass:break
        output=[]
        for s in rows:
            if s['family']=='rc'and s['burden']/reference<config['burdenGate']:
                p=[0.]*len(CLASSES);p[BI]=1.
                output.append({**s,'family':'break','tag':'break','confidence':1.,'probability':p})
            else:output.append(s)
        modified.append({'partitions':[{'cases':[{'raw':output}]}]})
    return pack(modified,{**config,'partition':0,'head':0,'burdenGate':0.})
