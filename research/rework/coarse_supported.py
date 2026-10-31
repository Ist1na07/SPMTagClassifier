"""Deterministic physical support after an unchanged fitted coarse expert.

The expert and its training objective are unchanged auxiliary components. This
new postprocessor's FINAL predictions require full configuration reselection;
it is not described as training the old expert on a newly masked objective.
"""
from coarse_model import Packed,predict as raw_predict,SCHEMA,FINE,np,torch


def masks(records,config,mode):
    if mode not in ['none','jack','ln','both']:raise ValueError('Unknown physical support mode')
    keys=SCHEMA['keys'];index={k:keys.index(k)for k in ['rowRate','occupancy','release:tailRatePerKey','lnHeadFraction','chordChange','overlap','ioiCv','rowBpm']}
    result=[]
    for r in records:
        for w in r['windows'][str(config['windowMs'])]:
            m={k:w['x'][j]for k,j in index.items()};seconds=max(1,w['contextEnd']-w['contextStart'])/1000
            rows=round(m['rowRate']*seconds);held=m['occupancy']>0;tails=m['release:tailRatePerKey']>0;tag=-1
            if mode in ['ln','both']and(w['notes']==0 and(held or tails)or rows<=1 and w['notes']>0 and m['lnHeadFraction']==1 and held):tag=FINE.index('ln')
            elif mode in ['jack','both']and rows>=3 and m['lnHeadFraction']==0 and not held and not tails and m['chordChange']==0 and m['overlap']==1 and m['ioiCv']<=1e-9 and m['rowBpm']>0:tag=FINE.index('jack')
            result.append(tag)
    return np.array(result,dtype=np.int8)


def prepare_prediction(model,records):
    data=Packed(records,model['spec'],model['config'],model['mean'],model['scale'],training=False)
    return {'data':data,'records':records,'spec':model['spec'],'mean':model['mean'],'scale':model['scale'],
      'windowMs':model['config']['windowMs'],'burdenGate':model['config']['burdenGate'],
      'rules':masks(records,model['config'],'both')}


def predict_modes(model,records,modes=('none','jack','ln','both'),prepared=None):
    prepared=prepare_prediction(model,records)if prepared is None else prepared
    # Supervised configuration knobs do not necessarily change scaler moments.
    # Require exact equality of the effective prediction inputs before reuse.
    if prepared['records']is not records or any(prepared[k]!=model[k]for k in ['spec','mean','scale'])or any(prepared[k]!=model['config'][k]for k in ['windowMs','burdenGate']):
        raise RuntimeError('Prepared prediction inputs differ; repack this context')
    data=prepared['data']
    with torch.no_grad():raw,raw_comp,raw_weight,raw_mass=data.forward(torch.tensor(model['theta']))
    active=data.eligible.numpy();all_rules=prepared['rules'];split=np.cumsum(data.lengths)[:-1];output={}
    for mode in modes:
        if mode not in ['none','jack','ln','both']:raise ValueError('Unknown physical support mode')
        rules=all_rules.copy()
        if mode=='none':rules[:]=-1
        elif mode!='both':rules[rules!=FINE.index(mode)]=-1
        matched=active&(rules>=0);p=raw.numpy().copy()
        if mode=='none':composition=raw_comp.numpy();weights=raw_weight.numpy();mass=raw_mass.numpy()
        else:
            p[matched]=0.;p[np.flatnonzero(matched),rules[matched]]=1.
            a,b=model['theta'][-2:];weights=data.duration.numpy()*active*data.effort.numpy()**a*(p*p).sum(1)**b
            mass=np.bincount(data.bag.numpy(),weights=weights,minlength=len(records));composition=np.zeros((len(records),len(FINE)))
            np.add.at(composition,data.bag.numpy(),p*weights[:,None]);composition/=np.maximum(mass[:,None],1e-300)
        local=np.zeros((len(p),len(FINE)+1));local[active,:-1]=p[active];local[~active,-1]=1.
        output[mode]={'composition':composition,'local':np.split(local,split),'weights':np.split(weights,split),'mass':mass,
              'rules':np.split(np.where(matched,rules,-1),split),
              'structuralWindows':{'jack':int(np.sum(matched&(rules==FINE.index('jack')))),'ln':int(np.sum(matched&(rules==FINE.index('ln'))))}}
    return output


def predict(model,records,mode='both'):
    if mode=='none':return raw_predict(model,records)
    return predict_modes(model,records,[mode])[mode]
