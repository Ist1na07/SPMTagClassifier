"""Three exact consumer marginals on genuinely local route intervals."""
from pathlib import Path
import sys,json,time
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from route_consumers import Consumers,read,write
from integral import atoms
ROOT=Path('.cache/local-feedback-activity-r6')


def run():
    if(ROOT/'complete.json').exists():return
    d=Consumers();schema=read('.cache/local-feedback-activity-r6.jsonl.schema.json');compact=read('.cache/perception-compact-r2/manifest.json');records=[];blocks=[];offset=0;maximum=0.;begin=time.monotonic()
    inputs=Path('.cache/perception-inputs-r1.jsonl').open(encoding='utf8');local=Path('.cache/local-feedback-activity-r6.jsonl').open(encoding='utf8')
    for i,(meta,a,b)in enumerate(zip(d.records,inputs,local)):
        row=json.loads(a);new=json.loads(b)
        if row['uid']!=meta['uid']or new['uid']!=meta['uid']:raise ValueError('Activity input inventory mismatch')
        frames=new['frames'];parent=np.array([f['parent']for f in frames]);row['frames']=frames;mass=[]
        for name in ['detail','actions','compact']:
            path=compact['curvePaths'][meta['uid']]if name=='compact'else meta['curvePath']
            with np.load(path)as z:curve={k:z[k].copy()for k in ['query','demand','Rbar','Pbar','Cbar','Jm']}
            recipe=d.profiles[name]['recipe'];atom=atoms(row,curve,recipe['width'],recipe['hold']);weight=atom['dt']*atom['logDemand']**recipe['alpha']*atom['logActions']**recipe['beta'];valid=atom['frame']>=0;mm=np.bincount(atom['frame'][valid],weights=weight[valid],minlength=len(frames))/max(weight.sum(),1e-30);mass.append(mm)
            old_a,old_b=meta['offset'];err=float(np.max(np.abs(np.bincount(parent,weights=mm,minlength=old_b-old_a)-d.masses[name][old_a:old_b])));maximum=max(maximum,err)
            if err>1e-12:raise ValueError('Refinement changed consumer burden integral')
        old_a,old_b=meta['offset'];blocks.append(dict(mass=np.array(mass),x=np.array(new['values']),parent=parent,globalParent=old_a+parent,
            dt=np.array([f['end']-f['start']for f in frames]),route=d.route[old_a+parent],expert=d.expert[old_a+parent]))
        records.append(dict(meta,offset=[offset,offset+len(frames)],frames=frames));offset+=len(frames)
        if len(records)%75==0:print('activity exact marginals',len(records),offset,flush=True)
    ROOT.mkdir(exist_ok=True);np.savez_compressed(ROOT/'arrays.npz',**{k:np.concatenate([b[k]for b in blocks],axis=1 if k in ['mass','x']else 0)for k in blocks[0]},bag=np.concatenate([np.full(len(b['dt']),i)for i,b in enumerate(blocks)]))
    write(ROOT/'manifest.json',dict(records=records,mainUids=[d.records[i]['uid']for i in d.main],manualUids=[d.records[i]['uid']for i in d.manual],classes=d.classes,fine=d.fine,profiles=d.profiles,consumers=['detail','actions','compact'],featureKeys=schema['featureKeys'],widths=schema['widths'],parentIntegralMaximumError=maximum))
    write(ROOT/'complete.json',dict(records=len(records),frames=offset,newScorerCalls=0,seconds=time.monotonic()-begin,parentIntegralMaximumError=maximum));print('complete activity marginals',offset,maximum,flush=True)


if __name__=='__main__':run()
