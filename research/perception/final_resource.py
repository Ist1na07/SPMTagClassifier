"""Export fixed shipping-profile replay; no tuning or fresh scoring."""
from pathlib import Path
import sys,json
import numpy as np
from extended import ExtendedData,WEIGHTS
from compact import CompactData
from anchored import base,predictions
from integral import atoms

def run():
    profiles=json.loads(Path('src/models/perception7k.json').read_text())['profiles']
    detail=ExtendedData();compact=CompactData();data={'actions':detail,'detail':detail,'compact':compact};expected={};endpoint=[]
    for name,p in profiles.items():
        d=data[name];wid=WEIGHTS.index(p['recipe']);_,_,share,family=d.representation(wid,p['supportPower']);q=predictions(d,share,family,base(d),[p['policy']],[p['family']])[0,0]
        expected[name]=(share,q)
    output=Path('.cache/perception-final-resource.jsonl');output.write_text('',encoding='utf8')
    for i,line in enumerate(Path('.cache/perception-inputs-r1.jsonl').open(encoding='utf8')):
        row=json.loads(line);ratings={}
        for name,d in [('detail',detail),('compact',compact)]:
            with np.load(d.records[i]['curvePath'])as z:curve={k:z[k].copy()for k in ['query','demand','Rbar','Pbar','Cbar','Jm']}
            p=profiles[name];ratings[name]=dict(schemaVersion=1,keyCount=7,speedRate=1,clock='post-rate-ms',modelIdentity=p['modelIdentity'],query=curve['query'].tolist(),demand=curve['demand'].tolist())
            if row['map']['lastNote']>curve['query'][-1]:
                a=atoms(row,curve,p['recipe']['width'],p['recipe']['hold']);w=a['dt']*a['logDemand']**p['recipe']['alpha']*a['logActions']**p['recipe']['beta'];end=row['map']['firstNote']+np.cumsum(a['dt']);tail=end-a['dt']>=curve['query'][-1]
                remaining=w.copy();remaining[tail]=0;valid=a['frame']>=0;mass=np.bincount(a['frame'][valid],weights=remaining[valid],minlength=len(row['frames']))/max(remaining.sum(),1e-30)
                start,stop=d.records[i]['offset'];s=mass@(d.soft[start:stop]**p['supportPower']);fam=mass@d.arrays['family'][start:stop]
                alt=predictions(d,s[None],fam[None],base(d)[i:i+1],[p['policy']],[p['family']])[0,0,0]
                endpoint.append(dict(uid=row['uid'],profile=name,extensionMs=float(row['map']['lastNote']-curve['query'][-1]),burdenFraction=float(w[tail].sum()/max(w.sum(),1e-30)),maxPercentagePointChange=float(np.abs(s-expected[name][0][i]).max()*100),tagsChanged=bool(np.any(alt!=expected[name][1][i]))))
        case=dict(uid=row['uid'],source=row['source'],ratings=ratings,baseline=row['legacyOverall'],frames=row['frames'],expected={name:dict(percentages=dict(zip(detail.fine,(100*v[0][i]).tolist())),tags=[c for c,y in zip(detail.classes,v[1][i])if y])for name,v in expected.items()})
        with output.open('a',encoding='utf8')as f:f.write(json.dumps(case)+'\n')
    Path('research/perception/results/final-runtime').mkdir(exist_ok=True)
    Path('research/perception/results/final-runtime/endpoint-sensitivity.json').write_text(json.dumps(dict(counterfactual='Remove all burden after the final physical head; sensitivity only, not alternate scoring truth',cases=endpoint),indent=2)+'\n',encoding='utf8')
    print('Fixed-profile resources',i+1,'endpoint cases',len(endpoint))
if __name__=='__main__':run()
