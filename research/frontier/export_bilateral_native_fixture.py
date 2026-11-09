"""Export accepted numerical consumers, without fitting or choosing models."""
from pathlib import Path
import json
import numpy as np
from bilateral_partition_model import World, frames, read
from whole_reference import write


def run(output,kinds):
    world=World('original',0);cases=[]
    root=Path('research/overlay-quality/bilateral-partition-r2/original/0')
    for kind in kinds:
        folder=root/kind/'0';read(folder/'complete.json')
        models=read(folder/'inner-models.json')
        for hid in [1,55,108]:
            selected=next(r for r in models if r['headId']==hid and r['inner']==0)
            model=selected['model']
            for uid in [world.main[0]['uid'],'manual/DS','concept/H01']:
                raw=world.features(uid,kind)
                cases.append(dict(kind=kind,headId=hid,uid=uid,source=world.inputs[uid]['source'],model=model,
                    features=[[float(v)if np.isfinite(v)else None for v in row]for row in raw],
                    original=world.inputs[uid]['frames'],expected=frames(world,uid,model)))
    output=Path(output)
    if output.exists():raise ValueError('Preserve fixture; no redundant regeneration')
    write(output,dict(scope='Predeclared original partition0/outer0/inner0 heads1,55,108; numerical parity only, no selection/accuracy claim.',cases=cases))
    print('Native numerical fixture exported',len(cases),flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='.cache/overlay-bilateral-native-fixture-r1.json')
    parser.add_argument('--kinds',nargs='+',choices=['bias','hand','bilateral','both'],default=['bias','hand','bilateral'])
    args=parser.parse_args();run(args.output,args.kinds)
