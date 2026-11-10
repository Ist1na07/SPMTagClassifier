"""Oracle for a native-recomputed JS consumer on the complete current inventory."""
from pathlib import Path
import sys,argparse
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from activity import *


def run(path):
    d=Data();saved=read(path);model=saved['model'];k=Kernel(d,d.main,d.manual,model['config']);theta=np.array(min(saved['receipt']['starts'],key=lambda r:r['loss'])['theta']);p,s,*_=k.forward(theta,all_rows=True);frames=d.routed_scores(p)
    export=k.export(theta)
    if export['coefficients']!=model['coefficients']:raise ValueError('Candidate TRAIN scope/export reconstruction mismatch')
    out=Path('.cache/local-feedback-activity-native-oracle.jsonl')
    with out.open('w',encoding='utf8')as f:
        import json
        for r in d.records:
            a,b=r['offset'];f.write(json.dumps(dict(uid=r['uid'],frames=frames[a:b],probabilities=p[a:b].tolist(),support=s[a:b].tolist(),inputs=k.all_x[a:b].tolist()))+'\n')
    write(str(out)+'.schema.json',dict(model=model,classes=d.fine,features=k.features,candidate=path,frames=len(d.route),records=len(d.records)))
    print('native activity oracle',len(d.route),'frames',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('candidate');run(parser.parse_args().candidate)
