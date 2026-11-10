"""Export existing accepted resource states; no re-fit or scorer calls."""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import competition as c
import routing as r


def run():
    d=r.Data();split=c.PLANS[0]['inner'][0];pack=dict(classes=d.fine,cases={})
    for name,mod,path in [('competition',c,'resource-r2.json'),('routing',r,'routing-resource-r3.json')]:
        saved=c.read('research/local_feedback/results/'+path);k=mod.Kernel(d,d.main[split['train']],d.manual[split['strongTrain']],saved['config']);theta=np.array(min(saved['receipt']['starts'],key=lambda s:s['loss'])['theta']);model=k.export(theta)
        if name=='competition':
            support,scores,_,_=k.forward(theta,all_rows=True);frames=c.frame_readout(d,scores);x=d.x
        else:
            p,support,_,_,_=k.forward(theta,all_rows=True);frames=d.routed_scores(p);x=k.all_x
        names=model['features'];model_x=[dict(zip(names,row))for row in x]
        pack['cases'][name]=dict(model=model,x=model_x,support=support[:,:len(d.fine)].tolist(),frames=frames,
            **(dict(probabilities=p.tolist())if name=='routing'else {}))
    pack['original']=[f for _,f in d.frame_records];pack['candidates']=d.candidates
    c.write('.cache/local-feedback-parity.json',pack)


if __name__=='__main__':run()
