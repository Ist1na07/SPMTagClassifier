"""Replay selected OUTER models; pool full349 P/R/F1 rather than fold means."""
from joint_core import *
from local_study import PLANS,mean_metrics
ROOT=Path('research/core_retune/results/joint-source-r1')


def model_frames(d,models):
    if not models:return d.baseline
    changes={};gains={}
    for family,m in models.items():
        b=d.blocks[family];changes[family]=b['z']@np.array(m['W']).reshape(len(m['classes']),-1).T+np.array(m['b'])-b['logits'];gains[family]=m['scoreGain']
    return d.frames(changes,gains=gains)


def run():
    read(ROOT/'oof.json');d=Data();w=Whole(d);sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'));from model import metrics
    classes=w.manifest['classes'];y=np.array([[c in r['tags']for c in classes]for r in w.records]);names=list(read(ROOT/'0/outer.json'));result={}
    for name in names+['baseline']:
        predicted={n:np.zeros_like(y)for n in w.consumers};manual=[];real=[];choices=[]
        for outer,plan in enumerate(PLANS):
            state=None if name=='baseline'else read(ROOT/str(outer)/'outer.json')[name];frames=d.baseline if state is None else model_frames(d,state['models']);vectors=w.evaluate(frames,return_vectors=True);ids=np.array(w.main)[plan['test']]
            for n in w.consumers:predicted[n][ids]=vectors[n][ids]
            ms=[d.manual[i]for i in plan['strongTest']];manual.extend(d.local(frames,ms)['details']);groups={w.records[i]['group']for i in ids};es={e['id']for e in d.extra if d.records[e['uid']]['group']in groups};real.extend(r for r in d.real(frames)['details']if r['id']in es);choices.append(-1 if state is None else state['choice']['id'])
        if len(manual)!=6 or len(real)!=8 or len({r['id']for r in real})!=8:raise ValueError('Incomplete source coverage')
        result[name]=dict(local=mean_metrics([dict(manual=dict(details=manual),real=dict(details=real))]),manual=manual,real=real,consumers={n:metrics(y[w.main],q[w.main],w.weight[w.main])for n,q in predicted.items()},choices=choices)
    write(ROOT/'oof-full349.json',dict(scope='Actual exported original b/W and gain, exact native support/gates; pooled full349 weighted P/R/F1. Earlier oof.json consumer fields are mean OUTER-fold metrics, not this full-cohort metric. Initial prior/route fixed as declared; confidence-dependent late rest is recomputed.',metrics=result))
    for name,r in result.items():print(name,r['local'],{n:round(v['macroF1']*100,4)for n,v in r['consumers'].items()},flush=True)


if __name__=='__main__':run()
