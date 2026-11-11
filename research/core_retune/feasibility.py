"""First gate: TRAIN-only expressivity and original-coordinate checks.

Not a model selection study; do not adopt weights from this output.
"""
from core import *


def run():
    d=Data();baseline=dict(local=d.local(d.baseline),real=d.real(d.baseline));uids=set(d.manual)|{e['uid']for e in d.extra};results=[]
    write('research/core_retune/results/feasibility-protocol.json',dict(scope='All labelled local data TRAIN-only diagnostic, NOT validation or a shippable model. Original native b/W directly fitted; frozen route, no external features.',modes=['bias','weights'],ridge=[.1,.01,.001],starts=[0,1],baseline=baseline,source=Path(__file__).read_text(encoding='utf8')))
    for mode in ['bias','weights']:
        for ridge in [.1,.01,.001]:
            changes={};models={};receipts={};gradient={};start=time.monotonic()
            for family in ['rc','ln']:
                k=Kernel(d,family,uids,mode=mode,ridge=ridge);rng=np.random.default_rng(433);theta=rng.normal(0,.01,k.size);v=rng.normal(size=k.size);v/=np.linalg.norm(v);eps=1e-5;l,g=k.objective(theta);fd=(k.objective(theta+eps*v)[0]-k.objective(theta-eps*v)[0])/(2*eps);gradient[family]=abs(float(fd-g@v))
                if gradient[family]>1e-6:raise ValueError('Gradient mismatch')
                theta,receipts[family]=fit(k);changes[family]=k.changes(theta);models[family]=k.export(theta)
                m=models[family];b=d.blocks[family];native=b['z']@np.array(m['W']).reshape(len(m['classes']),-1).T+np.array(m['b']);error=float(np.max(abs(native-(b['logits']+changes[family]))))
                if error>1e-9:raise ValueError('Original b/W coordinate mismatch '+str(error))
            frames=d.frames(changes);row=dict(mode=mode,ridge=ridge,local=d.local(frames),real=d.real(frames),gradientErrors=gradient,receipts=receipts,seconds=time.monotonic()-start,models=models);results.append(row)
            write('research/core_retune/results/feasibility-'+mode+'-'+str(ridge)+'.json',row)
            print(mode,ridge,'TRAIN six',row['local']['accuracy'],'real',row['real']['accuracy'],'TechDense',row['local']['techToDenseMs'],'seconds',row['seconds'],flush=True)
    write('research/core_retune/results/feasibility-summary.json',dict(scope='TRAIN-only expressivity; not heldout improvement. Native score/decision parity checked for8483 parents.',baseline=baseline,results=[{k:v for k,v in r.items()if k!='models'}for r in results]))


if __name__=='__main__':run()
