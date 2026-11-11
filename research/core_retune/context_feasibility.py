"""Context mechanism feasibility; explicitly TRAIN-only, not adoption evidence."""
from core import *


def run():
    results=[]
    for width in [1000,2000,4000]:
        d=Data('.cache/core-retune-local-context-'+str(width)+'.jsonl.gz');baseline=dict(local=d.local(d.baseline),real=d.real(d.baseline));uids=set(d.manual)|{e['uid']for e in d.extra}
        print('width',width,'unfitted',baseline['local']['accuracy'],'TechDense',baseline['local']['techToDenseMs'],flush=True)
        for ridge in [.1,.01,.001]:
            changes={};models={};receipts={}
            for family in ['rc','ln']:
                k=Kernel(d,family,uids,ridge=ridge);theta,receipts[family]=fit(k);changes[family]=k.changes(theta);models[family]=k.export(theta)
            frames=d.frames(changes);row=dict(width=width,ridge=ridge,local=d.local(frames),real=d.real(frames),baseline=baseline,models=models,receipts=receipts);results.append({k:v for k,v in row.items()if k!='models'})
            write('research/core_retune/results/context-feasibility-'+str(width)+'-'+str(ridge)+'.json',row)
            print('width',width,'ridge',ridge,'TRAIN',row['local']['accuracy'],'real',row['real']['accuracy'],'TechDense',row['local']['techToDenseMs'],flush=True)
    write('research/core_retune/results/context-feasibility-summary.json',dict(scope='TRAIN-only mechanism diagnosis, no validation claim; centered physical context replaces whole-parent vector, original expert b/W directly optimized.',results=results))


if __name__=='__main__':run()
