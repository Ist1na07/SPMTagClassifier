"""Export an oracle for native physical extraction and JS type allocation."""
from pathlib import Path
import os,sys,argparse,json
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from joint_activity import Data,Kernel,read,write
from competition import allocate as independent_projection
from scipy.special import softmax


def run(path):
    d=Data();saved=read(path);model=saved['model'];cfg=model['config'];names,x=d.inputs[cfg['widthId'],cfg['context'],cfg['obligations']]
    if names!=model['features']:raise ValueError('Native feature order changed')
    # Exported raw-feature coefficients are the effective artifact. Do not
    # recreate a whitening eigensystem merely to read that artifact: signs or
    # roundoff can vary across BLAS backends without changing its meaning.
    residual=[np.c_[np.ones(len(x)),x]@np.array(b['coefficients']).T for b in model['blocks']];logits=residual[0]+np.eye(3)[d.route]*model['priorStrength'];eligible=np.ones((len(x),3),bool)
    if cfg.get('physicalEligibility',False):
        eligible[:,1]=x[:,names.index('npsPerKey')]>0;eligible[:,2]=np.any(x[:,[names.index(k)for k in ['lnHeadFraction','occupancy','tailRate']]]>0,axis=1);logits=np.where(eligible,logits,-np.inf)
    p=softmax(logits,axis=1);s=np.zeros((len(x),len(d.fine)));qs=[]
    for family,block in enumerate(model['blocks'][1:]):
        q0=[]
        for pair in d.expert_raw:
            raw=pair[family];den=max(1.,max(raw.values(),default=0.));q0.append([min(.999,max(0.,raw.get(c,0.))/den)for c in block['classes']])
        q0=np.array(q0);prior=np.full_like(q0,-np.inf);np.log(q0,out=prior,where=q0>0);prior-=np.log1p(-q0);q=q0.copy();valid=eligible[:,family+1];q[valid]=independent_projection(prior[valid],residual[family+1][valid],q0[valid].sum(1));qs.append(q)
        for j,c in enumerate(block['classes']):
            if c in d.fine:s[:,d.fine.index(c)]=p[:,family+1]*q[:,j]
    frames=[]
    for j,(_,old)in enumerate(d.frame_records):
        route=int(p[j].argmax())
        if not route:frames.append(dict(old,tag='break',family='break',secondary=None,scores={'break':1.},confidence=float(p[j,0])));continue
        scores=dict(zip(model['blocks'][route]['classes'],qs[route-1][j].tolist()));ranked=sorted(scores,key=lambda c:-scores[c]);tag=ranked[0];secondary=ranked[1]if scores[ranked[1]]>=.75*scores[tag]and scores[ranked[1]]>.3 else None;frames.append(dict(old,tag=tag,family=['break','rc','ln'][route],secondary=secondary,scores=scores,confidence=float(scores[tag]*p[j,route])))
    out=Path('.cache/local-feedback-joint-native-oracle.jsonl')
    with out.open('w',encoding='utf8')as f:
        for r in d.records:
            a,b=r['offset'];f.write(json.dumps(dict(uid=r['uid'],frames=frames[a:b],probabilities=p[a:b].tolist(),support=s[a:b].tolist(),inputs=x[a:b].tolist()))+'\n')
    write(str(out)+'.schema.json',dict(model=model,classes=d.fine,features=names,candidate=path,records=len(d.records),frames=len(d.route)));print('joint native oracle',len(d.records),len(d.route),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('candidate');run(parser.parse_args().candidate)
