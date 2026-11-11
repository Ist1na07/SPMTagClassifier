"""Check every saved selected fit in original model coordinates after arithmetic repair."""
from joint_core import *
from joint_study import CONFIGS,PLANS,ROOT


def theta_from_model(k,model):
    old=k.d.models[k.family];dw=np.array(model['W']).reshape(len(model['classes']),-1)-np.array(old['W']).reshape(len(model['classes']),-1);db=np.array(model['b'])-np.array(old['b']);coefficient=np.vstack([db+dw@k.mean,(dw[:,k.keep]*k.sd[k.keep]).T]);theta=(coefficient@k.contrast).ravel()
    if k.learn_gain:theta=np.r_[theta,np.log(model['scoreGain']/1.4)]
    return theta


def run():
    d=Data();w=Whole(d);max_error=0.;max_gradient=0.;count=0
    for outer,plan in enumerate(PLANS):
        for ci,c in enumerate(CONFIGS):
            path=ROOT/str(outer)/('config-'+str(ci)+'.json')
            if not path.exists():continue
            row=read(path)
            for saved in row['fits']:
                split=plan['inner'][saved['inner']];train=np.array(w.main)[split['train']];uids=set(saved['trainUids'])
                for family in ['rc','ln']:
                    k=JointKernel(d,w,family,uids,train,ridge=c['ridge'],whole_weight=c['wholeWeight'],learn_gain=c['learnGain'],normalization='smooth32');theta=theta_from_model(k,saved['models'][family]);loss,g=k.objective(theta);expected=min(r['loss']for r in saved['receipts'][family]);error=abs(loss-expected);gradient=float(abs(g).max());max_error=max(max_error,error);max_gradient=max(max_gradient,gradient);count+=1
                    if error>1e-8 or gradient>1e-5:raise ValueError('Invalid reuse '+str(path)+' '+family+' '+str((error,gradient)))
    receipt=dict(selectedFamilyFits=count,maxReconstructedLossError=max_error,maxReconstructedGradient=max_gradient,allReusable=True,repair='Equivalent logsumexp local bag objective and direct log-score gradient avoid underflow in failing unaccepted line search. Every saved best model reconstructed in original W/b coordinates and rechecked under repaired objective. Existing protocol retained; numerical execution addendum only.');write(ROOT/'numeric-reuse-check.json',receipt);print(receipt)


if __name__=='__main__':run()
