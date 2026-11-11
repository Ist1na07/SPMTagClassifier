"""Whole-integral derivatives into actual original W/b and score gain.

Original parent inputs and fixed initial routing; no extra classifier head.
Exact support max-normalization/cap derivative, three existing burden fields.
Hard post-competition rest/gates remain evaluation operations, not derivatives.
"""
from core import *
from scipy.special import expit,logsumexp


class Whole:
    def __init__(self,d):
        self.manifest=read('.cache/local-feedback-activity-r6b/manifest.json');self.records=self.manifest['records'];self.lookup={r['uid']:i for i,r in enumerate(self.records)};self.main=[self.lookup[u]for u in self.manifest['mainUids']];self.fine=self.manifest['fine'];self.profiles=self.manifest['profiles'];self.consumers=self.manifest['consumers'];self.bag=np.zeros(len(d.parents),int)
        for uid,(a,b)in d.offset.items():self.bag[a:b]=self.lookup[uid]
        with np.load('.cache/local-feedback-activity-r6b/arrays.npz')as z:self.mass=np.array([np.bincount(z['globalParent'],weights=v,minlength=len(d.parents))for v in z['mass']])
        self.mass=self.mass[[self.consumers.index(n)for n in self.consumers]];self.weight=np.array([r['weight']for r in self.records]);self.y=np.array([[c in r['tags']for c in self.fine]for r in self.records],float);self.base=np.array([[c in r['legacyOverall']['tags']for c in self.fine]for r in self.records]);self.mixed=np.array([any(c in ['hybrid','rc mix','ln mix']for c in r['tags'])for r in self.records]);self.prior_mixed=np.array([any(c in ['hybrid','rc mix','ln mix']for c in r['legacyOverall']['tags'])for r in self.records])

    def evaluate(self,frames,ids=None,return_vectors=False):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
        from model import metrics
        ids=self.main if ids is None else ids;classes=self.manifest['classes'];y=np.array([[c in r['tags']for c in classes]for r in self.records]);base=np.array([[c in r['legacyOverall']['tags']for c in classes]for r in self.records]);finecols=[classes.index(c)for c in self.fine];mixcols=[classes.index(c)for c in ['hybrid','rc mix','ln mix']];support=np.zeros((len(frames),len(self.fine)));family=np.zeros((len(frames),2))
        for j,f in enumerate(frames):
            if f['family']=='break':continue
            s=f['scores'];den=max(1.,max(s.values(),default=0));support[j]=[min(.999,max(0,s.get(c,0))/den)for c in self.fine];family[j,int(f['family']=='ln')]=1
        out={}
        for pi,name in enumerate(self.consumers):
            mass=self.mass[pi];share=np.zeros((len(self.records),len(self.fine)));families=np.zeros((len(self.records),2));np.add.at(share,self.bag,mass[:,None]*support);np.add.at(families,self.bag,mass[:,None]*family);p=self.profiles[name];gate=p['policy'];q=np.zeros_like(base);q[:,finecols]=(base[:,finecols]&(share>=gate['keepFloor']))|(share>=gate['addFloor']);q[self.prior_mixed]=base[self.prior_mixed];strong=share>=gate['mixOverride'];override=self.prior_mixed&strong.any(1);q[np.flatnonzero(override)[:,None],np.array(finecols)[None,:]]=strong[override];q[np.flatnonzero(override)[:,None],np.array(mixcols)[None,:]]=False
            fp=p['family'];hybrid=(families[:,0]>=fp['minRc'])&(families[:,1]>=fp['minLn'])if fp['enabled']else np.zeros(len(self.records),bool);q[hybrid]=False;q[hybrid,classes.index('hybrid')]=True;out[name]=q if return_vectors else metrics(y[ids],q[ids],self.weight[ids])
        return out


class JointKernel(Kernel):
    def __init__(self,d,whole,family,local_uids,train,*,ridge=.01,whole_weight=1.,learn_gain=False,normalization='exact'):
        super().__init__(d,family,local_uids,ridge=ridge);self.whole=whole;self.whole_weight=whole_weight;self.learn_gain=learn_gain;self.normalization=normalization;self.coef_size=self.size;self.size+=int(learn_gain);b=self.block;oldids=self.ids.copy();observed={int(j)for j in oldids};train=np.array(train,int);self.train=train
        ids=np.flatnonzero(np.isin(whole.bag,train)&np.array([p['baseline']['family']==family for p in d.parents]));self.ids=np.array(sorted(observed|set(ids)));lookup={int(j):i for i,j in enumerate(self.ids)}
        for o in self.obs:o['at']=np.array([lookup[int(oldids[j])]for j in o['at']])
        # Whole-only rows legitimately contribute TRAIN feature scaling here.
        x=b['z'];mean=x[self.ids].mean(0);sd=x[self.ids].std(0);keep=np.flatnonzero(sd>1e-6);self.keep=keep;self.mean=mean;self.sd=sd;self.x=np.c_[np.ones(len(x)),(x[:,keep]-mean[keep])/sd[keep]];self.shape=(self.x.shape[1],len(b['classes'])-1);self.coef_size=int(np.prod(self.shape));self.size=self.coef_size+int(learn_gain);self.training_x=self.x[self.ids]
        self.active=np.array([d.parents[j]['baseline']['family']==family for j in self.ids]);self.own=np.array([whole.fine.index(c)for c in b['names']if c in whole.fine]);self.cols=np.array([b['names'].index(c)for c in b['names']if c in whole.fine]);available=np.zeros(len(whole.records),bool);np.maximum.at(available,whole.bag[self.ids],self.active);w=whole.weight[train]*(~whole.mixed[train])*available[train];self.cw=w/max(w.sum(),1e-30)/len(self.own)
        self.threshold=[]
        for name in whole.consumers:
            gate=whole.profiles[name]['policy'];t=np.where(whole.base[train][:,self.own],gate['keepFloor'],gate['addFloor']);t[whole.prior_mixed[train]]=gate['mixOverride'];self.threshold.append(np.clip(t,1e-6,1-1e-6))

    def objective(self,theta):
        b=self.block;coef=theta[:self.coef_size].reshape(self.shape);gain=1.4*np.exp(theta[-1])if self.learn_gain else 1.4;z=b['logits'][self.ids]+self.training_x@coef@self.contrast.T;p=softmax(z,axis=1);s=gain*p*b['mask'];s[:,b['wild']]=b['fixed'][self.ids,0]
        if self.family=='rc':s=np.c_[s,b['fixed'][self.ids,1]]
        ds=np.zeros_like(s);loss=0.;local_log_gradient=np.zeros_like(s)
        logp=z-logsumexp(z,axis=1,keepdims=True);logs=np.log(gain)+logp
        logs[:,b['wild']]=np.log(b['fixed'][self.ids,0],where=b['fixed'][self.ids,0]>0,out=np.full(len(self.ids),-np.inf))
        if self.family=='rc':logs=np.c_[logs,np.log(b['fixed'][self.ids,1],where=b['fixed'][self.ids,1]>0,out=np.full(len(self.ids),-np.inf))]
        for o in self.obs:
            at=o['at'];w=o['w'];allowed=o['allowed'];terms=logs[at]+np.log(w)[:,None];positive=np.where(allowed,terms,-np.inf);num=logsumexp(positive);den=logsumexp(terms);cost=o['cost'];loss+=cost*(den-num);np.add.at(local_log_gradient,at,cost*(np.exp(terms-den)-np.exp(positive-num)))
        if self.whole_weight:
            den=np.maximum(1,s.max(1))
            if self.normalization=='smooth32':den*=((1/den)**32+np.sum((s/den[:,None])**32,axis=1))**(1/32)
            raw=s/den[:,None];support=(raw if self.normalization=='smooth32'else np.minimum(.999,raw))*self.active[:,None];gs=np.zeros_like(support);bags=self.whole.bag[self.ids];nr=len(self.whole.records);target=self.whole.y[self.train][:,self.own]
            for pi,name in enumerate(self.whole.consumers):
                mass=self.whole.mass[pi,self.ids];share=np.zeros((nr,len(self.own)))
                for j,col in enumerate(self.cols):share[:,j]=np.bincount(bags,weights=mass*support[:,col],minlength=nr)
                unclip=share[self.train];val=np.clip(unclip,1e-12,1-1e-12);t=self.threshold[pi];logits=np.log(val)-np.log1p(-val)-np.log(t)+np.log1p(-t);factor=self.whole_weight/len(self.whole.consumers);loss+=factor*np.sum(self.cw[:,None]*(np.logaddexp(0,logits)-target*logits));g=np.zeros_like(share);g[self.train]=factor*self.cw[:,None]*(expit(logits)-target)/(val*(1-val))*((unclip>1e-12)&(unclip<1-1e-12))
                gs[:,self.cols]+=mass[:,None]*g[bags]
            if self.normalization=='smooth32':
                gs*=self.active[:,None];ds+=gs/den[:,None]-(gs*s).sum(1)[:,None]/(den[:,None]**2)*raw**31
            else:
                gs*=self.active[:,None]*(raw<.999);ds+=gs/den[:,None];winner=s.argmax(1);np.add.at(ds,(np.arange(len(s)),winner),-((den>1)*(gs*s).sum(1))/(den*den))
        log_gradient=(local_log_gradient[:,:len(b['classes'])]+ds[:,:len(b['classes'])]*(gain*p))*b['mask'];dz=log_gradient-p*log_gradient.sum(1,keepdims=True);loss+=.5*self.ridge*np.sum(coef*coef);g=(self.training_x.T@dz@self.contrast+self.ridge*coef).ravel()
        if self.learn_gain:
            value=theta[-1];loss+=.5*self.ridge*value*value;gg=float(np.sum(log_gradient))+self.ridge*value;g=np.r_[g,gg]
        return float(loss),g

    def changes(self,theta):return self.x@theta[:self.coef_size].reshape(self.shape)@self.contrast.T

    def export(self,theta):
        out=super().export(theta[:self.coef_size]);out['scoreGain']=float(1.4*np.exp(theta[-1]))if self.learn_gain else 1.4;return out


def resource(normalization='exact'):
    d=Data();w=Whole(d);plan=read('research/rework/results/mil-stage1-r4/protocol.json')['plans'][0]['inner'][0];train=np.array(w.main)[plan['train']];groups={w.records[i]['group']for i in train};uids={d.manual[i]for i in plan['strongTrain']}|{e['uid']for e in d.extra if d.records[e['uid']]['group']in groups};results=[]
    for joint in [False,True]:
        for free in [False,True]:
            row=dict(wholeWeight=float(joint),learnGain=free,families={})
            for family in ['rc','ln']:
                k=JointKernel(d,w,family,uids,train,ridge=.01,whole_weight=float(joint),learn_gain=free,normalization=normalization);rng=np.random.default_rng(250);theta=rng.normal(0,.005,k.size);v=rng.normal(size=k.size);v/=np.linalg.norm(v);eps=1e-5;loss,g=k.objective(theta);err=abs((k.objective(theta+eps*v)[0]-k.objective(theta-eps*v)[0])/(2*eps)-g@v)
                if err>1e-6:raise ValueError('Joint core gradient '+str(err))
                theta,receipt=fit(k);row['families'][family]=dict(gradientError=float(err),receipt=receipt,model=k.export(theta),dimension=k.size);print('core joint resource',joint,free,family,'gradient',err,'seconds',sum(r['seconds']for r in receipt),flush=True)
            results.append(row)
    write('research/core_retune/results/joint-core-resource-'+normalization+'.json',dict(scope='Resource/derivative check on first INNER TRAIN only; no validation examined. smooth32, when selected, is a conservative smooth normalization TRAIN surrogate; evaluation remains exact native support. Not adoption evidence.',normalization=normalization,results=results))


if __name__=='__main__':resource(sys.argv[1]if len(sys.argv)>1 else'exact')
