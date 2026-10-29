"""Frozen grouped searches; every ablation is refitted and retuned from scratch.

Run after extract.cjs and legacy-features.cjs. No raw charts are published.
The persisted JSON models are consumed directly by the JavaScript runtime.
"""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import json, re, itertools, warnings, time
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from threadpoolctl import threadpool_limits
warnings.filterwarnings('ignore', message='The least populated class')
OUT = Path(os.environ.get('RESULT_DIR','research/results')); OUT.mkdir(exist_ok=True, parents=True)
MODELS = Path(os.environ.get('MODEL_DIR','src/models')); MODELS.mkdir(exist_ok=True, parents=True)
SCHEMA = json.loads(Path('.cache/feature-schema.json').read_text())
KEYS = SCHEMA['FEATURE_KEYS']; GROUPS = SCHEMA['GROUPS']; D = len(KEYS)
LOCAL7 = ['chordjack','minijack','dense chordstream','fast chordstream','speed','tech','vibro',
          'coordination','density','inverse','release','technical','wildcard','ln wildcard','break']
LOCAL4 = ['jack','ln','speed','stamina','tech']
GRID = {'C':[0.03,0.3,3.0], 'classWeight':[None,'balanced'],
        'windowMs':[2000,4000,8000], 'threshold':[0.25,0.4,0.55],
        'temperature':[0.5,1.0,2.0], 'smoothingPenalty':[0.0,1.0,3.0]}
VARIANTS = ['full','base_only','no_geometry','no_rhythm','no_holds','no_sequence','no_burden',
            'hard_evidence','duration_evidence','effort_evidence','confidence_evidence','no_evidence','legacy_features']
if os.environ.get('VARIANTS'):VARIANTS=os.environ['VARIANTS'].split(',')
def dump(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def canonical(s):
    s = re.sub(r'\b(?:x\s*)?\d+(?:\.\d+)?\s*x\b|\bx\s*\d+(?:\.\d+)?\b','',s.lower())
    return re.sub(r'[^\w]+','',s)
def load():
    data=[]; inventory=[]
    with open(os.environ.get('FEATURE_INPUT','.cache/features.jsonl'),encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            inventory.append({k:r.get(k) for k in ['uid','keys','duplicateOf','dataset']})
            if r['keys']==6 or 'apolar' in r['dataset']: continue
            for win,ws in r['windows'].items():
                r['windows'][win]={'x':np.array([w['values'] for w in ws]),
                    'start':np.array([w['start'] for w in ws]),'end':np.array([w['end'] for w in ws]),
                    'n':np.array([w['notes'] for w in ws])}
            data.append(r)
    return data,inventory
class Union:
    def __init__(self,ids): self.p={x:x for x in ids}
    def find(self,x):
        if self.p[x]!=x:self.p[x]=self.find(self.p[x])
        return self.p[x]
    def join(self,a,b):
        if a in self.p and b in self.p:self.p[self.find(b)]=self.find(a)
def group(data):
    u=Union([r['uid'] for r in data]); identities={}
    for r in data:
        if r.get('duplicateOf'):u.join(r['uid'],r['duplicateOf'])
        m=r['meta']; title=canonical(m.get('Title','')); artist=canonical(m.get('Artist',''))
        ids=[('song',r['keys'],artist,title)] if title else []
        for k in ['BeatmapID','BeatmapSetID']:
            v=m.get(k,'');
            if v.isdigit() and int(v)>0:ids.append((k,r['keys'],v))
        b=r.get('extra',{}).get('base_title')
        if b:ids.append(('base',r['keys'],canonical(b)))
        for identity in ids:
            if identity in identities:u.join(r['uid'],identities[identity])
            else:identities[identity]=r['uid']
    for r in data:r['groupId']=u.find(r['uid'])
    manual={r['groupId'] for r in data if r['dataset']=='manual'}
    return manual
def target(r):
    if r['keys']==4:return [r['subset']] if r['subset'] in LOCAL4 else []
    return sorted(set('dense chordstream' if t=='chordstream' else t for t in r['tags']))
def split(records):
    groups=np.array([r['groupId'] for r in records]); counts=Counter(t for r in records for t in target(r))
    y=[min(target(r),key=lambda t:counts[t]) for r in records]
    folds=list(StratifiedGroupKFold(5,shuffle=True,random_state=20261030).split(np.zeros(len(y)),y,groups))
    test=set(folds[0][1]); val=set(folds[1][1]); train=set(range(len(y)))-test-val
    return [np.array(sorted(s)) for s in (train,val,test)]
def feature_indices(variant):
    if variant=='base_only':return [KEYS.index(k) for k in GROUPS['base']]
    drop=variant[3:] if variant.startswith('no_') and variant[3:] in GROUPS else None
    return [i for i,k in enumerate(KEYS) if (not drop or k not in GROUPS[drop])
            and not (variant in ['no_geometry','no_holds'] and k=='effort')]
def fit_model(x,y,classes,C,cw,weights=None,multilabel=False):
    weights=np.ones(len(x)) if weights is None else weights
    scaler=StandardScaler().fit(x,sample_weight=weights); z=scaler.transform(x)
    coefs=[];bias=[]
    if multilabel:
        for i,c in enumerate(classes):
            yi=np.array([int(c in t) for t in y]); prior=(np.dot(weights,yi)+.5)/(weights.sum()+1)
            if len(np.unique(yi))<2:
                coefs.append(np.zeros(x.shape[1]));bias.append(float(np.log(prior/(1-prior))));continue
            clf=LogisticRegression(C=C,class_weight=cw,max_iter=1000,tol=1e-6).fit(z,yi,sample_weight=weights)
            coefs.append(clf.coef_[0]);bias.append(clf.intercept_[0])
        W=np.array(coefs);b=np.array(bias)
    else:
        clf=LogisticRegression(C=C,class_weight=cw,max_iter=1000,tol=1e-6).fit(z,y,sample_weight=weights)
        W=np.zeros((len(classes),x.shape[1])); b=np.full(len(classes),-30.)
        if len(clf.classes_)==2:
            for c,sign in zip(clf.classes_,[-.5,.5]):
                j=classes.index(c);W[j]=clf.coef_[0]*sign;b[j]=clf.intercept_[0]*sign
        else:
            for c,w,bb in zip(clf.classes_,clf.coef_,clf.intercept_):j=classes.index(c);W[j]=w;b[j]=bb
    return {'classes':classes,'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist(),
            'W':W.tolist(),'b':b.tolist(),'multilabel':multilabel,'temperature':1.0,'C':C,'classWeight':cw}
def predict(m,x,temperature=None):
    z=((x-np.array(m['mean']))/np.array(m['scale']))@np.array(m['W']).T+np.array(m['b'])
    z=z/(temperature or m.get('temperature',1))
    if m['multilabel']:return 1/(1+np.exp(-np.clip(z,-700,700)))
    z-=z.max(axis=1,keepdims=True);p=np.exp(z);return p/p.sum(axis=1,keepdims=True)
def weak_samples(records,win,idx,classes):
    xs=[];ys=[];weights=[]
    for r in records:
        labels=target(r);labels=[t for t in labels if t in classes and t not in ['break','wildcard','ln wildcard']]
        if len(labels)!=1:continue
        # A whole-map mix label is never copied onto individual intervals.
        w=r['windows'][str(win)];x=w['x'];mask=(w['n']>=np.quantile(w['n'],.25))&(w['n']>0)
        x=x[mask][:,idx]
        if not len(x):continue
        xs.extend(x);ys.extend([labels[0]]*len(x));weights.extend([1/len(x)]*len(x))
    return xs,ys,weights
def annotation_at(r,t):
    for a in r.get('annotations',[]):
        if a['start']<=t<a['end']:return a
    return None
def strong_samples(records,win,idx,classes):
    xs=[];ys=[];weights=[]
    for r in records:
        ws=r['windows'][str(win)]
        for a in r['annotations']:
            ts=(ws['start']+ws['end'])/2;chosen=np.flatnonzero((ts>=a['start'])&(ts<a['end']))
            labels=[t for t in a['tags'] if t in classes]
            for j in chosen:
                for t in labels:
                    xs.append(ws['x'][j,idx]);ys.append(t);weights.append(1/max(1,len(chosen)*len(labels)))
    return xs,ys,weights
def local_fit(records,manual,win,idx,classes,C,cw):
    x,y,w=weak_samples(records,win,idx,classes); a,b,c=strong_samples(manual,win,idx,classes)
    return fit_model(np.array(x+a),y+b,classes,C,cw,np.array(w+c))
def aggregate(x,p,mode):
    if mode in ['no_geometry','no_holds','no_burden']:mode='confidence_evidence'
    if mode not in ['hard_evidence','duration_evidence','effort_evidence','confidence_evidence']:mode='full'
    weight=np.ones(len(x));effort=x[:,KEYS.index('effort')]
    if mode in ['full','effort_evidence']:weight*=np.maximum(effort,1e-8)
    if mode in ['full','confidence_evidence']:weight*=p.max(axis=1)
    if mode=='hard_evidence':p=np.eye(p.shape[1])[p.argmax(axis=1)]
    weight/=max(weight.sum(),1e-12)
    return (p*weight[:,None]).sum(axis=0)
def pool(r,win,idx,local,mode):
    if mode=='legacy_features':return np.array(r['legacyVector'])
    x=r['windows'][str(win)]['x'];base=x[:,idx]
    v=np.r_[base.mean(axis=0),base.std(axis=0),np.quantile(base,.9,axis=0)]
    if mode!='no_evidence':
        p=predict(local,base);ag=aggregate(x,p,mode)
        v=np.r_[v,ag]
    return v
def encode(labels,classes):return np.array([[int(c in y) for c in classes] for y in labels])
def threshold(p,t):
    b=p>=t
    for i in np.flatnonzero(~b.any(axis=1)):b[i,np.argmax(p[i])]=True
    return b.astype(int)
def measures(y,p,t,classes):
    pred=threshold(p,t)
    return {'macroF1':float(f1_score(y,pred,average='macro',zero_division=0)),
      'microF1':float(f1_score(y,pred,average='micro',zero_division=0)),
      'sampleF1':float(f1_score(y,pred,average='samples',zero_division=0)),
      'exactMatch':float(accuracy_score(y,pred)),
      'brier':float(np.mean((y-p)**2)),
      'perClass':classification_report(y,pred,target_names=classes,output_dict=True,zero_division=0)}
def local_metric(records,win,idx,m,classes):
    truth=[];probs=[]
    for r in records:
        labels=target(r)
        if len(labels)!=1 or labels[0] not in classes:continue
        x=r['windows'][str(win)]['x'];p=predict(m,x[:,idx]);truth.append(classes.index(labels[0]));probs.append(aggregate(x,p,'full'))
    return float(f1_score(truth,np.argmax(probs,axis=1),labels=list(range(len(classes))),average='macro',zero_division=0)) if truth else 0
def run_maps(records,manual,key):
    if os.environ.get('SPLIT_INPUT_DIR'):
        frozen=json.loads((Path(os.environ['SPLIT_INPUT_DIR'])/f'{key}k-split.json').read_text(encoding='utf-8'))
        splits=[np.array([i for i,r in enumerate(records) if frozen[r['uid']]['split']==s]) for s in ['train','validation','test']]
    else:splits=split(records)
    train,val,test=[[records[i] for i in ids] for ids in splits]
    assert not ({r['groupId'] for r in train}&{r['groupId'] for r in val})
    assert not ({r['groupId'] for r in train+val}&{r['groupId'] for r in test})
    classes=sorted({t for r in records for t in target(r)});localclasses=LOCAL7 if key==7 else LOCAL4
    labels=[target(r) for r in records]; y=encode(labels,classes)
    dump(OUT/f'{key}k-split.json',{r['uid']:{'group':r['groupId'],'split':s} for s,rs in [('train',train),('validation',val),('test',test)] for r in rs})
    search=[];selected={};candidates={}
    for variant in VARIANTS:
        idx=feature_indices(variant);best=None
        wins=GRID['windowMs'] if variant!='legacy_features' else [4000]
        for win in wins:
            local=None;ls=-1; localparams=None
            if variant not in ['legacy_features','no_evidence']:
                for C,cw in itertools.product(GRID['C'],GRID['classWeight']):
                    m=local_fit(train,manual,win,idx,localclasses,C,cw)
                    score=local_metric(val,win,idx,m,localclasses)
                    search.append({'stage':'local','variant':variant,'win':win,'C':C,'classWeight':cw,'score':score})
                    if score>ls:local,ls,localparams=m,score,{'C':C,'classWeight':cw}
            x=np.array([pool(r,win,idx,local,variant) for r in records]);tr,va,te=splits
            for C,cw in itertools.product(GRID['C'],GRID['classWeight']):
                m=fit_model(x[tr],[labels[i] for i in tr],classes,C,cw,multilabel=True);p=predict(m,x[va])
                for t in GRID['threshold']:
                    score=measures(y[va],p,t,classes)['macroF1']
                    config={'variant':variant,'windowMs':win,'C':C,'classWeight':cw,'threshold':t,'local':localparams}
                    search.append({'stage':'map',**config,'score':score})
                    if best is None or score>best[0]:best=(score,config,idx,local,m,x)
        score,config,idx,local,m,x=best; selected[variant]={'validationMacroF1':score,**config}
        candidates[variant]=best
        print(key,'map',variant,'val',round(score,4),config,flush=True)
        dump(OUT/f'{key}k-search.json',search)
    # Legacy features are a fair retrained comparator, not a key-normalized deployment candidate.
    winner=max((v for v in selected if v!='legacy_features'),key=lambda v:selected[v]['validationMacroF1'])
    # Immutable selection is written before any test metric is computed.
    dump(OUT/f'{key}k-selection.json',{'winner':winner,'variants':selected,'grid':GRID,'testOpened':False})
    results={};predictions=[]
    for variant,best in candidates.items():
        score,config,idx,local,m,x=best;tr,va,te=splits
        p=predict(m,x[te]);results[variant]={'validationMacroF1':score,**measures(y[te],p,config['threshold'],classes)}
        pred=threshold(p,config['threshold'])
        for j,i in enumerate(te):predictions.append({'variant':variant,'uid':records[i]['uid'],'group':records[i]['groupId'],
            'truth':labels[i],'predicted':[c for c,b in zip(classes,pred[j]) if b],'scores':dict(zip(classes,p[j].tolist()))})
        if variant==winner:
            # Acceptance model is trained only on training data, exactly as tested.
            deploy={'keyCount':key,'variant':variant,'indices':idx,'windowMs':config['windowMs'],'stepMs':2000,
                    'rowMergeMs':float(os.environ.get('ROW_MERGE_MS',10)),
                    'threshold':config['threshold'],'local':local,'map':m,'featureKeys':KEYS,
                    'trainingScope':'training split only; validation selects parameters; test untouched until selection',
                    'evidenceLevel':'weak whole-map labels' if key==4 else 'whole-map labels plus six manually segmented training charts'}
            dump(MODELS/f'{key}k.json',deploy)
    if key==7 and all('legacy' in r for r in records):
        tr,va,te=splits;p=[]
        for i in te:
            tags=records[i]['legacy']['overall']['tags'];p.append([float(c in tags) for c in classes])
        results['v1.1.0_untouched']=measures(y[te],np.array(p),.5,classes)
    dump(OUT/f'{key}k-metrics.json',{'winner':winner,'classes':classes,'counts':dict(Counter(t for r in records for t in target(r))),
        'records':len(records),'splitCounts':[len(s) for s in splits],'groupCounts':[len({records[i]['groupId'] for i in s}) for s in splits],
        'results':results})
    dump(OUT/f'{key}k-test-predictions.json',predictions)
    return train,val,test,selected

def viterbi(p,penalty):
    cost=-np.log(np.maximum(p,1e-15));score=cost[0].copy();backs=[]
    for c in cost[1:]:
        trans=score[:,None]+penalty*(1-np.eye(len(score)));prev=trans.argmin(axis=0)
        score=c+trans[prev,np.arange(len(score))];backs.append(prev)
    out=[int(score.argmin())]
    for b in reversed(backs):out.append(int(b[out[-1]]))
    return np.array(out[::-1])
def manual_eval(rs,win,idx,m,T,penalty):
    correct=weight=loss=0.;rows=[]
    for r in rs:
        w=r['windows'][str(win)];p=predict(m,w['x'][:,idx],T);pred=viterbi(p,penalty)
        silent=(w['n']==0)&(w['x'][:,KEYS.index('occupancy')]==0)
        if 'break' in m['classes']:
            p[silent]=0;p[silent,m['classes'].index('break')]=1;pred=viterbi(p,penalty)
        for j,(start,end) in enumerate(zip(w['start'],w['end'])):
            # Integrate intersection with every annotation, not one sampled timestamp.
            for a in r['annotations']:
                dur=max(0,min(end,a['end'])-max(start,a['start']))
                if not dur:continue
                tags=a['tags'];tag=m['classes'][pred[j]];ok=tag in tags
                allowed=[m['classes'].index(t) for t in tags if t in m['classes']]
                mass=float(p[j,allowed].sum()) if allowed else 0.
                correct+=dur*ok;weight+=dur;loss-=dur*np.log(max(mass,1e-15))
                rows.append({'uid':r['uid'],'start':float(max(start,a['start'])),'end':float(min(end,a['end'])),
                    'truth':tags,'predicted':tag,'correct':ok,'scores':dict(zip(m['classes'],p[j].tolist()))})
    return correct/max(weight,1),loss/max(weight,1),rows
def run_manual(weak,manual):
    results=[];allpred=[];search=[]
    # Outer six leave-one-chart-out tests, one rotating inner validation chart.
    # All candidates are selected from validation before scoring that outer test.
    variants=['full','base_only','no_geometry','no_rhythm','no_holds','no_sequence','no_burden']
    for variant in variants:
      idx=feature_indices(variant)
      for outer in range(len(manual)):
        val=manual[(outer+1)%len(manual)];test=manual[outer]
        train=[r for r in manual if r['uid'] not in [test['uid'],val['uid']]]
        best=None
        for win,C,cw in itertools.product(GRID['windowMs'],GRID['C'],GRID['classWeight']):
          m=local_fit(weak,train,win,idx,LOCAL7,C,cw)
          for T,penalty in itertools.product(GRID['temperature'],GRID['smoothingPenalty']):
            acc,nll,_=manual_eval([val],win,idx,m,T,penalty)
            config={'variant':variant,'outer':test['id'],'validation':val['id'],'windowMs':win,'C':C,'classWeight':cw,'temperature':T,'smoothingPenalty':penalty}
            search.append({**config,'validationAccuracy':acc,'validationNll':nll})
            if best is None or (acc,-nll)>best[:2]:best=(acc,-nll,config,m)
        _,_,config,m=best
        acc,nll,rows=manual_eval([test],config['windowMs'],idx,m,config['temperature'],config['smoothingPenalty'])
        results.append({**config,'testAccuracy':acc,'testNll':nll,'labelledMs':sum(r['end']-r['start'] for r in rows)})
        allpred.extend({**r,'variant':variant} for r in rows)
        print('segment',variant,test['id'],round(acc,4),flush=True)
        dump(OUT/'segment-search.json',search);dump(OUT/'segment-folds.json',results)
    dump(OUT/'segment-test-predictions.json',allpred)
    # Baseline is evaluated on precisely the same labelled intervals.
    base=[]
    for r in manual:
      good=duration=0
      for a in r['annotations']:
       for s in r['legacy']['segments']:
        d=max(0,min(a['end'],s['end'])-max(a['start'],s['start']));duration+=d;good+=d*(s['tag'] in a['tags'])
      base.append({'chart':r['id'],'accuracy':good/max(1,duration),'labelledMs':duration})
    dump(OUT/'segment-baseline.json',base)
    # Choose the display head using the mean INNER validation score at each configuration.
    # Outer test values are deliberately not consulted.
    grouped=defaultdict(list)
    for s in search:
      if s['variant']=='full':
        k=tuple(s[x] for x in ['windowMs','C','classWeight','temperature','smoothingPenalty'])
        grouped[k].append((s['validationAccuracy'],s['validationNll']))
    chosen=max(grouped,key=lambda k:(np.mean([x[0] for x in grouped[k]]),-np.mean([x[1] for x in grouped[k]])))
    win,C,cw,T,penalty=chosen
    model=local_fit(weak,manual,win,list(range(D)),LOCAL7,C,cw);model['temperature']=T
    dump(MODELS/'segment7k.json',{'model':model,'indices':list(range(D)),'windowMs':win,'stepMs':2000,
         'smoothingPenalty':penalty,'selection':'mean inner-validation accuracy; NLL tie break; outer test not consulted'})
    # Deployment segment head uses hyperparameters selected in the map study.
    # Manual OOF results are evaluation only; no outer test-driven retuning.
    return results

def main():
    dump(OUT/'protocol.json',{'seed':20261030,'grid':GRID,'variants':VARIANTS,
        'mapSelection':'maximum validation macro multi-label F1; deterministic first tie',
        'deploymentEligibility':'all normalized variants; legacy_features is a retrained comparator only',
        'segmentSelection':'maximum inner validation duration acceptable-label accuracy; NLL tie break',
        'grouping':'transitive union of canonical song, base title, positive beatmap/set IDs, rate-normalized event duplicates',
        'sixK':'no pattern labels; interpolation/invariants/corpus audit only',
        'trainingScope':'ship the evaluated training-only fit, not a silently refitted all-data model'})
    data,inventory=load(); manualgroups=group(data);manual=[r for r in data if r['dataset']=='manual']
    legacy={json.loads(l)['uid']:json.loads(l)['vector'] for l in open('.cache/legacy-features.jsonl',encoding='utf-8')}
    for r in data:
        if r['uid'] in legacy:r['legacyVector']=legacy[r['uid']]
    records={k:[r for r in data if r['keys']==k and r['dataset']!='manual' and target(r) and r['groupId'] not in manualgroups] for k in [4,7]}
    dump(OUT/'exclusions.json',{'manualGroupsExcludedFromMapStudy':sorted(manualgroups),
        'excluded':[{'uid':r['uid'],'reason':'manual-chart group' if r['groupId'] in manualgroups else 'no pattern label / course'}
                    for r in data if r['dataset']!='manual' and (not target(r) or r['groupId'] in manualgroups)]})
    train7,_,_,_=run_maps(records[7],manual,7)
    run_maps(records[4],[],4)
    if not os.environ.get('SKIP_MANUAL'):run_manual(train7,manual)
    print('All searches and frozen test evaluations complete',flush=True)
if __name__=='__main__':
    with threadpool_limits(limits=1):main()
