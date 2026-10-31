"""Extra subtype: dominant-finger Anchor *within labelled 4K jack charts*.

All feature ablations refit scaling and logistic regression and jointly select
window, C, class weight and threshold. No 7K label participates in selection.
"""
from mil import canonical,grouped_folds,dump
import json,itertools
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,precision_score,recall_score,average_precision_score
OUT=Path('research/rework/results/anchor')
DATA=json.loads(Path('.cache/rework-anchor.json').read_text(encoding='utf8'))
FEATURES=DATA['features']
FEATURE_SETS={'peak_only':['peakShare'],'concentration_only':['peakExcess','repeatConcentration'],
 'full':['peakExcess','repeatConcentration','runAsymmetry','runCoverage','regularity','regularAnchor'],
 'no_runs':['peakExcess','repeatConcentration','regularity','regularAnchor'],
 'no_timing':['peakExcess','repeatConcentration','runAsymmetry','runCoverage'],
 'no_interaction':['peakExcess','repeatConcentration','runAsymmetry','runCoverage','regularity']}
GRID={'windowMs':[2000,4000,8000],'C':[.01,.1,1.,10.],'classWeight':[None,'balanced']}
THRESHOLDS=[.2,.35,.5,.65,.8]
def groups(data):
 parent={r['uid']:r['uid']for r in data};seen={}
 def find(x):
  if parent[x]!=x:parent[x]=find(parent[x])
  return parent[x]
 for r in data:
  ids=[('song',canonical(r['meta'].get('Artist','')),canonical(r['meta'].get('Title','')))]
  for k in ['BeatmapID','BeatmapSetID']:
   v=r['meta'].get(k,'')
   if v.isdigit()and int(v)>0:ids.append((k,v))
  if r.get('extra',{}).get('base_title'):ids.append(('base',canonical(r['extra']['base_title'])))
  if r.get('duplicateOf'):parent[find(r['uid'])]=find(r['duplicateOf'])
  for identity in ids:
   if identity in seen:parent[find(r['uid'])]=find(seen[identity])
   else:seen[identity]=r['uid']
 for r in data:r['group']=find(r['uid'])
def score(y,p,w,t):
 pred=p>=t
 return {'macroF1':float(f1_score(y,pred,average='macro',sample_weight=w,zero_division=0)),
  'positiveF1':float(f1_score(y,pred,sample_weight=w,zero_division=0)),
  'precision':float(precision_score(y,pred,sample_weight=w,zero_division=0)),
  'recall':float(recall_score(y,pred,sample_weight=w,zero_division=0)),
  'averagePrecision':float(average_precision_score(y,p,sample_weight=w))}
def indices(variant):return [i for i,k in enumerate(FEATURES)if k.split(':')[0]in FEATURE_SETS[variant]]
def fit(records,variant,cfg):
 idx=indices(variant);x=np.array([r['features'][str(cfg['windowMs'])]for r in records])[:,idx]
 y=np.array([r['target']for r in records]);w=np.array([r['weight']for r in records])
 scale=StandardScaler().fit(x,sample_weight=w)
 cw=None if cfg['classWeight']is None else {k:w.sum()/(2*w[y==k].sum())for k in [0,1]}
 model=LogisticRegression(C=cfg['C'],class_weight=cw,max_iter=2000,tol=1e-8).fit(scale.transform(x),y,sample_weight=w)
 return scale,model
def predict(records,variant,cfg,scale,model):
 x=np.array([r['features'][str(cfg['windowMs'])]for r in records])[:,indices(variant)]
 return model.predict_proba(scale.transform(x))[:,1]
def main():
 records=DATA['records'];groups(records)
 source=[r for r in records if r['dataset']=='4k'];transfer=[r for r in records if r['dataset']!='4k']
 source_counts=Counter(r['file']for r in source)
 for r in source:r['weight']=1/source_counts[r['file']];r['tags']=['anchor'if r['target']else'other jack']
 folds=grouped_folds(source,6);configs=[dict(zip(GRID,v))for v in itertools.product(*GRID.values())]
 plans=[]
 for outer,test in enumerate(folds):
  tr=[i for i in range(len(source))if i not in set(test)]
  inner=grouped_folds([source[i]for i in tr],3)
  plans.append({'train':tr,'test':test,'inner':[[tr[i]for i in fold]for fold in inner]})
 dump(OUT/'protocol.json',{'task':'Anchor chordjack versus other named jack subtypes, conditional on the source jack bucket',
  'sourceRecords':len(source),'positiveRecords':sum(r['target']for r in source),
  'sourceGroups':len({r['group']for r in source}),'positiveGroups':len({r['group']for r in source if r['target']}),
  'exclusions':'7 Unsigned jack records are excluded; all non-jack buckets outside this subtype task',
  'grid':GRID,'thresholds':THRESHOLDS,'featureSets':FEATURE_SETS,'selection':'pooled inner binary macro-F1',
  'plans':plans,'records':[{'index':i,'uid':r['uid'],'group':r['group'],'target':r['target'],'weight':r['weight']}for i,r in enumerate(source)],
  'transfer':'7K labels never used in fitting/selection. Missing Anchor tags are NOT negative truth. Mark cross-key source-song overlap.'})
 search=[];choices=[];outerpred=[]
 for variant in FEATURE_SETS:
  for outer,plan in enumerate(plans):
   best=None
   for cid,cfg in enumerate(configs):
    ids=[];probs=[]
    for va in plan['inner']:
     tr=[i for i in plan['train']if i not in set(va)]
     scale,model=fit([source[i]for i in tr],variant,cfg)
     ids.extend(va);probs.extend(predict([source[i]for i in va],variant,cfg,scale,model).tolist())
    y=np.array([source[i]['target']for i in ids]);w=np.array([source[i]['weight']for i in ids]);p=np.array(probs)
    for t in THRESHOLDS:
     metric=score(y,p,w,t);row={'variant':variant,'outer':outer,'configId':cid,'config':cfg,'threshold':t,'validation':metric};search.append(row)
     if best is None or metric['macroF1']>best[0]:best=(metric['macroF1'],row)
   choice=best[1];choices.append(choice)
   # Write the selected configuration before evaluating its outer prediction.
   dump(OUT/'selection.json',choices)
   scale,model=fit([source[i]for i in plan['train']],variant,choice['config']);te=[source[i]for i in plan['test']]
   p=predict(te,variant,choice['config'],scale,model)
   for r,prob in zip(te,p):outerpred.append({'variant':variant,'outer':outer,'uid':r['uid'],'group':r['group'],
     'truth':r['target'],'weight':r['weight'],'probability':float(prob),'predicted':int(prob>=choice['threshold'])})
   print('anchor outer',variant,outer,flush=True)
 dump(OUT/'search.json',search);dump(OUT/'outer-predictions.json',outerpred)
 metrics={}
 for variant in FEATURE_SETS:
  rs=[r for r in outerpred if r['variant']==variant];y=np.array([r['truth']for r in rs]);p=np.array([r['predicted']for r in rs]);w=np.array([r['weight']for r in rs])
  metrics[variant]=score(y,p,w,.5);metrics[variant]['averagePrecision']=float(average_precision_score(y,[r['probability']for r in rs],sample_weight=w))
 dump(OUT/'metrics.json',metrics)
 # Freeze deployment configuration from inner results, not outer outcomes.
 full=[r for r in search if r['variant']=='full'];summaries=defaultdict(list)
 for r in full:summaries[(r['configId'],r['threshold'])].append(r['validation']['macroF1'])
 cid,t=max(summaries,key=lambda k:np.mean(summaries[k]));cfg=configs[cid];scale,model=fit(source,'full',cfg)
 dump(OUT/'candidate.json',{'scope':'all named source 4K jack records; outer evaluation fits were separate',
  'config':cfg,'threshold':t,'features':[FEATURES[i]for i in indices('full')],
  'mean':scale.mean_.tolist(),'scale':scale.scale_.tolist(),'W':model.coef_[0].tolist(),'b':float(model.intercept_[0]),
  'crossKey':'normalised descriptors extrapolate morphology, not a validated 6K/7K subtype probability'})
 p=predict(transfer,'full',cfg,scale,model);known={r['group']for r in source};tp=[]
 for r,prob in zip(transfer,p):tp.append({'uid':r['uid'],'group':r['group'],'sourceSongSeen':r['group']in known,
  'annotatedAnchor':'anchor'in r['tags'],'score':float(prob),'predicted':bool(prob>=t)})
 positives=[r for r in tp if r['annotatedAnchor']and not r['sourceSongSeen']]
 dump(OUT/'7k-transfer.json',{'rows':tp,'unseenPositiveRecords':len(positives),'unseenPositiveGroups':len({r['group']for r in positives}),
  'positiveRecall':sum(r['predicted']for r in positives)/len(positives)if positives else None,
  'interpretation':'positive-only diagnostic; untagged 7K charts are not confirmed negatives'})
 print(metrics,flush=True)
if __name__=='__main__':main()
