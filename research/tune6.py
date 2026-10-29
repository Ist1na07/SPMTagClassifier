import json,itertools,re
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score,classification_report,accuracy_score
classes=['jack','ln','speed','stamina','tech']
records=[json.loads(l) for l in open('research/results/6k-projections.jsonl',encoding='utf-8')]
# The two source studies have separate grouped splits. Joint transfer needs an
# additional cross-domain screen: no held-out song may be known to either head.
identities={};known={4:set(),7:set()}
def identity(file):
 text=Path(file).read_text(encoding='utf-8-sig',errors='replace').split('[HitObjects]')[0]
 m={}
 for line in text.splitlines():
  if ':' in line:
   k,v=line.split(':',1)
   if k in ['Artist','Title','BeatmapID','BeatmapSetID']:m[k]=v.strip()
 norm=lambda s:re.sub(r'[^\w]+','',re.sub(r'\bx\s*\d+(?:\.\d+)?\b','',s.lower()))
 out={('song',norm(m.get('Artist','')),norm(m.get('Title','')))}
 for k in ['BeatmapID','BeatmapSetID']:
  if m.get(k,'').isdigit() and int(m[k])>0:out.add((k,m[k]))
 return out
for key,folder in [(4,'spm_dataset-4k-0.2.0'),(7,'spm_dataset-1.1.1')]:
 split=json.loads(Path(f'research/results/{key}k-split.json').read_text(encoding='utf-8'))
 labels=json.loads(Path('dataset',folder,'labels.json').read_text(encoding='utf-8-sig'))
 for i,r in enumerate(labels):
  uid=f'{folder}/{r["id"]}/{i}'
  if uid not in split:continue
  ids=identity(Path('dataset',folder,r['file']));identities[uid]=ids
  if split[uid]['split']=='train':known[key].update(ids)
for file in Path('dataset/TagClassifierDataset').glob('*/*.osu'):known[7].update(identity(file))
screened=[];exclusions=[]
for r in records:
 other=7 if r['sourceKeys']==4 else 4
 if r['split'] in ['validation','test'] and identities[r['uid']]&known[other]:
  exclusions.append({'uid':r['uid'],'reason':'song/ID seen by the other source-domain training head'})
 else:screened.append(r)
validation_ids=set().union(*(identities[r['uid']] for r in screened if r['split']=='validation'))
records=[]
for r in screened:
 if r['split']=='test' and identities[r['uid']]&validation_ids:
  exclusions.append({'uid':r['uid'],'reason':'cross-domain validation/test song overlap'})
 else:records.append(r)
Path('research/results/6k-crossdomain-exclusions.json').write_text(json.dumps(exclusions,ensure_ascii=False,indent=2),encoding='utf-8')
grid={'lambda':[0,1/3,2/3,1],'threshold':[.25,.4,.55]}
y=np.array([[int(c in r['truth']) for c in classes] for r in records])
val=np.array([i for i,r in enumerate(records) if r['split']=='validation']);test=np.array([i for i,r in enumerate(records) if r['split']=='test'])
def threshold(p,t):
 b=p>=t
 for i in np.flatnonzero(~b.any(axis=1)):b[i,np.argmax(p[i])]=True
 return b
def predict(adapt,lam):
 data=[r if adapt else r['noKeyAdaptation'] for r in records]
 return np.array([[lam*r['p7'][c]+(1-lam)*r['p4'][c] for c in classes] for r in data])
variants={'tuned_transfer':(True,grid['lambda']),'pure4':(True,[0]),'pure7':(True,[1]),'barycentric':(True,[2/3]),'no_key_adaptation':(False,grid['lambda'])}
search=[];selected={}
for name,(adapt,lambdas) in variants.items():
 best=None
 for lam,t in itertools.product(lambdas,grid['threshold']):
  p=predict(adapt,lam);f=float(f1_score(y[val],threshold(p[val],t),average='macro',zero_division=0))
  row={'variant':name,'adapt':adapt,'lambda':lam,'threshold':t,'validationMacroF1':f};search.append(row)
  if best is None or f>best[0]:best=(f,row,p)
 selected[name]=best
Path('research/results/6k-selection.json').write_text(json.dumps({k:v[1] for k,v in selected.items()},indent=2),encoding='utf-8')
results={}
for name,(_,config,p) in selected.items():
 pred=threshold(p[test],config['threshold']);results[name]={**config,
  'testMacroF1':float(f1_score(y[test],pred,average='macro',zero_division=0)),
  'testMicroF1':float(f1_score(y[test],pred,average='micro',zero_division=0)),
  'testExactMatch':float(accuracy_score(y[test],pred)),
  'perClass':classification_report(y[test],pred,target_names=classes,output_dict=True,zero_division=0)}
config=selected['tuned_transfer'][1]
Path('src/models/6k.json').write_text(json.dumps({'classes':classes,'lambda':config['lambda'],'threshold':config['threshold'],
 'segmentLambda':2/3,'evidenceLevel':'source-label-preserving projection surrogate, not human 6K labels',
 'projection':'4→6 symmetric vacant-column embedding; 7→6 remove center column; source song groups retain original split'},indent=2),encoding='utf-8')
Path('research/results/6k-surrogate-metrics.json').write_text(json.dumps({'grid':grid,'validation':len(val),'test':len(test),
 'results':results,'search':search},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:{'val':v['validationMacroF1'],'test':v['testMacroF1']} for k,v in results.items()},indent=2))
