"""Actual native music-segmentation grid, using maintained whole/local metrics."""
import sys,json,gzip
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'perception'))
from model import metrics
ROOT=Path('research/action_clock/results')
manifest=json.loads(Path('.cache/local-feedback-activity-r6b/manifest.json').read_text(encoding='utf-8'));records={r['uid']:r for r in manifest['records']};main=manifest['mainUids'];classes=manifest['classes']
y=np.array([[c in records[u]['tags']for c in classes]for u in main]);w=np.array([records[u]['weight']for u in main])
def local(record,segments):
 correct=total=false_break=tp_break=truth_break=pred_break=0.;confusions={}
 for truth in record['annotations']:
  if not truth['tags']:continue
  total+=truth['end']-truth['start'];truth_break+=(truth['end']-truth['start'])*('break'in truth['tags'])
  for s in segments:
   dt=max(0,min(s['end'],truth['end'])-max(s['start'],truth['start']));ok=s['tag']in truth['tags'];correct+=dt*ok
   if s['tag']=='break':pred_break+=dt;tp_break+=dt*('break'in truth['tags']);false_break+=dt*('break'not in truth['tags'])
   key='|'.join(truth['tags'])+' -> '+s['tag'];confusions[key]=confusions.get(key,0)+dt
 return dict(accuracy=correct/total,totalMs=total,correctMs=correct,falseBreakMs=false_break,breakTruePositiveMs=tp_break,trueBreakMs=truth_break,predictedBreakMs=pred_break,confusions=confusions)
extra=[]
for rnd in ['r4','r5','r6']:
 for name,a in json.loads(Path('research/frontier/annotations-'+rnd+'.json').read_text(encoding='utf8'))['cases'].items():extra.append(dict(id=name,uid=a['sourceUid'],interval=a['sourceTargetMs'],tags=[a['localCategory']]))
for name,a in json.loads(Path('research/frontier/annotations-r10.json').read_text(encoding='utf8'))['originalLocalRevision']['cases'].items():extra.append(dict(id=name,uid=a['sourceUid'],interval=a['sourceTargetMs'],tags=[a['primaryCategory']]))
def report(rows):
 rows={r['uid']:r for r in rows};assert set(rows)==set(records);out={}
 for consumer in ['actions','detail','compact']:
  pred=np.array([[c in rows[u]['consumers'][consumer]['tags']for c in classes]for u in main]);whole=metrics(y,pred,w)
  details=[dict(uid=u,**local(records[u],rows[u]['consumers'][consumer]['segments']))for u in manifest['manualUids']]
  real=[]
  for e in extra:
   a,b=e['interval'];correct=sum(max(0,min(b,s['end'])-max(a,s['start']))*(s['tag']in e['tags'])for s in rows[e['uid']]['consumers'][consumer]['segments']);real.append(dict(id=e['id'],accuracy=correct/(b-a)))
  tp=sum(d['breakTruePositiveMs']for d in details);predbreak=sum(d['predictedBreakMs']for d in details);truthbreak=sum(d['trueBreakMs']for d in details)
  out[consumer]=dict(whole=whole,manualAccuracy=float(np.mean([d['accuracy']for d in details])),manual=details,real=dict(accuracy=float(np.mean([d['accuracy']for d in real])),details=real),
   breaks=dict(precision=tp/max(1,predbreak),recall=tp/max(1,truthbreak),falseMs=predbreak-tp,missedMs=truthbreak-tp),
   displaySegments=sum(len(rows[u]['consumers'][consumer]['segments'])for u in main),subsecondSegments=sum(s['end']-s['start']<1000 for u in main for s in rows[u]['consumers'][consumer]['segments']))
 return out
if __name__=='__main__':
 result={}
 for name in ['off','nearest-0.5','nearest-1','nearest-2','contrast-0.5','contrast-1','contrast-2']:
  p=ROOT/f'music-{name}.jsonl.gz'
  if not p.exists():continue
  with gzip.open(p,'rt',encoding='utf8')as f:r=report([json.loads(s)for s in f])
  score=float(np.mean([c['manualAccuracy']+c['whole']['macroF1']for c in r.values()]));result[name]=dict(selectionScore=score,consumers=r)
  print(name,score,[(n,round(v['manualAccuracy']*100,4),round(v['whole']['macroF1']*100,4))for n,v in r.items()],flush=True)
 (ROOT/'music-full-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
