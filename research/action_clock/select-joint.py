"""Nested source-scoped selection for the complete native music/rest grid."""
from report import *
PLANS=json.loads(Path('research/rework/results/mil-stage1-r4/protocol.json').read_text(encoding='utf-8'))['plans']

def scope_score(output,consumer,main_ids,manual_ids):
 rows={r['uid']:r for r in output['consumers'][consumer]};uids=[main[i]for i in main_ids];mus=[manifest['manualUids'][i]for i in manual_ids]
 pred=np.array([[c in rows[u]['tags']for c in classes]for u in uids]);truth=np.array([[c in records[u]['tags']for c in classes]for u in uids]);weight=np.array([records[u]['weight']for u in uids]);whole=metrics(truth,pred,weight)
 details=[dict(uid=u,**local(records[u],rows[u]['segments']))for u in mus];accuracy=float(np.mean([d['accuracy']for d in details]))
 tp=sum(d['breakTruePositiveMs']for d in details);fp=sum(d['falseBreakMs']for d in details);fn=sum(d['trueBreakMs']-d['breakTruePositiveMs']for d in details);p=tp/max(1,tp+fp);r=tp/max(1,tp+fn);f=2*p*r/max(1e-30,p+r)
 groups={records[u]['group']for u in uids};real=[]
 for e in extra:
  if records[e['uid']]['group']not in groups:continue
  a,b=e['interval'];correct=sum(max(0,min(b,s['end'])-max(a,s['start']))*(s['tag']in e['tags'])for s in rows[e['uid']]['segments']);real.append(dict(id=e['id'],accuracy=correct/(b-a)))
 real_accuracy=float(np.mean([e['accuracy']for e in real]))if real else accuracy
 return dict(score=.3*accuracy+.15*f+.15*real_accuracy+.4*whole['macroF1'],manualAccuracy=accuracy,whole=whole,
  breaks=dict(precision=p,recall=r,f1=f,falseMs=fp,missedMs=fn),real=dict(accuracy=real_accuracy,details=real),manual=details)

def summarize(v):
 return dict(score=v['score'],manualAccuracy=v['manualAccuracy'],whole={k:v['whole'][k]for k in ['macroPrecision','macroRecall','macroF1','exactMatch']},breaks=v['breaks'],realAccuracy=v['real']['accuracy'])

def choose(scores,outputs):
 # Knob aliases on flat regions: choose the declared domain centre, not an
 # asymmetric old-model veto. All off-radius parameters are searched too.
 radius_choices=[]
 for radius,mode in [(None,None)]+[(r,m)for r in [.5,1,2]for m in ['nearest','contrast']]:
  if not any(o['config']['radius']==radius and o['config'].get('mode','nearest'if radius else None)==mode for o in outputs):continue
  choices={}
  for name in ['actions','detail','compact']:
   ids=[o['id']for o in outputs if o['config']['radius']==radius and o['config'].get('mode','nearest'if radius else None)==mode]
   winner=max(ids,key=lambda i:(scores[i][name]['score'],-abs(outputs[i]['config']['confidenceFloor']-.5)-abs(outputs[i]['config']['lowNpsScale']-1),-i))
   choices[name]=dict(id=winner,config=outputs[winner]['config'],score=scores[winner][name]['score'])
  radius_choices.append(dict(radius=radius,mode=mode,score=float(np.mean([c['score']for c in choices.values()])),consumers=choices))
 best=max(radius_choices,key=lambda r:(r['score'],-(r['radius']or 0),r['mode']!='contrast'))
 return dict(selected=best,byRadius=radius_choices)

def run():
 with gzip.open(ROOT/'joint-grid-native.json.gz','rt',encoding='utf-8')as f:data=json.load(f)
 outputs=data['outputs'];full=[{n:scope_score(o,n,list(range(len(main))),list(range(6)))for n in ['actions','detail','compact']}for o in outputs]
 selection=choose(full,outputs)
 (ROOT/'joint-full-grid.json').write_text(json.dumps([dict(id=o['id'],config=o['config'],consumers=full[o['id']])for o in outputs],indent=2),encoding='utf-8')
 (ROOT/'joint-full-selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
 outer_choices=[];inner_reports=[]
 for fold,p in enumerate(PLANS):
  scores=[];rows=[]
  for o in outputs:
   consumer_scores={};detail={}
   for n in ['actions','detail','compact']:
    scopes=[scope_score(o,n,s['validation'],s['strongValidation'])for s in p['inner']]
    consumer_scores[n]=dict(score=float(np.mean([s['score']for s in scopes])))
    detail[n]=[summarize(s)for s in scopes]
   scores.append(consumer_scores);rows.append(dict(id=o['id'],config=o['config'],consumers=detail))
  choice=choose(scores,outputs);outer_choices.append(choice);inner_reports.append(dict(fold=fold,grid=rows,choice=choice))
 # Choices frozen before OUTER metrics are evaluated.
 (ROOT/'joint-inner-selections.json').write_text(json.dumps(inner_reports,indent=2),encoding='utf-8')
 pooled={n:{}for n in ['actions','detail','compact']};outer=[]
 for fold,(p,choice)in enumerate(zip(PLANS,outer_choices)):
  detail={}
  for n in pooled:
   o=outputs[choice['selected']['consumers'][n]['id']];rows={r['uid']:r for r in o['consumers'][n]}
   uids={main[i]for i in p['test']}|{manifest['manualUids'][i]for i in p['strongTest']}
   for u in uids:
    if u in pooled[n]:raise ValueError('Duplicate source fold '+u)
    pooled[n][u]=rows[u]
   detail[n]=summarize(scope_score(o,n,p['test'],p['strongTest']))
  outer.append(dict(fold=fold,choice=choice['selected'],metrics=detail))
 dummy=dict(consumers={n:list(rows.values())for n,rows in pooled.items()});oof={n:scope_score(dummy,n,list(range(len(main))),list(range(6)))for n in pooled}
 result=dict(scope='Retrospective source-scoped parameter selection, not untouched-W/b blind evaluation.',selection=selection,
  sourceOOF=oof,outer=outer,checkedPublicControlCases=data['checked'],uniqueNativeDecisions=data['computed'],identicalDecisionReuses=data['hits'],orderingExtension=data.get('orderingExtension'),executionRepairs=data.get('executionRepairs',[]),executionCountMeaning='Counters sum native decision executions and checks across the initial grid and repairs; not independent maps or trials.')
 (ROOT/'joint-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 print('SELECTED',selection['selected'],flush=True)
 for radius in selection['byRadius']:
  print('radius',radius['radius'],'score',radius['score'],flush=True)
  for n,c in radius['consumers'].items():print(n,c['config'],summarize(full[c['id']][n]),flush=True)
 print('SOURCE OOF',{n:summarize(v)for n,v in oof.items()},flush=True)

if __name__=='__main__':run()
