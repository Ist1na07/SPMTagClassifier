from mil import *
from sklearn.metrics import classification_report
main,manual,data=load();source=json.loads(Path('.cache/rework-aggregate-grid.json').read_text(encoding='utf8'))
byid={r['uid']:r for r in source['rows']};folds=grouped_folds(main,6);predictions=[];search=[];selection=[]
def hard_metrics(rows,p):
    # Preserve actual empty tag outputs; the learned-head argmax fallback must
    # not invent an Anchor prediction for a baseline empty vector.
    y=encode(rows);w=np.array([r['recordWeight']for r in rows])
    return {'macroF1':float(f1_score(y,p,average='macro',sample_weight=w,zero_division=0)),
      'microF1':float(f1_score(y,p,average='micro',sample_weight=w,zero_division=0)),
      'sampleF1':float(f1_score(y,p,average='samples',sample_weight=w,zero_division=0)),
      'exactMatch':float(accuracy_score(y,p,sample_weight=w))}
def matrix_labels(rows,field,cid=None):
    tags=[byid[r['uid']][field] if cid is None else byid[r['uid']][field][cid] for r in rows]
    return np.array([[float(c in ts)for c in MAP_CLASSES]for ts in tags])
for field,variant in [('original','original_fixed'),('normalizedDefault','normalized_fixed'),
                     ('originalGrid','original_aggregation_retuned'),('normalizedGrid','normalized_aggregation_retuned')]:
    for outer,test in enumerate(folds):
        train=[i for i in range(len(main))if i not in set(test)]
        tr=[main[i]for i in train];te=[main[i]for i in test];best=None
        for cid in range(len(source['grid'])) if 'retuned' in variant else [None]:
            prob=matrix_labels(tr,field,cid);metrics=hard_metrics(tr,prob)
            row={'variant':variant,'outer':outer,'configId':cid,'params':source['grid'][cid]if cid is not None else {},'validation':metrics}
            search.append(row)
            if best is None or metrics['macroF1']>best[0]:best=(metrics['macroF1'],row)
        choice=best[1];selection.append(choice);prob=matrix_labels(te,field,choice['configId'])
        for r,p in zip(te,prob):predictions.append({'variant':variant,'outer':outer,'uid':r['uid'],'group':r['group'],
          'truth':target(r),'weight':r['recordWeight'],'predicted':[c for c,v in zip(MAP_CLASSES,p)if v]})
out=Path('research/rework/results/aggregation')
dump(out/'protocol.json',{'scope':'fixed original pretrained segments/head; aggregation-only grouped six-fold parameter selection',
 'grid':source['grid'],'classSupport':{c:len({r['group']for r in main if c in target(r)})for c in MAP_CLASSES},
 'manual':'local predictions unchanged; no manual labels needed for aggregation-only selection',
 'exposure':'retrospective corpus; original model historical training exposure unknown'})
dump(out/'search.json',search);dump(out/'selection.json',selection);dump(out/'predictions.json',predictions)
metrics={}
for variant in dict.fromkeys(r['variant']for r in predictions):
    rs=[r for r in predictions if r['variant']==variant];uid={r['uid']:r for r in main};rows=[uid[r['uid']]for r in rs]
    p=np.array([[float(c in r['predicted'])for c in MAP_CLASSES]for r in rs]);y=encode(rows)
    metrics[variant]=hard_metrics(rows,p)
    metrics[variant]['perClass']=classification_report(y,p,target_names=MAP_CLASSES,
      sample_weight=np.array([r['recordWeight']for r in rows]),output_dict=True,zero_division=0)
dump(out/'metrics.json',metrics)
print({v:{k:m[k]for k in ['macroF1','microF1','exactMatch']}for v,m in metrics.items()})
