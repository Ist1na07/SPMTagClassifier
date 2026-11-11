"""Actual candidate product report, preserving the old product receipt."""
from core import *


def run():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_feedback'))
    from product_readout_report import local
    from competition import metrics
    manifest=read('.cache/local-feedback-activity-r6b/manifest.json');records={r['uid']:r for r in manifest['records']};rows={r['uid']:r for r in [json.loads(s)for s in Path('.cache/core-retune-product-candidate.jsonl').read_text(encoding='utf8').splitlines()]};main=manifest['mainUids'];classes=manifest['classes'];y=np.array([[c in records[u]['tags']for c in classes]for u in main]);w=np.array([records[u]['weight']for u in main]);whole={};display={};short={}
    for name in ['actions','detail','compact']:
        pred=np.array([[c in rows[u]['consumers'][name]['tags']for c in classes]for u in main]);whole[name]=metrics(y,pred,w);display[name]=sum(len(rows[u]['consumers'][name]['segments'])for u in main);short[name]=sum(s['end']-s['start']<1000 for u in main for s in rows[u]['consumers'][name]['segments'])
    details=[local(records[u],rows[u]['consumers']['actions']['segments'])for u in manifest['manualUids']];real=[]
    for e in Data().extra:
        s,t=e['interval'];frames=rows[e['uid']]['consumers']['actions']['segments'];correct=sum(max(0,min(t,f['end'])-max(s,f['start']))*(f['tag']in e['tags'])for f in frames);real.append(dict(id=e['id'],accuracy=correct/(t-s)))
    result=dict(scope='Actual candidate public static engine, original full349 labels and six charts, existing same-rate original released curves. Retrospective fixed logical repair replay, not fresh blind validation.',whole=whole,manual=details,real=dict(accuracy=float(np.mean([r['accuracy']for r in real])),details=real),local=dict(accuracy=float(np.mean([r['accuracy']for r in details])),techToDenseMs=sum(r['techToDenseMs']for r in details),techCsConfusionMs=sum(r['techCsConfusionMs']for r in details)),displaySegments=display,subsecondSegments=short);write('research/core_retune/results/product-report.json',result);print(result['local']);print({n:{k:round(v[k]*100,4)for k in ['macroPrecision','macroRecall','macroF1']}for n,v in whole.items()})


if __name__=='__main__':run()
