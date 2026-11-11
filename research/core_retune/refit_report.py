"""Actual full-data fitted product replay; no frozen-classic-prior shortcut."""
from core import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_feedback'))
from product_readout_report import local
from competition import metrics


def report(file):
    receipt=read(str(file).replace('-product.jsonl','-native.json'));prior='fixed reference classic prior'if receipt.get('preservePrior')else'recomputed classic prior'
    manifest=read('.cache/local-feedback-activity-r6b/manifest.json');records={r['uid']:r for r in manifest['records']}
    rows={r['uid']:r for r in [json.loads(s)for s in Path(file).read_text(encoding='utf8').splitlines()]}
    if set(rows)!=set(records):raise ValueError('Incomplete replay')
    main=manifest['mainUids'];classes=manifest['classes'];y=np.array([[c in records[u]['tags']for c in classes]for u in main]);w=np.array([records[u]['weight']for u in main]);whole={};display={};short={}
    for name in ['actions','detail','compact']:
        pred=np.array([[c in rows[u]['consumers'][name]['tags']for c in classes]for u in main]);whole[name]=metrics(y,pred,w)
        display[name]=sum(len(rows[u]['consumers'][name]['segments'])for u in main);short[name]=sum(s['end']-s['start']<1000 for u in main for s in rows[u]['consumers'][name]['segments'])
    details=[local(records[u],rows[u]['consumers']['actions']['segments'])for u in manifest['manualUids']];real=[]
    for e in Data().extra:
        s,t=e['interval'];frames=rows[e['uid']]['consumers']['actions']['segments'];correct=sum(max(0,min(t,f['end'])-max(s,f['start']))*(f['tag']in e['tags'])for f in frames)
        real.append(dict(id=e['id'],accuracy=correct/(t-s)))
    result=dict(scope='Full-data fitted parameter set, actual public engine plus all three consumers and '+prior+'. Retrospective development metrics, alongside existing source-study evidence.',whole=whole,manual=details,real=dict(accuracy=float(np.mean([r['accuracy']for r in real])),details=real),local=dict(accuracy=float(np.mean([r['accuracy']for r in details])),techToDenseMs=sum(r['techToDenseMs']for r in details),techCsConfusionMs=sum(r['techCsConfusionMs']for r in details)),displaySegments=display,subsecondSegments=short)
    target=Path(file).with_suffix('.report.json');write(target,result)
    print('actual',target,result['local'],result['real']['accuracy'],{n:{k:v[k]for k in ['macroPrecision','macroRecall','macroF1','exactMatch']}for n,v in whole.items()},flush=True)


if __name__=='__main__':report(sys.argv[1])
