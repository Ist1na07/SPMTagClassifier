"""Report actual product output, not a cached learned-head surrogate."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import metrics,write
from reanalyze_allowed import corrected


def local(record,frames):
    total=correct=0.;transitions={}
    for truth in record['annotations']:
        tags=truth['tags']
        if not tags:continue
        total+=truth['end']-truth['start']
        for f in frames:
            dt=max(0.,min(truth['end'],f['end'])-max(truth['start'],f['start']))
            if not dt:continue
            correct+=dt*(f['tag']in tags);key='|'.join(tags)+' -> '+f['tag'];transitions[key]=transitions.get(key,0.)+dt
    return corrected([dict(uid=record['uid'],accuracy=correct/max(total,1),correctMs=correct,totalMs=total,transitions=transitions)])[0]


def run():
    manifest=json.loads(Path('.cache/local-feedback-activity-r6b/manifest.json').read_text(encoding='utf8'));rows=[json.loads(s)for s in Path('.cache/local-feedback-product-readout.jsonl').read_text(encoding='utf8').splitlines()];lookup={r['uid']:r for r in rows};records={r['uid']:r for r in manifest['records']};classes=manifest['classes'];main=manifest['mainUids'];whole={}
    y=np.array([[c in records[u]['tags']for c in classes]for u in main],float);weight=np.array([records[u]['weight']for u in main])
    for name in ['detail','actions','compact']:
        pred=np.array([[c in lookup[u]['consumers'][name]['tags']for c in classes]for u in main]);whole[name]=metrics(y,pred,weight)
    manual=[local(records[u],lookup[u]['consumers']['actions']['segments'])for u in manifest['manualUids']]
    display={n:sum(len(r['consumers'][n]['segments'])for r in rows if r['uid']in main)for n in ['actions','detail','compact']};short={n:sum(s['end']-s['start']<1000 for r in rows if r['uid']in main for s in r['consumers'][n]['segments'])for n in display}
    receipt=dict(scope='Actual static product full349 replay; original6manual with allowed-tag duration scoring. This is not a new held-out accuracy study. No fitted joint R7 weights adopted.',whole=whole,manual=manual,local=dict(accuracy=float(np.mean([r['accuracy']for r in manual])),techToDenseMs=sum(r['techToDenseMs']for r in manual),techCsConfusionMs=sum(r['techCsConfusionMs']for r in manual)),displaySegments=display,subsecondSegments=short)
    write('research/local_feedback/results/product-readout-report.json',receipt);print(receipt['local']);print({n:round(m['macroF1']*100,4)for n,m in whole.items()});print(display,short)


if __name__=='__main__':run()
