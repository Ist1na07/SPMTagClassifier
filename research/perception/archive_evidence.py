"""Losslessly package completed parameter tables; no fitting or selection."""
from pathlib import Path
import argparse,gzip,json
import numpy as np
ROOT=Path(__file__).with_name('results')

def pack():
    inventory=[]
    previous=ROOT/'archive-inventory.json'
    prior={r['path'].replace('\\','/'):r for r in json.loads(previous.read_text(encoding='utf8'))['files']}if previous.exists()else{}
    for p in ROOT.rglob('*.json'):
        if p.stat().st_size<256000 and p.name not in ['analysis.json','fit-receipts.json','inner-grid.json','outer-predictions.json']:continue
        target=p.with_suffix('.json.gz')
        with gzip.GzipFile(filename=str(target),mode='wb',mtime=0)as f:f.write(p.read_bytes())
        inventory.append(dict(path=target.relative_to(ROOT).as_posix(),originalBytes=p.stat().st_size,archiveBytes=target.stat().st_size))
    for study in ['anchored-r1','anchored-r2','anchored-compact-r2','threshold-r1']:
        for p in (ROOT/study).glob('*/inner-scores.npz'):
            target=p.with_name('inner-scores.compact.npz')
            key=target.relative_to(ROOT).as_posix()
            # Completed, immutable source studies: keep already verified archives.
            if target.exists()and key in prior:
                inventory.append({**prior[key],'path':key});continue
            with np.load(p)as z:a=z['scores'].copy()
            values,index=np.unique(a.reshape(-1,a.shape[-1]),axis=0,return_inverse=True)
            index=index.astype(np.uint16 if len(values)<65536 else np.uint32)
            if not np.array_equal(values[index].reshape(a.shape),a):raise ValueError('Lossless archive failed: '+str(p))
            np.savez_compressed(target,values=values,index=index,shape=a.shape)
            inventory.append(dict(path=key,originalShape=list(a.shape),uniqueScoreVectors=len(values),originalBytes=p.stat().st_size,archiveBytes=target.stat().st_size))
            print(study,p.parent.name,len(values),target.stat().st_size,flush=True)
    (ROOT/'archive-inventory.json').write_text(json.dumps(dict(lossless=True,scope='All large JSON evidence and complete anchored/threshold parameter metric tables; redundant per-representation and optimizer vectors remain local',files=inventory),indent=2)+'\n',encoding='utf8')

def unpack():
    for p in ROOT.rglob('*.json.gz'):
        target=p.with_suffix('');data=gzip.decompress(p.read_bytes())
        if target.exists()and target.read_bytes()!=data:raise ValueError('Preserve changed evidence: '+str(target))
        if not target.exists():target.write_bytes(data)
    for p in ROOT.rglob('inner-scores.compact.npz'):
        target=p.with_name('inner-scores.npz')
        if target.exists():continue
        with np.load(p)as z:np.savez_compressed(target,scores=z['values'][z['index']].reshape(z['shape']))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--unpack',action='store_true');args=parser.parse_args()
    unpack()if args.unpack else pack()
