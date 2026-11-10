"""Reuse the exact accepted interval/mass/expert arrays; change only features."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from competition import read,write

if __name__=='__main__':
    src=Path('.cache/local-feedback-activity-r6');dst=Path('.cache/local-feedback-activity-r6b')
    if(dst/'complete.json').exists():raise ValueError('Preserve completed obligation preparation')
    meta=read(src/'manifest.json');schema=read('.cache/local-feedback-activity-r6b.jsonl.schema.json');blocks=[]
    for record,line in zip(meta['records'],Path('.cache/local-feedback-activity-r6b.jsonl').open(encoding='utf8')):
        row=json.loads(line)
        if row['uid']!=record['uid']or row['frames']!=record['frames']:raise ValueError('Obligation refinement changed intervals/experts')
        blocks.append(np.array(row['values']))
    if len(blocks)!=len(meta['records']):raise ValueError('Obligation inventory incomplete')
    with np.load(src/'arrays.npz')as z:arrays={k:z[k].copy()for k in z.files if k!='x'}
    arrays['x']=np.concatenate(blocks,axis=1);dst.mkdir(exist_ok=True);np.savez_compressed(dst/'arrays.npz',**arrays);meta['featureKeys']=schema['featureKeys'];meta['obligationSchema']=schema;write(dst/'manifest.json',meta);write(dst/'complete.json',dict(records=len(blocks),frames=len(arrays['route']),newScorerCalls=0,marginalsReusedFrom=str(src),parentIntegralMaximumError=meta['parentIntegralMaximumError']));print('prepared obligations',arrays['x'].shape,flush=True)
