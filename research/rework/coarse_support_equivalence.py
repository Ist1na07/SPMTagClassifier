"""Verify the no-support arm reproduces the complete raw study, every config."""
import argparse
from coarse_search import read,write
from coarse_model import Path,np


def run(directory,parent):
    root=Path(directory);source=Path(parent);complete=read(root/'complete.json');variants=complete['variants']
    original=read(source/'outer-predictions.json');old_segments=read(source/'outer-segments.json');folds=configs=maps=segments=0
    maximum={};rounded=0
    def near(key,a,b):
        nonlocal rounded
        if a!=b:rounded+=1
        error=float(np.abs(np.array(a)-np.array(b)).max());maximum[key]=max(maximum.get(key,0.),error)
        assert error<=1e-12,(key,error)
    for variant in variants:
        for o in range(6):
            selected=read(root/variant/'none'/str(o)/'selection.json');old=read(source/variant/str(o)/'selection.json')
            assert selected['configId']==old['configId']and selected['config']==old['config'],(variant,o,'selection')
            assert read(root/variant/'none'/str(o)/'outer-model.json')['model']==read(source/variant/str(o)/'outer-model.json')['model']
            for i in range(3):
                a=source/variant/str(o)/str(i)/'inner-predictions.npz';b=root/variant/'none'/str(o)/str(i)/'inner-predictions.npz'
                with np.load(a)as x,np.load(b)as y:
                    for key in ['map','segmentAccuracy']:assert np.array_equal(x[key],y[key]),(variant,o,i,key)
                    configs+=len(x['map'])
                folds+=1
        old={r['uid']:r for r in original if r['variant']==variant};new=read(root/variant/'outer-predictions.json');new={r['uid']:r for r in new if r['variant']=='none'}
        assert old.keys()==new.keys()
        for uid,a in old.items():
            b=new[uid];assert a['predicted']==b['predicted'],(variant,uid);near('composition',a['composition'],b['composition']);maps+=1
        old=sorted([r for r in old_segments if r['variant']==variant],key=lambda r:(r['uid'],r['start']))
        new=sorted([r for r in read(root/variant/'outer-segments.json')if r['variant']=='none'],key=lambda r:(r['uid'],r['start']))
        assert len(old)==len(new)
        for a,b in zip(old,new,strict=True):
            assert all(a[k]==b[k]for k in ['uid','start','end','tag']),(variant,a['uid'],a['start'])
            near('probability',a['probability'],b['probability']);near('aggregationWeight',a['aggregationWeight'],b['aggregationWeight']);segments+=1
    result={'scope':'none arm vs original complete raw study, every effective inner configuration and selected outer output',
      'expertVariants':len(variants),'innerFoldJobs':folds,'innerConfigurations':configs,'outerMapOutputs':maps,'outerRawIntervals':segments,
      'equivalence':'exact inner hard predictions/accuracies, selected configs/models and outer labels/intervals; outer numeric scores/weights use1e-12 tolerance',
      'maximumAbsoluteErrors':maximum,'nonBitwiseNumericFields':rounded,'numericToleranceReason':'actual batching-roundoff.json reproduction proves separate vs combined test GEMM batching; initial bitwise-only outer check failed and is retained in the strict-check log','newFits':0}
    write(root/'raw-control-equivalence.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='research/rework/results/coarse-supported-all-r1')
    p.add_argument('--parent',default='research/rework/results/coarse-shared-r3');a=p.parse_args();run(a.directory,a.parent)
