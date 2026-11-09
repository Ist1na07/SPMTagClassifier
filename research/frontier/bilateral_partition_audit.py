"""Consume optimizer receipts and disambiguate frozen partition profiles."""
import numpy as np
from bilateral_partition_study import ROOT
from pressure_data import mil, read
from rc_action_report import unpack
from rc_action_runtime import policy_predictions
from whole_reference import write


def run():
    read(ROOT/'analysis.json');main,_,_=mil.load();weights=np.array([r['recordWeight']for r in main])
    fits=starts=reused=0;maximum_pg=spread=0.;inclusions=0;worst_margin=-np.inf
    for file in ROOT.glob('*/*/*/*/fits.json'):
        for row in read(file):
            fits+=1;starts+=len(row['starts']);reused+=int('reusedFrom'in row)
            maximum_pg=max(maximum_pg,row['maximumProjectedGradient'])
            losses=[r['loss']for r in row['starts']];spread=max(spread,max(losses)-min(losses))
            selected=next(r['loss']for r in row['starts']if r['start']==row['selectedStart'])
            parents=row.get('nestedTrainProofs',[])
            if isinstance(parents,dict):parents=[parents]
            for parent in parents:
                if 'parentLoss'not in parent:continue
                inclusions+=1;worst_margin=max(worst_margin,selected-parent['parentLoss'])
    if fits!=read(ROOT/'complete.json')['targetInnerFits']or maximum_pg>1e-5 or worst_margin>1e-8:
        raise ValueError('Incomplete or unaccepted stored inner fits')
    # The original consumer's partition curves are conditional on an ACTIVE
    # correction head. Frozen has inactive training weights, but partition
    # remains active: give its own complete nine-value, post-retuned curve.
    frozen=[]
    for variant in ['original','physical']:
        for outer in range(6):
            points=[];vectors=[];ids0=None
            for partition in range(9):
                folder=ROOT/variant/str(partition)/'bias'/str(outer)
                selected=read(folder/'search.json')[0]
                with np.load(folder/'inner-bases.npz')as saved:z={k:saved[k]for k in saved.files}
                ids=z['validationIds']
                if ids0 is None:ids0=ids
                if not np.array_equal(ids0,ids):raise ValueError('Frozen partition validation scope changed')
                vectors.append(policy_predictions(unpack(z,0),[selected['policy']])[0])
                points.append(dict(partition=partition,**selected))
            ref=max(range(9),key=lambda i:(points[i]['macroF1'],points[i]['macroRecall'],points[i]['segmentAccuracy'],-abs(i-4),-i))
            w=weights[ids0]
            for i,point in enumerate(points):point['changedTagSetMass']=float(w@np.any(vectors[i]!=vectors[ref],axis=1)/w.sum())
            frozen.append(dict(variant=variant,outer=outer,referencePartition=ref,points=points))
    value=dict(optimizer=dict(targetInnerFitReceipts=fits,trainingStarts=starts,reusedTargetFitReceipts=reused,
        maximumProjectedGradient=maximum_pg,maximumStartLossSpread=spread,nestedParentComparisons=inclusions,
        worstChildMinusParentTrainObjective=None if inclusions==0 else worst_margin),
        profileScopeCorrection='analysis.json training-coordinate and partition curves condition on active correction heads. Frozen training coordinates are inactive; the full post-retuned frozen partition curves are supplied here. No fits or selections changed.',
        frozenPartitionProfiles=frozen)
    write(ROOT/'audit-and-profile-scope.json',value);print(value['optimizer'],flush=True)


if __name__=='__main__':run()
