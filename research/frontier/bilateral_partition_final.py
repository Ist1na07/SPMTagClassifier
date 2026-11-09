"""One deployment configuration from INNER tables, never an outer winner."""
from pathlib import Path
import argparse
import numpy as np
from bilateral_partition_model import World, Packed, fit, lift_bias, read
from bilateral_partition_study import ROOT, CONFIGS
from whole_reference import write

OUTPUT=Path('research/overlay-quality/bilateral-final-r1')
KINDS=['bias','hand','bilateral','both']


def declare():
    value=dict(
        selection='Mean six original pooled3INNER score tables. Jointly search variant,all9partitions,kind,108head configs plus frozen,all245post. Eligible active heads have conceptWeight>0. Head0 retained only in bias arm: representation/training parameters are inactive.',
        guard='Mean INNER macroF1 >= best frozen across BOTH signal arms and all9partitions; then maximize mean INNER manual accuracy,F1,recall,prefer frozen,original signals,partition closest4,lower dimension kind,lower partition/head/policy IDs.',
        fitting='After adoption only: chosen configuration fit on all349main+6manual. No held-out coefficients copied. Three standard starts plus exact mapped accepted parent; both seeded from best of matched hand/bilateral, each seeded from bias. TRAIN objective inclusion checked before acceptance.',
        use='Selection saved before any final fit. Outer metrics only decide adoption, not parameter identity. Negative studies need no final fit or deployment.',
        scope='Repeated historical data; mean INNER criteria are development selection, not new generalization evidence.',
        source=Path(__file__).read_text(encoding='utf8'))
    file=OUTPUT/'protocol.json'
    if file.exists()and read(file)!=value:raise ValueError('Finalization protocol changed')
    if not file.exists():write(file,value)


def select():
    declare();read(ROOT/'complete.json');tables=[]
    for variant in ['original','physical']:
        for partition in range(9):
            for kind in KINDS:
                arrays=[]
                for outer in range(6):
                    file=ROOT/variant/str(partition)/kind/str(outer)/'inner-bases.npz'
                    with np.load(file)as z:
                        arrays.append({k:z[k]for k in ['macroF1','macroRecall','manualAccuracy']})
                tables.append(dict(variant=variant,partition=partition,kind=kind,
                    f=np.mean([r['macroF1']for r in arrays],axis=0),
                    r=np.mean([r['macroRecall']for r in arrays],axis=0),
                    a=np.mean([np.mean(r['manualAccuracy'],axis=1)for r in arrays],axis=0)))
    reference=max(float(t['f'][0].max())for t in tables if t['kind']=='bias')
    post=read(ROOT/'protocol.json')['postPolicies']
    winner=None;frozen=None
    for t in tables:
        for hid in range(len(CONFIGS)+1):
            if (hid==0 and t['kind']!='bias')or(hid>0 and CONFIGS[hid-1]['conceptWeight']<=0):continue
            for pid in np.flatnonzero(t['f'][hid]>=reference-1e-12):
                key=(float(t['a'][hid]),float(t['f'][hid,pid]),float(t['r'][hid,pid]),hid==0,t['variant']=='original',
                    -abs(t['partition']-4),-KINDS.index(t['kind']),-t['partition'],-hid,-int(pid))
                row=dict(variant=t['variant'],partition=t['partition'],kind=t['kind'],headId=hid,config=None if hid==0 else CONFIGS[hid-1],
                    policyId=int(pid),policy=post[pid],
                    meanInnerAccuracy=key[0],meanInnerMacroF1=key[1],meanInnerMacroRecall=key[2])
                if winner is None or key>winner[0]:winner=(key,row)
                if hid==0 and(frozen is None or key>frozen[0]):frozen=(key,row)
    value=dict(selection=winner[1],frozenReference=reference,bestEligibleFrozen=frozen[1],usesOuterTruth=False)
    write(OUTPUT/'selection.json',value);print(value,flush=True);return value['selection']


def finalize():
    choice=select()
    decision=read(OUTPUT/'adoption.json')
    if decision.get('adopt')is not True:raise ValueError('No positive adoption decision')
    world=World(choice['variant'],choice['partition']);accepted={}
    def train(kind):
        if kind in accepted:return accepted[kind]
        packed=Packed(world,world.main,world.manual,kind=kind);seed=None;parents=[]
        if kind!='bias':
            for parent_kind in (['hand','bilateral']if kind=='both'else['bias']):
                model,trace=train(parent_kind);source=Packed(world,world.main,world.manual,kind=parent_kind)
                theta=np.asarray(model['theta']);loss,gradient=source.objective(theta,choice['config'])
                if np.max(np.abs(gradient))>1e-5:raise ValueError('Unaccepted full-training parent')
                parents.append((loss,lift_bias(source,packed,theta,choice['config']),parent_kind))
            seed=min(parents,key=lambda r:r[0])[1]
        model,trace=fit(packed,choice['config'],seed)
        if parents and trace['starts'][trace['selectedStart']]['loss']>min(p[0]for p in parents)+1e-8:
            raise ValueError('Full-training nested objective inclusion failed')
        trace['parents']=[dict(kind=p[2],loss=p[0])for p in parents]
        accepted[kind]=(model,trace);write(OUTPUT/(kind+'-full-fit.json'),dict(model=model,fit=trace,
            trainUids=[r['uid']for r in world.main],strongTrainUids=[r['uid']for r in world.manual]))
        return accepted[kind]
    model=None if choice['headId']==0 else train(choice['kind'])[0]
    write(OUTPUT/'profile.json',dict(**choice,model=model,fitScope='all349main+6manual; constructed NOT-CS constraints unchanged',
        selectionSource='mean six original3INNER tables; no outer scores used to choose parameters'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declare-only',action='store_true');parser.add_argument('--select-only',action='store_true');args=parser.parse_args()
    declare()if args.declare_only else select()if args.select_only else finalize()
