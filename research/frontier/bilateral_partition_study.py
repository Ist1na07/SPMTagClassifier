"""Joint partition/model/aggregation study, using existing grouped runner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path
import time
import numpy as np
import rc_action_study as base
from cut_consistency_study import CONFIGS, PLANS, overlay_choice
from bilateral_partition_model import World, Packed, fit as basic_fit, frames, segment_accuracy, lift_bias, read
from whole_reference import write

ROOT = Path('research/overlay-quality/bilateral-partition-r2')
_WORLD_KEY = None
_VARIANT = _PARTITION = _OUTER = None
_PARENT_FILES = {}


def prepare_world(variant, partition):
    global _WORLD_KEY
    if _WORLD_KEY != (variant, partition):
        base._WORLD = World(variant, partition)
        _WORLD_KEY = variant, partition
        _PARENT_FILES.clear()


def fit(p, cfg):
    if p.kind == 'bias':
        return basic_fit(p, cfg)
    candidates=[];proofs=[]
    for parent_kind in (['hand','bilateral']if p.kind=='both'else['bias']):
        source = Packed(p.world, p.main_train, p.manual_train, kind=parent_kind)
        file = ROOT / _VARIANT / str(_PARTITION) / parent_kind / str(_OUTER) / 'inner-models.json'
        if str(file) not in _PARENT_FILES:_PARENT_FILES[str(file)] = read(file)
        wanted = [r['uid'] for r in p.main_train]
        matches = [r for r in _PARENT_FILES[str(file)] if r['headId'] == CONFIGS.index(cfg) + 1 and r['trainUids'] == wanted]
        if matches:model = matches[0]['model'];parent=dict(reused=True,source=str(file),kind=parent_kind)
        else:
            model,trace=fit(source,cfg);parent=dict(reused=False,additionalParentFit=trace,kind=parent_kind)
        theta=np.asarray(model['theta']);parent_loss,gradient=source.objective(theta,cfg)
        if np.max(np.abs(gradient))>1e-5:raise ValueError('Bilateral parent is not an accepted TRAIN state')
        candidates.append((parent_loss,lift_bias(source,p,theta,cfg)));proofs.append(dict(parentLoss=parent_loss,**parent))
    parent_loss,seed=min(candidates,key=lambda r:r[0])
    model, trace = basic_fit(p, cfg, seed)
    if trace['starts'][trace['selectedStart']]['loss'] > parent_loss + 1e-8:
        raise ValueError('Bilateral expansion worse than its embedded TRAIN parent')
    trace['nestedTrainProofs'] = proofs
    return model, trace


def reusable_original_bias(world, outer,kind='bias'):
    # The unchanged 3000ms/.05 partition, bias-only data and all 108 configs
    # already have valid fits. Verify the effective packed inputs, then reuse.
    from cut_consistency_model import World as OldWorld
    original = OldWorld()
    for scope in PLANS[outer]['inner']+[PLANS[outer]]:
        old = Packed(original, [original.main[i] for i in scope['train']],
            [original.manual[i] for i in scope['strongTrain']], kind=kind)
        new = Packed(world, [world.main[i] for i in scope['train']],
            [world.manual[i] for i in scope['strongTrain']], kind=kind)
        for key in ['x', 'log_prior', 'teacher', 'mass', 'class_mass']:
            np.testing.assert_allclose(getattr(old,key), getattr(new,key), atol=1e-13, rtol=1e-13)
        # strong entries contain numpy masks; compare their primitive form.
        def primitive(rows):
            return [(r[0], r[1].tolist(), r[2]) for r in rows]
        if primitive(old.strong) != primitive(new.strong) or old.negative != new.negative:
            raise ValueError('Original partition supervision changed')
        for a, b in zip(old.weak, new.weak):
            for key in ['range', 'target', 'weight']:
                if a[key] != b[key]: raise ValueError('Original weak bag scope changed')
            for key in ['duration','effort']:
                np.testing.assert_allclose(a[key],b[key],atol=1e-12,rtol=1e-12)
    return Path('research/overlay-quality/cut-consistency-r2')


def job(task):
    global _VARIANT, _PARTITION, _OUTER
    variant, partition, kind, outer = task
    _VARIANT, _PARTITION, _OUTER = variant, partition, outer
    prepare_world(variant, partition)
    parent = reusable_original_bias(base._WORLD, outer,kind) if variant=='original'and partition==4 and kind in ['bias','hand'] else None
    output = ROOT / variant / str(partition)
    return base.evaluate((kind,outer,str(output)),CONFIGS,Packed,parent,
        fit_function=fit,frame_function=frames,accuracy_function=segment_accuracy,
        concept_uids=['concept/H01','concept/H02','concept/H03','concept/H04','local/F01','local/F02'],
        checkpoint_dir=ROOT/'checkpoints'/variant/str(partition),extra_selections={'overlay_guarded':overlay_choice})


def declare():
    data = read('.cache/overlay-bilateral-partitions-r1/schema.json')
    names = ['bilateral_partition_model.py','bilateral_partition_study.py','cut_consistency_model.py',
        'cut_consistency_study.py','rc_action_study.py','rc_action_runtime.py','rc_type_model.py','type_bag_objective.py']
    protocol = dict(scope='Bounded physical-partition signals x bilateral Tech/CS odds, retrospective original349+6manual 6x3 song-group folds.',
        data=data,configs=CONFIGS,plans=PLANS,variants=['original','physical'],kinds=['bias','hand','bilateral','both'],
        postPolicies=base.POST,targetInnerFits=2*9*4*108*18,reusedOriginalTargetFits=2*108*18,
        partitionSelection='Within EACH original outer TRAIN, select partition, head and245post policy together from pooled THREE INNER. Never select a partition by outer truth.',
        productSelection='overlay_guarded retains same-inner frozen macroF1 constraint and prioritizes manual local accuracy; existing macro-first controls also retained.',
        model='Unchanged scalar Tech/CS correction and partial NOT-CS likelihood. Hand substitution/fixed-grip x bilateral coactivation 2x2, crossed with physical partition signal OFF/ON. All other class scores and Dense/Fast ratio fixed WITHIN each same partition; repartitioning may change routing and is evaluated across all17classes.',
        budget='One coupled partition experiment within existing7-hour RC allowance; no automatic grid extension.',
        sourceSnapshots={n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in names})
    file=ROOT/'protocol.json'
    if file.exists()and read(file)!=protocol:raise ValueError('Bilateral partition protocol changed')
    if not file.exists():write(file,protocol)


def run(args):
    declare()
    if args.resource_check:
        prepare_world('physical',4);scope=PLANS[0]['inner'][0]
        train=[base._WORLD.main[i]for i in scope['train']];manual=[base._WORLD.manual[i]for i in scope['strongTrain']]
        cfg=dict(localL2=.003,strongWeight=1.,conceptWeight=1.,effortPower=1.,weakBalance=True,primaryPower=4.)
        start=time.monotonic();checks=[]
        for kind in ['bias','hand','bilateral','both']:
            p=Packed(base._WORLD,train,manual,kind=kind);model,trace=basic_fit(p,cfg)
            checks.append(dict(kind=kind,dimension=p.dimension,model=model,fit=trace))
        write(ROOT/'resource-check.json',dict(config=cfg,checks=checks,seconds=time.monotonic()-start))
        print('bilateral resource seconds',time.monotonic()-start,flush=True);return
    read('.cache/overlay-bilateral-partitions-r1/complete.json');read(ROOT/'resource-check.json')
    write(ROOT/'full-run-started.json',dict(started=time.time(),workers=args.workers))
    with ProcessPoolExecutor(max_workers=args.workers)as pool:
        for phase in [['bias'],['hand','bilateral'],['both']]:
            tasks=[(v,p,kind,o)for v,p,kind,o in itertools.product(['original','physical'],range(9),phase,range(6))]
            for result in pool.map(job,tasks):print('bilateral partition',result,flush=True)
    write(ROOT/'complete.json',dict(targetInnerFits=2*9*4*108*18,partitionOuterScopes=432,
        reusedOriginalTargetFits=2*108*18,requiresJointInnerPartitionConsumer=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resource-check',action='store_true');parser.add_argument('--workers',type=int,default=6)
    run(parser.parse_args())
