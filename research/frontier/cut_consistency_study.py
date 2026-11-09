"""Bounded Tech/CS correction: complete declared grid, existing grouped runner."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import itertools
from pathlib import Path
import time
import numpy as np
import rc_action_study as base
from cut_consistency_model import World, Packed, fit as basic_fit, frames, segment_accuracy, KINDS, mil, read
from whole_reference import write

ROOT = Path('research/overlay-quality/cut-consistency-r2')
CONFIGS = [dict(localL2=l, strongWeight=s, conceptWeight=c, effortPower=a, weakBalance=True, primaryPower=p)
    for l, s, c, a, p in itertools.product([.0003, .003, .03], [0., 1., 4.], [0., 1., 4.], [0., 1.], [1., 4.])]
PLANS = read('research/rework/results/mil-stage1-r4/protocol.json')['plans']
PARENTS = {'bias': [], 'hand': ['bias'], 'rhythm': ['bias'], 'both': ['hand', 'rhythm']}
_OUTER = None
_CACHE = {}
_FILES = {}


def init():
    base._WORLD = World()


def names(p):
    return ['intercept'] + [k for k, keep in zip(p.scaler['expandedKeys'], p.scaler['keep']) if keep]


def lift(source, target, theta, cfg):
    result = np.zeros(target.dimension)
    for key, value in zip(names(source), theta):
        result[names(target).index(key)] = value
    np.testing.assert_allclose(source.x @ theta, target.x @ result, atol=1e-12)
    if abs(source.objective(theta, cfg)[0] - target.objective(result, cfg)[0]) > 1e-10:
        raise ValueError('Nested feature initializer changed the TRAIN objective')
    return result


def parent_fit(p, cfg):
    key = (p.kind, tuple(r['uid'] for r in p.main_train), tuple(r['uid'] for r in p.manual_train), tuple(cfg.items()))
    if key in _CACHE:
        model, trace = _CACHE[key]
        return model, dict(trace, reusedParent=True)
    if _OUTER is not None:
        folder = ROOT / p.kind / str(_OUTER)
        file = folder / 'inner-models.json'
        if file.exists():
            if str(file) not in _FILES:
                _FILES[str(file)] = read(file)
            head = CONFIGS.index(cfg) + 1
            wanted = [r['uid'] for r in p.main_train]
            matches = [r for r in _FILES[str(file)] if r['headId'] == head and r['trainUids'] == wanted]
            if matches:
                model = matches[0]['model']; loss, gradient = p.objective(np.asarray(model['theta']), cfg)
                if np.max(np.abs(gradient)) > 1e-5:
                    raise ValueError('Cached nested parent no longer satisfies its TRAIN stationarity check')
                trace = dict(reusedParent=True, source=str(file), sourceHeadId=head, loss=loss)
                _CACHE[key] = model, trace
                return model, trace
    model, trace = fit(p, cfg)
    _CACHE[key] = model, trace
    return model, trace


def fit(p, cfg, seed_theta=None):
    candidates, proofs = [], []
    for kind in PARENTS[p.kind]:
        source = Packed(p.world, p.main_train, p.manual_train, kind=kind)
        model, trace = parent_fit(source, cfg)
        mapped = lift(source, p, np.asarray(model['theta']), cfg)
        value = p.objective(mapped, cfg)[0]
        candidates.append((value, mapped))
        proofs.append(dict(kind=kind, loss=value, reused=bool(trace.get('reusedParent')), parentTrace=trace))
    if seed_theta is not None:
        candidates.append((p.objective(seed_theta, cfg)[0], seed_theta))
    seed = min(candidates, key=lambda r: r[0])[1] if candidates else None
    model, trace = basic_fit(p, cfg, seed)
    selected_loss = trace['starts'][trace['selectedStart']]['loss']
    if candidates and selected_loss > min(r[0] for r in candidates) + 1e-8:
        raise ValueError('Expanded model is worse on TRAIN than its nested parent; repair before validation')
    trace['nestedTrainProofs'] = proofs
    return model, trace


def overlay_choice(heads):
    # Preserve the same scope's optimally retuned frozen macro-F1, then choose
    # actual annotated local accuracy. This is a development selection rule,
    # not a claim that outer metrics are guaranteed not to regress.
    reference = heads[0]['macroF1']
    eligible = [h for h in heads if h['macroF1'] >= reference - 1e-12 and
        (h['headId'] == 0 or h['config']['conceptWeight'] > 0)]
    return max(eligible, key=lambda h: (h['segmentAccuracy'], h['macroF1'], h['macroRecall'],
        h['headId'] == 0, -h['headId'], -h['policyId']))


def job(task):
    global _OUTER
    kind, outer, root = task
    _OUTER = outer
    return base.evaluate(task, CONFIGS, Packed, fit_function=fit, frame_function=frames,
        accuracy_function=segment_accuracy, concept_uids=['concept/H01', 'concept/H02', 'concept/H03', 'concept/H04',
            'concept/C16', 'concept/C24', 'concept/C32', 'concept/C48', 'local/F01', 'local/F02'],
        checkpoint_dir=ROOT / 'checkpoints', extra_selections={'overlay_guarded': overlay_choice})


def declare():
    world = World()
    files = ['cut_consistency_model.py', 'cut_consistency_study.py', 'rc_action_study.py',
        'rc_action_runtime.py', 'rc_type_model.py', 'type_bag_objective.py']
    protocol = dict(scope='Bounded Tech versus CS correction, original 349 whole and 6 manual song-grouped folds; retrospective development evidence.',
        configs=CONFIGS, kinds=KINDS, postPolicies=base.POST, plans=PLANS,
        innerFits=4 * len(CONFIGS) * 18, classBalance='fixed equal-class weak primary likelihood',
        model='One scalar Tech/combined-CS log-odds correction; preserve Dense/Fast ratio, total triad mass, all other raw class scores, LN/Break route and segmentation.',
        supervision='Whole singleton CS/Tech labels supervise bags, actual manual intervals supervise partial labels. H01/H03 assert only NOT-CS including unchanged alternatives; no invented Tech label. No real-query answers newly trained.',
        primarySelection='overlay_guarded: best manual interval accuracy subject to same-inner macroF1 >= retuned frozen head; then macroF1/recall/frozen/tie. Also retain existing macro-first, frozen, no-concept and no-strong controls.',
        postScope='Complete Cartesian 3 existing policy coordinates (245 choices); all other legacy policy values fixed, not claimed globally retuned.',
        sources=[dict(uid=r['uid'], group=r['group'], tags=mil.target(r), weight=r['recordWeight']) for r in world.main],
        nestedStarts='Three deterministic starts plus the lower TRAIN-loss exact embedded accepted parent; bias -> hand/rhythm -> both. No validation basin selection.',
        limitations='Prior historical training exposure unknown; held-out corpus has repeatedly been studied; constructed rule checks are not random accuracy. C24 Speed error and full-map F02 Break gate are outside this restricted head.',
        sourceSnapshots={n:Path(__file__).with_name(n).read_text(encoding='utf8') for n in files},
        cutInputSchema=read('.cache/overlay-cut-inputs-r1.jsonl.schema.json'))
    path = ROOT / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Preserve declared cut study; source or protocol changed')
    if not path.exists():
        write(path, protocol)


def run(args):
    declare()
    if args.resource_check:
        init(); scope = PLANS[0]['inner'][0]
        train = [base._WORLD.main[i] for i in scope['train']]
        manual = [base._WORLD.manual[i] for i in scope['strongTrain']]
        cfg = dict(localL2=.003, strongWeight=1., conceptWeight=1., effortPower=1., weakBalance=True, primaryPower=4.)
        start = time.monotonic(); checks = []
        for kind in KINDS:
            p = Packed(base._WORLD, train, manual, kind=kind); model, trace = fit(p, cfg)
            checks.append(dict(kind=kind, model=model, fit=trace, dimension=p.dimension))
        write(ROOT / 'resource-check.json', dict(config=cfg, checks=checks, seconds=time.monotonic()-start,
            scope='First original INNER TRAIN, fixed declared resource config; no selection or outer scoring'))
        print('cut resource seconds', time.monotonic()-start, flush=True)
        return
    read(ROOT / 'resource-check.json')
    write(ROOT / 'full-run-started.json', dict(started=time.time(), declaredInnerFits=4 * len(CONFIGS) * 18))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init) as pool:
        for phase in [['bias'], ['hand', 'rhythm'], ['both']]:
            tasks = [(kind, outer, str(ROOT)) for kind, outer in itertools.product(phase, range(6))]
            for result in pool.map(job, tasks):
                print('cut consistency', result, flush=True)
    for file in ['outer-predictions', 'outer-local', 'concept-predictions']:
        write(ROOT / (file + '.json'), [r for kind, outer in itertools.product(KINDS, range(6))
            for r in read(ROOT / kind / str(outer) / (file + '.json'))])
    counts = [read(ROOT / k / str(o) / 'complete.json') for k, o in itertools.product(KINDS, range(6))]
    write(ROOT / 'complete.json', dict(innerFits=sum(r['innerFits'] for r in counts),
        innerTrainingStarts=sum(r['innerTrainingStarts'] for r in counts),
        note='Nested parent fits absent from saved parent models are separately included in nestedTrainProofs; above is target-head accounting.',
        selectedOuterHeadFits=sum(r['selectedOuterHeadFits'] for r in counts)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resource-check', action='store_true')
    parser.add_argument('--workers', type=int, default=6)
    run(parser.parse_args())
