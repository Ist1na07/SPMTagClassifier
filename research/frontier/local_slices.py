"""Fill missing one-coordinate controls around one EXISTING inner winner.

Diagnostic only: fixed learned head, inner held-out records, no outer labels,
no coefficient refit, no product selection. All other coordinates stay fixed.
"""
import argparse
import copy
import itertools
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rework'))
import ln_adapter_search as s
from ln_gate import pack_gated
from landscape_audit import multilabel_metrics


def coordinates(policies):
    result = {}
    for k in policies[0]:
        if k == 'familyThresholds':
            for j in [0, 1]:
                result[f'{k}[{j}]'] = sorted({c[k][j] for c in policies})
        else:
            values = {json.dumps(c[k]): c[k] for c in policies}
            if len(values) > 1: result[k] = list(values.values())
    return result


def replace_coordinate(base, key, value):
    c = copy.deepcopy(base)
    if key.startswith('familyThresholds['):
        c['familyThresholds'][int(key[-2])] = value
    else: c[key] = value
    return c


def run(args):
    source, out = Path(args.source), Path(args.output)
    protocol = s.read(source/'protocol.json'); plans = protocol['plans']
    choice = s.read(source/'ln_protected/geometry3'/str(args.outer)/'selection.json')
    plan = plans[args.outer]
    s.init(str(out), args.checkpoints, plans)
    validation = [s._MAIN[i] for i in plan['train']] + [s._MANUAL[i] for i in plan['strongTrain']]
    nmap = len(plan['train']); position = {r['uid']: i for i, r in enumerate(validation)}
    probabilities = [None]*len(validation); fit_sources = []
    for inner, split in enumerate(plan['inner']):
        records = [s._MAIN[i] for i in split['validation']] + [s._MANUAL[i] for i in split['strongValidation']]
        if choice['head']:
            path = Path(args.checkpoints)/'geometry3'/str(args.outer)/str(inner)/(str(choice['head']-1)+'.json.gz')
            saved = s.read(path)  # Missing checkpoint fails; NEVER fit silently.
            expected = [s._MAIN[i]['uid'] for i in split['train']] + [s._MANUAL[i]['uid'] for i in split['strongTrain']]
            if saved['scope'] != expected or saved['config'] != choice['fitConfig'] or saved['variant'] != 'geometry3' or saved['fit']['status'] != 'small_gradient':
                raise ValueError('Saved fit scope/config/status differs')
            local = s.adapter.predict(saved['model'], records); fit_sources.append(str(path))
        else: local = [s._LN[r['uid']] for r in records]
        for r, p in zip(records, local): probabilities[position[r['uid']]] = p
    if any(p is None for p in probabilities): raise ValueError('Missing inner held-out prediction')
    base = choice['policy']; pending = [('baseline', None, base)]
    for key, values in coordinates(protocol['postPolicies']['ln_protected']).items():
        for value in values: pending.append((key, value, replace_coordinate(base, key, value)))
    cache = {}; rows = []; y = s.encode(validation[:nmap]); w = s.np.array([r['recordWeight'] for r in validation[:nmap]])
    for key, value, c in pending:
        route = (c['lnRatio'], c['scorer'])
        if route not in cache: cache[route] = s.cases(validation, probabilities, *route)
        data = pack_gated(cache[route], c, 'ln_protected')
        predicted = s.decisions(s.summaries(data, c), c)[:nmap]
        segment = float(s.np.mean([s.segment_accuracy(r, chunks) for r, chunks in zip(validation[nmap:], data['chunks'][nmap:])]))
        metrics = multilabel_metrics(y, predicted, w, s.MAP_CLASSES, [r['group'] for r in validation[:nmap]])
        joint = .5*(metrics['macro']['f1']+segment)
        rows.append({'coordinate': key, 'value': value, 'config': c, 'joint': joint,
            'segmentAccuracy': segment, 'metrics': metrics, 'mapPredictions': predicted.tolist()})
        if key == 'baseline':
            if abs(joint-choice['inner']['joint']) > 1e-12 or abs(segment-choice['inner']['segmentAccuracy']) > 1e-12:
                raise ValueError('Recomputed baseline differs from actual saved inner selection')
        print(key, value, 'joint', joint, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    result = {'scope': 'fixed-head single-coordinate slices of existing inner-held-out selection, NOT new generalization evidence',
        'outer': args.outer, 'source': str(source), 'selection': choice, 'fitSources': fit_sources,
        'newFits': 0, 'readsOuterPredictionsOrModels': False,
        'mapUids': [r['uid'] for r in validation[:nmap]], 'results': rows}
    s.write(out/'slices.json', result)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--outer', type=int, default=0)
    p.add_argument('--source', default='research/rework/results/ln-extra-gate-r1')
    p.add_argument('--checkpoints', default='.cache/rework-ln-adapter-r1-fits')
    p.add_argument('--output', default='research/frontier/results/local-slices-outer0-r1')
    run(p.parse_args())
