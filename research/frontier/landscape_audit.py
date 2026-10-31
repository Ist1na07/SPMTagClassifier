"""Read-only diagnosis of saved experiments, not a new model selection study.

Single-coordinate effects are reported ONLY for actually matched contexts.
Equal scalar scores are distinguished from equal whole-chart predictions.
No surrogate is used to fill unobserved cells of the sampled policy design.
"""
import argparse
from collections import defaultdict
import gzip
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.metrics import precision_recall_fscore_support, jaccard_score


def read(path):
    path = Path(path)
    with (gzip.open(path, 'rt', encoding='utf8') if path.suffix == '.gz'
          else path.open(encoding='utf8')) as f:
        return json.load(f)


def matched_contexts(configurations, parameter):
    """Return comparisons with every OTHER declared coordinate held fixed."""
    groups = defaultdict(list)
    for i, c in enumerate(configurations):
        key = json.dumps({k: v for k, v in c.items() if k != parameter}, sort_keys=True)
        groups[key].append(i)
    return [ids for ids in groups.values()
            if len({json.dumps(configurations[i][parameter]) for i in ids}) > 1]


def multilabel_metrics(y, p, weights, classes, groups):
    y, p, weights = np.asarray(y), np.asarray(p), np.asarray(weights)
    result = {}
    for average in ['macro', 'micro', 'samples']:
        pr, re, f1, _ = precision_recall_fscore_support(
            y, p, sample_weight=weights, average=average, zero_division=0)
        result[average] = {'precision': float(pr), 'recall': float(re), 'f1': float(f1)}
    result.update(exactMatch=float(np.average(np.all(y == p, axis=1), weights=weights)),
                  jaccard=float(jaccard_score(y, p, average='samples', sample_weight=weights, zero_division=0)),
                  hammingLoss=float(np.average(np.mean(y != p, axis=1), weights=weights)),
                  meanTrueTags=float(np.average(y.sum(1), weights=weights)),
                  meanPredictedTags=float(np.average(p.sum(1), weights=weights)))
    pr, re, f1, _ = precision_recall_fscore_support(y, p, sample_weight=weights, average=None, zero_division=0)
    result['perClass'] = {c: {'precision': float(pr[j]), 'recall': float(re[j]), 'f1': float(f1[j]),
        'positivePhysicalWeight': float((weights * y[:, j]).sum()),
        'positiveSongGroups': len({g for g, v in zip(groups, y[:, j]) if v}),
        'predictedPhysicalWeight': float((weights * p[:, j]).sum())} for j, c in enumerate(classes)}
    return result


def comparison_effects(p, scores, pairs, coordinate):
    counts = dict(comparisons=0, changedWholeTagVectors=0, changedMapMacroF1=0,
                  changedJoint=0, sameJointButChangedWholeTagVectors=0)
    largest = 0.
    for i, j in pairs:
        # p is [head, policy, chart, class]. Vary only the requested axis.
        a, b = (p[i], p[j]) if coordinate == 'head' else (p[:, i], p[:, j])
        sa, sb = (scores[i], scores[j]) if coordinate == 'head' else (scores[:, i], scores[:, j])
        changed = np.any(a != b, axis=(-2, -1))
        joint = np.abs(sa[..., 0] - sb[..., 0])
        macro = np.abs(sa[..., 1] - sb[..., 1])
        same = joint <= 1e-12
        counts['comparisons'] += changed.size
        counts['changedWholeTagVectors'] += int(changed.sum())
        counts['changedMapMacroF1'] += int((macro > 1e-12).sum())
        counts['changedJoint'] += int((~same).sum())
        counts['sameJointButChangedWholeTagVectors'] += int((same & changed).sum())
        largest = max(largest, float(joint.max()))
    return {**counts, 'maxAbsoluteJointDifference': largest if counts['comparisons'] else None}


def paired_group_intervals(y, predictions, weights, groups, draws=2000):
    names = list(predictions)
    unique, gi = np.unique(groups, return_inverse=True)
    multiplicity = np.random.default_rng(20261101).multinomial(
        len(unique), np.full(len(unique), 1/len(unique)), size=draws)
    def ratio(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b > 0)
    values = {}
    for name, p in predictions.items():
        blocks = []
        for x in [y*p, (1-y)*p, y*(1-p)]:
            b = np.zeros((len(unique), y.shape[1]))
            np.add.at(b, gi, x*weights[:, None])
            blocks.append(multiplicity @ b)
        tp, fp, fn = blocks
        t, f, n = (v.sum(1) for v in blocks)
        b = np.zeros((len(unique), 2))
        np.add.at(b, gi, np.stack([np.all(y == p, axis=1)*weights, weights], axis=1))
        exact = multiplicity @ b
        values[name] = dict(macroPrecision=ratio(tp, tp+fp).mean(1),
            macroRecall=ratio(tp, tp+fn).mean(1), macroF1=ratio(2*tp, 2*tp+fp+fn).mean(1),
            microPrecision=ratio(t, t+f), microRecall=ratio(t, t+n), microF1=ratio(2*t, 2*t+f+n),
            exactMatch=ratio(exact[:, 0], exact[:, 1]))
    return {'method': 'paired song-group bootstrap; physical-file weights; fixed 17-class convention, zero_division=0',
        'draws': draws, 'baseline': names[0], 'candidate': names[1],
        'delta95': {k: np.quantile(values[names[1]][k]-values[names[0]][k], [.025, .975]).tolist()
                    for k in values[names[0]]}}


def map_comparison(root, protocol):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rework'))
    from mil import load, encode, MAP_CLASSES
    weak, _, _ = load()
    baseline = [r for r in read('research/rework/results/aggregation/predictions.json')
                if r['variant'] == 'original_fixed']
    candidate = [r for r in read(root / 'ln_protected/outer-predictions.json')
                 if r['variant'] == 'geometry3']
    y = encode(weak)
    w = np.array([r['recordWeight'] for r in weak])
    groups = [r['group'] for r in weak]
    results, predictions = {}, {}
    for name, rows in [('original_fixed', baseline), ('v1.2.1_LN_candidate_outer', candidate)]:
        byid = {r['uid']: r for r in rows}
        if len(rows) != len(byid) or set(byid) != {r['uid'] for r in weak}:
            raise ValueError('Outer prediction inventory differs')
        for i, r in enumerate(weak):
            if set(byid[r['uid']]['truth']) != set(MAP_CLASSES[j] for j in np.flatnonzero(y[i])):
                raise ValueError('Stored prediction truth differs')
        p = np.array([[int(c in byid[r['uid']]['predicted']) for c in MAP_CLASSES] for r in weak])
        predictions[name] = p
        results[name] = multilabel_metrics(y, p, w, MAP_CLASSES, groups)
    results['pairedIntervals'] = paired_group_intervals(y, predictions, w, groups)
    results['scope'] = {'records': len(weak), 'physicalWeight': float(w.sum()),
        'classes': MAP_CLASSES, 'historicalPriorExposure': 'unknown',
        'newIndependentEvaluation': False,
        'truthConvention': 'User confirmed on 2026-11-01: 7K whole-chart tags manually aim to cover primary player impressions; unlisted tags are negatives at the output threshold; weak continuous evidence should receive low continuous loss.',
        'averagePrecision': 'not computed: saved final tag vectors are hard decisions, not ranking scores'}
    return results


def run(args):
    root, out = Path(args.source), Path(args.output)
    protocol = read(root / 'protocol.json')
    policies = protocol['postPolicies']['ln_protected']
    configs = [dict(zip(protocol['fitGrid'], values))
               for values in itertools.product(*protocol['fitGrid'].values())]
    post_contexts = {k: matched_contexts(policies, k) for k in policies[0]
                     if len({json.dumps(c[k]) for c in policies}) > 1}
    fit_contexts = {k: matched_contexts(configs, k) for k in configs[0]}
    result = {'scope': 'diagnosis of saved geometry3/LN-protected inner predictions; no new fit or candidate selected',
        'source': str(root), 'postPolicyCount': len(policies),
        'postDesign': {k: {'matchedContexts': len(ids), 'policiesInMatchedContexts': len({i for g in ids for i in g}),
                         'testedValues': list(dict.fromkeys(json.dumps(c[k]) for c in policies))}
                       for k, ids in post_contexts.items()}, 'folds': []}
    for o in range(len(protocol['plans'])):
        folder = root / 'ln_protected/geometry3' / str(o)
        rows = read(folder / 'search.json.gz')
        selection = read(folder / 'selection.json')
        h, n = len(configs) + 1, len(policies)
        if [(r['head'], r['policy']) for r in rows] != list(itertools.product(range(h), range(n))):
            raise ValueError('Stored search ordering differs')
        scores = np.array([[r[k] for k in ['joint', 'mapMacroF1', 'segmentAccuracy']] for r in rows]).reshape(h, n, 3)
        with np.load(folder / 'inner-predictions.npz') as z:
            p = z['map'].reshape(h, n, -1, z['map'].shape[-1])
        chosen = selection['policyId']
        post = {}
        for k, contexts in post_contexts.items():
            pairs = [pair for ids in contexts for pair in itertools.combinations(ids, 2)]
            profile = next((ids for ids in contexts if chosen in ids), [chosen])
            post[k] = {**comparison_effects(p, scores, pairs, 'policy'),
                'selectedContextTestedValues': [policies[i][k] for i in profile],
                'selectedContextIdentifiableFromDesign': len(profile) > 1}
        fit = {k: comparison_effects(p, scores,
            [(i + 1, j + 1) for ids in contexts for i, j in itertools.combinations(ids, 2)], 'head')
            for k, contexts in fit_contexts.items()}
        packed = np.packbits(p.reshape(h*n, -1), axis=1)
        unique = len(np.unique(packed, axis=0))
        best = float(scores[..., 0].max())
        near = {str(e): int((scores[..., 0] >= best-e-1e-12).sum()) for e in [0., .001, .005]}
        result['folds'].append({'outer': o, 'jointConfigurations': h*n,
            'uniqueWholeTagVectorsOnInnerCorpus': unique, 'nearBestCounts': near,
            'postCoordinates': post, 'fitCoordinates': fit})
        print('audited saved fold', o, 'unique whole-tag vectors', unique, flush=True)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in [('landscape.json', result), ('multimetric.json', map_comparison(root, protocol))]:
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    print('diagnostic outputs:', out, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='research/rework/results/ln-extra-gate-r1')
    p.add_argument('--output', default='research/frontier/results/diagnosis-r1')
    run(p.parse_args())
