"""Same restricted Tech/CS likelihood, now with bilateral action features.

Partition changes are upstream physical signals, not label-derived cuts.
The original trained-data folds, type odds correction and loss stay intact.
"""
from pathlib import Path
import json
import numpy as np
from cut_consistency_model import World as BaseWorld, Packed, fit, probability, frames, segment_accuracy, read

ROOT = Path('.cache/overlay-bilateral-partitions-r1')
MODEL_KINDS = ['bias', 'hand', 'bilateral', 'both']
KEYS = ['bilateralRowShare', 'bilateralNoteShare']
HAND_KEYS = ['handSubstitutionFraction', 'handFixedGripTransitionShare']


class World(BaseWorld):
    def __init__(self, variant, partition):
        super().__init__()
        self.variant, self.partition = variant, partition
        read(ROOT / f'{variant}-{partition}.jsonl.complete.json')
        self.inputs = {r['uid']: r for r in map(json.loads, (ROOT / f'{variant}-{partition}.jsonl').read_text().splitlines())}
        hand_file=Path('.cache/overlay-partition-hands-r1')/f'{variant}-{partition}.jsonl'
        self.partition_hands={r['uid']:r for r in map(json.loads,hand_file.read_text().splitlines())}
        self.cache = {}; self.scalers = {}
        for uid, row in self.inputs.items():
            if row['intervals']!=self.partition_hands[uid]['intervals']:raise ValueError('Partition hand timing differs')
            self.records[uid]['segments'] = [dict(start=a, end=b, effort=e)
                for (a, b), e in zip(row['intervals'], row['effort'])]

    def feature_keys(self, kind):
        if kind not in MODEL_KINDS:
            raise ValueError('Unknown bilateral model representation')
        return (HAND_KEYS if kind in ['hand','both']else[])+(KEYS if kind in ['bilateral','both']else[])

    def features(self, uid, kind):
        if kind == 'bias':
            return np.empty((len(self.inputs[uid]['intervals']), 0))
        if kind not in MODEL_KINDS:
            raise ValueError('Unknown bilateral model representation')
        arrays=[]
        if kind in ['hand','both']:arrays.append(np.asarray(self.partition_hands[uid]['values'],float))
        if kind in ['bilateral','both']:arrays.append(np.asarray(self.inputs[uid]['bilateral'],float))
        return np.concatenate(arrays,axis=1)


def feature_names(p):
    return ['intercept'] + [k for k, keep in zip(p.scaler['expandedKeys'], p.scaler['keep']) if keep]


def lift_bias(source, target, theta, cfg):
    value = np.zeros(target.dimension)
    for key, coefficient in zip(feature_names(source), theta):
        value[feature_names(target).index(key)] = coefficient
    np.testing.assert_allclose(source.x @ theta, target.x @ value, atol=1e-12)
    if abs(source.objective(theta, cfg)[0] - target.objective(value, cfg)[0]) > 1e-10:
        raise ValueError('Bilateral extension changed nested TRAIN objective')
    return value
