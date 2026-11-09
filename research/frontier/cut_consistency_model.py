"""Narrow Tech-versus-CS odds correction; other class scores remain fixed.

Dense/Fast CS keep their original conditional ratio. Unlike the earlier
Speed/Tech/Minijack redistribution, this cannot take probability from Speed,
Jack, LN or Break. H01/H03 supervise only NOT-CS, not an invented Tech label.
"""
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from rc_type_model import World as BaseWorld, fit as existing_fit, mil, read, SCHEMA
from type_bag_objective import TypeBagObjective

TYPES = ['dense chordstream', 'fast chordstream', 'tech']
INDEX = [mil.CLASSES.index(c) for c in TYPES]
OTHER = [i for i in range(len(mil.CLASSES)) if i not in INDEX]
KINDS = ['bias', 'hand', 'rhythm', 'both']
NEGATIVE_CS = ['concept/H01', 'concept/H03']
HAND_KEYS = ['handSubstitutionFraction', 'handFixedGripTransitionShare']
RHYTHM_KEYS = ['logIoiSd', 'ioiChange']


class World(BaseWorld):
    def __init__(self):
        super().__init__()
        self.cuts = {r['uid']: r for r in map(__import__('json').loads,
            Path('.cache/overlay-cut-inputs-r1.jsonl').read_text().splitlines())}
        self.scalers = {}
        for uid, row in self.inputs.items():
            if row['intervals'] != self.cuts[uid]['intervals']:
                raise ValueError('Cut inputs must retain exact original intervals')

    def feature_keys(self, kind):
        if kind not in KINDS:
            raise ValueError('Unknown bounded cut representation')
        return (HAND_KEYS if kind in ['hand', 'both'] else []) + (
            RHYTHM_KEYS if kind in ['rhythm', 'both'] else [])

    def features(self, uid, kind):
        n = len(self.inputs[uid]['intervals'])
        columns = []
        if kind in ['hand', 'both']:
            columns.extend(np.asarray([w['values'] for w in self.cuts[uid]['windows']], float).T)
        if kind in ['rhythm', 'both']:
            event = np.asarray(self.inputs[uid]['event'], float)
            columns.extend(event[:, SCHEMA['eventKeys'].index(k)] for k in RHYTHM_KEYS)
        return np.column_stack(columns) if columns else np.empty((n, 0))

    def scaler(self, records, kind):
        key = tuple(r['uid'] for r in records), kind
        if key in self.scalers:
            return self.scalers[key]
        arrays, weights = [], []
        for row in records:
            take = self.teacher(row['uid'])[:, INDEX].sum(1) > 0
            if take.any():
                arrays.append(self.features(row['uid'], kind)[take])
                weights.append(np.full(take.sum(), row['recordWeight'] / take.sum()))
        raw, weight = np.concatenate(arrays), np.concatenate(weights)
        mu = np.array([np.average(raw[np.isfinite(raw[:, j]), j], weights=weight[np.isfinite(raw[:, j])])
            if np.isfinite(raw[:, j]).any() else 0. for j in range(raw.shape[1])])
        filled = np.where(np.isfinite(raw), raw, mu)
        expanded = np.c_[filled, ~np.isfinite(raw)]
        mean = np.average(expanded, axis=0, weights=weight)
        scale = np.sqrt(np.average((expanded - mean) ** 2, axis=0, weights=weight))
        keep = np.ptp(expanded, axis=0) > 0
        keys = self.feature_keys(kind)
        if kind in ['hand', 'both']:
            # Both hand fractions are jointly observed or missing by
            # construction. Do not allocate two identical missingness slopes.
            if not np.array_equal(expanded[:, len(keys)], expanded[:, len(keys) + 1]):
                raise ValueError('Hand descriptor missingness contract changed')
            keep[len(keys) + 1] = False
        result = dict(rawImputation=mu.tolist(), mean=mean.tolist(), scale=np.where(keep, scale, 1.).tolist(),
            keep=keep.tolist(), expandedKeys=keys + [k + ':missing' for k in keys])
        self.scalers[key] = result
        return result

    def prior(self, uid):
        value = self.teacher(uid)[:, INDEX]
        mass = value.sum(1)
        conditional = np.divide(value, mass[:, None], out=np.full_like(value, 1 / 3), where=mass[:, None] > 0)
        # Preserve exact zero support; no numerical floor creates a new class.
        with np.errstate(divide='ignore'):
            return np.log(conditional)


class Packed(TypeBagObjective):
    def __init__(self, world, main_train, manual_train, *, kind='both'):
        self.world, self.kind = world, kind
        self.main_train, self.manual_train = main_train, manual_train
        self.types, self.index, self.other = TYPES, INDEX, OTHER
        self.contrast = np.array([[0.], [0.], [1.]])
        self.scaler = world.scaler(main_train, kind)
        weak = [r for r in main_train if len(mil.target(r)) == 1 and mil.target(r)[0] in TYPES]
        self.rows = weak + manual_train + [world.records[u] for u in NEGATIVE_CS]
        self.weak, self.strong, self.concepts, self.ranks, self.negative, self.offsets = [], [], [], [], [], {}
        xs, priors, teachers = [], [], []
        offset = 0
        total_weight = sum(r['recordWeight'] for r in weak)
        for row in self.rows:
            uid = row['uid']; teacher = world.teacher(uid); n = len(teacher)
            self.offsets[uid] = offset, offset + n
            xs.extend(world.design(uid, kind, self.scaler)); priors.extend(world.prior(uid)); teachers.extend(teacher)
            duration = np.diff(np.asarray(world.inputs[uid]['intervals']), axis=1)[:, 0]
            effort = np.maximum(1e-6, [s['effort'] for s in row['segments']])
            if row in weak:
                self.weak.append(dict(range=(offset, offset + n), target=TYPES.index(mil.target(row)[0]),
                    duration=duration, effort=effort, weight=row['recordWeight'] / total_weight))
            if row['dataset'] == 'manual':
                total = sum(a['end'] - a['start'] for a in row['annotations'] if a['tags'])
                for annotation in row['annotations']:
                    mask = np.array([c in annotation['tags'] for c in mil.CLASSES], float)
                    if not mask[INDEX].any():
                        continue
                    for j, (a, b) in enumerate(world.inputs[uid]['intervals']):
                        d = max(0, min(b, annotation['end']) - max(a, annotation['start']))
                        if d:
                            self.strong.append((offset + j, mask, d / total / max(1, len(manual_train))))
            if uid in NEGATIVE_CS:
                self.negative.extend((offset + j, float(d / duration.sum() / len(NEGATIVE_CS))) for j, d in enumerate(duration))
            offset += n
        self.x, self.log_prior, self.teacher = np.array(xs), np.array(priors), np.array(teachers)
        self.mass = self.teacher[:, INDEX].sum(1)
        self.dimension = self.x.shape[1]
        self.class_mass = np.array([sum(r['weight'] for r in self.weak if r['target'] == c) for c in range(len(TYPES))])
        self.localTrainUids = []  # No real-query answers are imported into this bounded head.

    def log_distribution(self, theta):
        z = self.log_prior + (self.x @ theta)[:, None] * self.contrast.T
        return z - logsumexp(z, axis=1, keepdims=True)

    def objective(self, theta, cfg):
        loss, gradient = super().objective(theta, cfg)
        if cfg['conceptWeight']:
            q = self.distribution(theta)
            for index, weight in self.negative:
                # P(not CS) includes EVERY unchanged non-CS class. This is
                # neither a mandatory Tech target nor an exhaustive slot label.
                probability = self.teacher[index, OTHER].sum() + self.mass[index] * q[index, 2]
                loss -= cfg['conceptWeight'] * weight * np.log(probability)
                derivative = cfg['conceptWeight'] * weight * self.mass[index] * q[index, 2] / probability * (q[index, 2] - 1)
                gradient += derivative * self.x[index]
        return float(loss), gradient


def fit(packed, cfg, seed_theta=None):
    model, trace = existing_fit(packed, cfg, seed_theta)
    model.update(types=TYPES, residual='Tech versus combined CS odds; Dense/Fast conditional ratio unchanged')
    return model, trace


def probability(world, uid, model):
    teacher = world.teacher(uid)
    if not np.any(model['theta']):
        return teacher.copy()
    x = world.design(uid, model['kind'], model['scaler'])
    z = world.prior(uid) + (x @ np.asarray(model['theta']))[:, None] * np.array([[0., 0., 1.]])
    q = np.exp(z - logsumexp(z, axis=1, keepdims=True))
    result = teacher.copy()
    result[:, INDEX] = q * teacher[:, INDEX].sum(1, keepdims=True)
    return result


def frames(world, uid, model=None):
    source = world.inputs[uid]['frames']
    if model is None or not np.any(model['theta']):
        return [dict(f) for f in source]
    value = probability(world, uid, model); out = []
    for j, frame in enumerate(source):
        if frame['family'] != 'rc' or not world.teacher(uid)[j, INDEX].sum():
            out.append(dict(frame)); continue
        scores = dict(frame['scores']); total = sum(scores.values())
        for c, index in zip(TYPES, INDEX):
            scores[c] = float(value[j, index] * total)
        ranked = sorted(scores.items(), key=lambda p: -p[1]); tag, confidence = ranked[0]
        if tag in ['wildcard', 'ln wildcard']:
            for c, v in ranked[1:]:
                if c not in ['wildcard', 'ln wildcard'] and v > .3:
                    tag, confidence = c, v; break
        secondary = ranked[1][0] if len(ranked) > 1 and ranked[1][1] >= ranked[0][1] * .75 and ranked[1][1] > .3 else None
        out.append({**frame, 'scores': scores, 'tag': tag, 'confidence': min(1., confidence), 'secondary': secondary})
    return out


def segment_accuracy(world, records, model):
    result = []
    for row in records:
        predicted = frames(world, row['uid'], model); correct = total = 0.
        for a in row['annotations']:
            if not a['tags']:
                continue
            total += a['end'] - a['start']
            for f in predicted:
                correct += max(0, min(a['end'], f['end']) - max(a['start'], f['start'])) * (f['tag'] in a['tags'])
        result.append(correct / total if total else 0.)
    return result
