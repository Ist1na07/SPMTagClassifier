"""Reuse an existing SPMRating result without invoking its scorer again.

The public bridge only needs numpy and an already computed include_curve result.
The optional component probe is research-only, pinned to upstream v1.2.0; it
observes the original predictor in a private namespace, never patches modules.
"""
import importlib
from pathlib import Path
import subprocess
import sys
from types import FunctionType, MethodType, SimpleNamespace
import numpy as np

RATING_COMMIT = 'a914a405d8e9a77f94c0518567c37511b0b7ac05'


class RatingEvidence:
    def __init__(self, result, *, expected_keys, expected_rate, channels=None):
        if result['key_count'] != expected_keys or result['rate'] != expected_rate:
            raise ValueError('Rating/key-count or post-rate clock mismatch')
        self.query = np.array(result['query'], dtype=float, copy=True)
        self.channels = {'demand': np.array(result['field'], dtype=float, copy=True)}
        if 'demand' in (channels or {}):
            raise ValueError('Components must not replace the public demand field')
        self.channels.update({k: np.array(v, dtype=float, copy=True) for k, v in (channels or {}).items()})
        if self.query.ndim != 1 or not len(self.query) or not np.all(np.isfinite(self.query)) or not np.all(np.diff(self.query) > 0):
            raise ValueError('Expected strictly increasing finite head-row query times')
        if any(v.shape != self.query.shape or not np.all(np.isfinite(v)) for v in self.channels.values()):
            raise ValueError('Finite component arrays on the SAME query grid required')
        self.metadata = {k: result[k] for k in ['model', 'model_identity', 'key_count', 'rate', 'raw_score']}

    def window(self, start, end):
        if not np.isfinite(start) or not np.isfinite(end) or not start < end:
            raise ValueError('Finite nonempty post-rate half-open interval required')
        take = (self.query >= start) & (self.query < end)
        count = int(take.sum())
        return {'start': start, 'end': end, 'headRows': count,
            'sampling': 'existing head-row demand samples; not uniform time or per-tail coverage',
            'channels': {k: {'rowMean': float(v[take].mean()) if count else None,
                'rowP90': float(np.quantile(v[take], .9)) if count else None,
                'rowMax': float(v[take].max()) if count else None} for k, v in self.channels.items()}}

    def envelope(self):
        return {**self.metadata, 'clock': 'post-rate milliseconds',
            'sampling': 'head-row field; headless intervals are missing, NOT zero difficulty',
            'query': self.query.tolist(), 'channels': {k: v.tolist() for k, v in self.channels.items()}}


def load_rating(root):
    root = Path(root).resolve()
    revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != RATING_COMMIT:
        raise ValueError('The private research probe requires the pinned SPMRating v1.2.0 checkout')
    sys.path.insert(0, str(root))
    rating = importlib.import_module('spm_rating')
    if Path(rating.__file__).resolve().parent.parent != root:
        raise ValueError('A different SPMRating module is already loaded')
    return rating, importlib.import_module('spm_rating.engine')


def capture_native(notes, model, engine_module):
    """Observe actual native-grid components during ONE predictor call.

    Fractional-source phase extension belongs to the public API. The probe
    refuses that path rather than silently relabeling one phase as its mean.
    """
    from spm_rating import role, clock
    if clock.plan(notes) is not None:
        raise ValueError('Private component probe does not represent fractional-source phase averaging; use the public curve bridge')
    if model.physical_objects:
        notes = role.canonical(notes)
    original = engine_module.Engine.predict
    numerical_model = original.__globals__['model']
    views = []

    def combine(struct, curves, parameters):
        views.append((struct.t.copy(), {k: v.copy() for k, v in curves.items()
            if isinstance(v, np.ndarray) and v.shape == struct.t.shape}))
        return numerical_model.combine(struct, curves, parameters)

    proxy = SimpleNamespace(combine=combine, aggregate=numerical_model.aggregate,
                            postprocess=numerical_model.postprocess)
    function = FunctionType(original.__code__, {**original.__globals__, 'model': proxy},
                            original.__name__, original.__defaults__)
    function.__kwdefaults__ = original.__kwdefaults__
    result = MethodType(function, engine_module.Engine())(notes, dict(model.parameters), model.role_price)
    if len(views) != 2 or any(not np.array_equal(t, result['query']) for t, _ in views):
        raise ValueError('Upstream view/query contract changed')
    if set(views[0][1]) != set(views[1][1]):
        raise ValueError('Upstream component inventories differ')
    channels = {k: .5*(views[0][1][k]+views[1][1][k]) for k in views[0][1]}
    public_shape = dict(model=model.name, model_identity=model.identity, key_count=notes.key_count,
        rate=notes.rate, raw_score=float(result['raw_score']), query=result['query'], field=result['field'])
    return RatingEvidence(public_shape, expected_keys=notes.key_count, expected_rate=notes.rate, channels=channels), {
        'source': 'private native-grid component observation; not a public upstream API',
        'ratingCommit': RATING_COMMIT, 'predictorCalls': 1, 'mirrorViews': 2,
        'componentMeaning': 'mean of mirror-view component fields; components do not add up to total difficulty',
        'sourceCoordinateFrame': 'SPMRating embedded seven-column frame; native key count retained'}
