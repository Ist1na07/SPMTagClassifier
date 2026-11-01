"""Extract pinned frozen SPMRating evidence ONCE per physical file/model.

All original399 record/interval identities remain in the manifest. Tag labels
are not read for feature computation. max-records permits a resource check;
it never writes a misleading complete.json for a partial corpus.
"""
import argparse
from pathlib import Path
import json
import time
import subprocess
import numpy as np
from rating_bridge import load_rating, capture_native, RATING_COMMIT


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf8')
    temporary.replace(path)


def run(args):
    rating, engine = load_rating(args.rating_root)
    records = [json.loads(s) for s in Path('.cache/rework7.jsonl').read_text(encoding='utf8').splitlines()]
    root = Path(args.output); models = {k: rating.load_model(k) for k in ['detail', 'compact']}
    protocol = {'id': 'frozen-spm-rating-evidence-r1', 'ratingTag': 'v1.2.0', 'ratingCommit': RATING_COMMIT,
        'models': {k: m.identity for k, m in models.items()},
        'scope': 'same original7K records and raw automatic intervals; frozen external difficulty prior; no tag fitting',
        'input': [{'uid': r['uid'], 'source': r['source'], 'intervals': [[s['start'], s['end']] for s in r['segments']]} for r in records],
        'clock': 'post-rate milliseconds, rate1; source native clock required for private component observation',
        'featureSampling': 'head-row statistics, missing observations retained as null, not zero difficulty',
        'provenance': 'upstream difficulty parameters are frozen pretrained information; not a clean fold-retrained tag prior',
        'sourceSnapshots': {name: Path(__file__).with_name(name).read_text(encoding='utf8') for name in ['rating_bridge.py', 'extract_rating.py']}}
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text(encoding='utf8')) != protocol:
        raise ValueError('Rating extraction protocol changed; use a new directory')
    write(root/'protocol.json', protocol)
    canonical = {}
    for r in records: canonical.setdefault(r['source'], r['uid'].replace('/', '__'))
    chosen = records[:args.max_records] if args.max_records else records
    for index, r in enumerate(chosen):
        paths = {k: root/k/(r['uid'].replace('/', '__')+'.json') for k in models}
        if all(p.exists() for p in paths.values()): continue
        notes = engine.parser.parse_file(r['source'], rate=1.)
        for k, model in models.items():
            curve_path = root/'curves'/k/(canonical[r['source']]+'.npz')
            if paths[k].exists(): continue
            before = time.perf_counter(); computed = False
            if curve_path.exists():
                with np.load(curve_path) as z:
                    query = z['query']; channels = {n: z[n] for n in z.files if n not in ['query', 'raw_score']}
                    raw = float(z['raw_score'])
                from rating_bridge import RatingEvidence
                e = RatingEvidence(dict(model=k, model_identity=model.identity, key_count=notes.key_count, rate=1., raw_score=raw,
                    query=query, field=channels.pop('demand')), expected_keys=notes.key_count, expected_rate=1., channels=channels)
                meta = {'source': 'same physical source/model curve reused', 'predictorCalls': 0}
            else:
                e, meta = capture_native(notes, model, engine); curve_path.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(curve_path, query=e.query, raw_score=np.array(e.metadata['raw_score']), **e.channels)
                computed = True
            windows = [e.window(s['start'], s['end']) for s in r['segments']]
            write(paths[k], {'uid': r['uid'], 'source': r['source'], 'rating': e.metadata, 'probe': meta,
                'curvePath': str(curve_path), 'newPredictorCall': computed, 'extractionSeconds': time.perf_counter()-before,
                'wholeCurve': e.window(float(e.query[0]), float(np.nextafter(e.query[-1], np.inf))), 'windows': windows})
        if (index+1)%10 == 0 or index < 3: print('rating features', index+1, '/', len(records), r['uid'], flush=True)
    complete = all((root/k/(r['uid'].replace('/', '__')+'.json')).exists() for r in records for k in models)
    if complete:
        rows = [json.loads((root/k/(r['uid'].replace('/', '__')+'.json')).read_text(encoding='utf8')) for r in records for k in models]
        summary = {'records': len(records), 'models': list(models), 'recordModelOutputs': len(rows),
            'uniquePhysicalSources': len(canonical), 'newPredictorCalls': sum(r['newPredictorCall'] for r in rows),
            'windowsPerModel': sum(len(r['segments']) for r in records),
            'headlessWindowsPerModel': {k: sum(w['headRows']==0 for r in rows if r['rating']['model']==k for w in r['windows']) for k in models},
            'scope': protocol['scope'], 'ratingCommit': RATING_COMMIT}
        write(root/'complete.json', summary); print(summary, flush=True)
    else: print('Partial resource check; no complete.json written.', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--rating-root', default='.cache/spm-rating-v1.2.0')
    p.add_argument('--output', default='.cache/frontier-rating-r1'); p.add_argument('--max-records', type=int)
    run(p.parse_args())
