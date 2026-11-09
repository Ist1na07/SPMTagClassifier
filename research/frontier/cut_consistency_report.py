"""Consume completed bounded RC study; no fitting or criterion changes."""
from pathlib import Path
import numpy as np
from cut_consistency_model import World, KINDS, mil, read
from cut_consistency_study import ROOT, CONFIGS
from rc_action_report import unpack
from rc_action_runtime import policy_predictions
from landscape_audit import multilabel_metrics
from whole_reference import write


def run():
    complete = read(ROOT / 'complete.json'); protocol = read(ROOT / 'protocol.json')
    world = World(); uids = [r['uid'] for r in world.main]
    y = mil.encode(world.main); weights = np.array([r['recordWeight'] for r in world.main])
    groups = [r['group'] for r in world.main]
    rows = read(ROOT / 'outer-predictions.json'); local = read(ROOT / 'outer-local.json')
    concepts = read(ROOT / 'concept-predictions.json'); metrics = {}; profiles = {}; diagnostics = []
    original = [r for r in read('research/rework/results/aggregation/predictions.json') if r['variant'] == 'original_fixed']
    variants = ['original_fixed'] + [kind + '/' + variant for kind in KINDS for variant in
        ['full', 'no_concept', 'no_strong', 'adaptive', 'frozen', 'overlay_guarded']]
    for name in variants:
        if name == 'original_fixed':
            selected = original
        else:
            kind, variant = name.split('/')
            selected = [r for r in rows if r['kind'] == kind and r['variant'] == variant]
        lookup = {r['uid']: r for r in selected}
        if len(lookup) != len(selected) or set(lookup) != set(uids):
            raise ValueError('Incomplete or duplicate whole-chart inventory')
        predicted = np.array([[c in lookup[uid]['predicted'] for c in mil.MAP_CLASSES] for uid in uids], int)
        value = multilabel_metrics(y, predicted, weights, mil.MAP_CLASSES, groups)
        value['perClassErrors'] = {c: {label: [uids[i] for i in np.flatnonzero(mask)] for label, mask in
            [('FP', (y[:, j] == 0) & (predicted[:, j] == 1)), ('FN', (y[:, j] == 1) & (predicted[:, j] == 0))]}
            for j, c in enumerate(mil.MAP_CLASSES)}
        if name != 'original_fixed':
            samples = [r for r in local if r['kind'] == kind and r['variant'] == variant]
            if len(samples) != 6 or len({r['uid'] for r in samples}) != 6:
                raise ValueError('Incomplete six-chart annotated local inventory')
            value['localEqualChartAccuracy'] = float(np.mean([r['rawPrimaryIntervalAccuracy'] for r in samples]))
            value['localByChart'] = {r['uid']: r['rawPrimaryIntervalAccuracy'] for r in samples}
            value['mechanisms'] = []
            for case in concepts:
                if case['kind'] != kind or case['variant'] != variant:
                    continue
                value['mechanisms'].append(dict(uid=case['uid'], outer=case['outer'],
                    primary=[r['tag'] for r in case['frames']], whole=case['overall']['tags']))
        metrics[name] = value
    for kind in KINDS:
        profiles[kind] = []
        for outer in range(6):
            folder = ROOT / kind / str(outer)
            searches = read(folder / 'search.json')
            choice = read(folder / 'selection.json')['selections']['overlay_guarded']
            with np.load(folder / 'inner-bases.npz') as saved:
                arrays = {k: saved[k] for k in saved.files}
            reference = policy_predictions(unpack(arrays, choice['headId']), [choice['policy']])[0]
            reference_weight = weights[arrays['validationIds']]
            curves = {}
            for coordinate in ['localL2', 'strongWeight', 'conceptWeight', 'effortPower', 'primaryPower']:
                points = []
                for setting in dict.fromkeys(c[coordinate] for c in CONFIGS):
                    eligible = [r for r in searches if r['headId'] and r['config'][coordinate] == setting and
                        r['config']['conceptWeight'] > 0 and r['macroF1'] >= searches[0]['macroF1'] - 1e-12]
                    if not eligible:
                        points.append(dict(value=setting, eligibleActiveModel=False)); continue
                    best = max(eligible, key=lambda h: (h['segmentAccuracy'], h['macroF1'], h['macroRecall'], -h['headId'], -h['policyId']))
                    prediction = policy_predictions(unpack(arrays, best['headId']), [best['policy']])[0]
                    mass = float(reference_weight @ np.any(prediction != reference, axis=1) / reference_weight.sum())
                    points.append(dict(value=setting, eligibleActiveModel=True, selected=best, changedTagSetMass=mass))
                curves[coordinate] = points
            profiles[kind].append(dict(outer=outer, choice=choice, fullyRetunedConditionalProfiles=curves,
                coordinateMeaning='Training parameter profiles concern active models only. Frozen selection has no identifiable training parameter. Other legacy coordinates remain fixed as declared.'))
            for trace in read(folder / 'fits.json'):
                if trace['maximumProjectedGradient'] > 1e-5:
                    raise ValueError('Unaccepted optimizer state')
                losses = [r['loss'] for r in trace['starts']]
                diagnostics.append(dict(kind=kind, outer=outer, head=trace['headId'], inner=trace['inner'],
                    projectedGradient=trace['maximumProjectedGradient'], startLossSpread=max(losses)-min(losses)))
    write(ROOT / 'analysis.json', dict(complete=complete, metrics=metrics, profiles=profiles, optimization=diagnostics,
        scope=protocol['scope'], productionAdopted=False, nativeRuntimeAuditRequired=True,
        noProbabilityMetrics='Whole outputs are hard tag decisions, not calibrated posterior probabilities; no fabricated AP/Brier.'))
    lines = ['# 面向 overlay 的 Tech／CS 限域修正', '',
        '四种特征处理各自完整重调同一108训练配置×245后处理策略，沿用6×3歌曲分组。以下是反复研究过的资料上的回顾性评价，不是新的盲测。', '',
        '| 处理 | macro P | macro R | macro F1 | micro F1 | 整谱集合全对 | 人工局部按谱准确率 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['original_fixed'] + [k + '/' + v for k in KINDS for v in ['frozen', 'full', 'overlay_guarded']]:
        m = metrics[name]
        values = [m['macro']['precision'], m['macro']['recall'], m['macro']['f1'], m['micro']['f1'], m['exactMatch']]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{100*v:.2f}%' for v in values) + ' | ' +
            (f"{100*m['localEqualChartAccuracy']:.2f}%" if 'localEqualChartAccuracy' in m else '—') + ' |')
    lines.extend(['', 'H01/H03 仅监督“不是切”，没有把其主类伪造为 Tech。Speed、Jack、LN、Break 原始分数及 Dense/Fast 的条件比例保持不变。',
        '', '完整逐类 P/R/F1、误报漏报谱身份、构造／真实局部读数、条件参数剖面和优化状态在 analysis.json。',
        '', '本报告不自动采用模型；生产端原始分段、平滑、聚合和浏览器消费者一致性仍须单独验证。'])
    (ROOT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    run()
