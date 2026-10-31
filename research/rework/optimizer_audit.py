"""Audit stored MIL fits, including the control heads' separate local stage.

Reads actual termination evidence; never substitutes optimizer messages for the
declared training-gradient or training-prediction stability acceptance policy.
"""
import json
from pathlib import Path
from collections import Counter

ACCEPTED = {'small_gradient', 'small_gradient_reused', 'train_prediction_stable'}


def audit_rows(rows):
    counts, local_counts, failed, gradients = Counter(), Counter(), [], []
    for row in rows:
        fit = row['fit']
        for stage, value, target in [('head', fit, counts), ('local', fit.get('localStage'), local_counts)]:
            if value is None:
                continue
            evidence = value.get('refinement', {})
            status = evidence.get('status', 'missing_stop_evidence')
            target[status] += 1
            if evidence.get('gradientMax') is not None:
                gradients.append(evidence['gradientMax'])
            if status not in ACCEPTED:
                failed.append({k: row[k] for k in ['key', 'variant', 'outer', 'inner', 'configId'] if k in row} |
                              {'stage': stage, 'status': status})
    return {'fits': len(rows), 'headStatuses': dict(counts), 'localStageStatuses': dict(local_counts),
            'unaccepted': failed, 'maxReportedGradient': max(gradients, default=None)}


def audit_directory(root):
    root = Path(root)
    result = {name: audit_rows(json.loads((root / filename).read_text(encoding='utf8')))
              for name, filename in [('inner', 'fits.json'), ('outer', 'outer-fits.json')]}
    result['accepted'] = not any(result[k]['unaccepted'] for k in ['inner', 'outer'])
    result['meaning'] = 'Prediction/loss stability is an allowed training-only stopping condition, not a stationary-point claim.'
    (root / 'optimizer-audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    return result


def report_lines(audit):
    failed = sum(len(audit[k]['unaccepted']) for k in ['inner', 'outer'])
    lines = ['', '## 优化终止审计', '',
             ('所有内层候选及外层拟合都达到预先声明的训练梯度或连续训练目标/预测稳定条件。'
              if audit['accepted'] else f'**仍有 {failed} 项拟合未达到声明的终止条件，本轮分数仅作暂定诊断，不能用于最终定型。**'),
             '', '| 拟合范围 | 主阶段实际状态 | 独立局部阶段实际状态 |', '|---|---|---|']
    for scope in ['inner', 'outer']:
        value = audit[scope]
        show = lambda d: ', '.join(f'{k}: {v}' for k, v in d.items()) or '无'
        lines.append(f"| {scope} | {show(value['headStatuses'])} | {show(value['localStageStatuses'])} |")
    return lines + ['', '连续预测/目标稳定不等于梯度已趋近于零。逐项未接受记录与实际梯度上界见 `optimizer-audit.json`。']
