# 规则节奏Break门控：相对密度×聚合完整联调

| 选择 | macro P | macro R | macro F1 | micro F1 | 集合全对 | 局部准确率 |
|---|---:|---:|---:|---:|---:|---:|
| original | 58.36% | 49.94% | 52.69% | 62.66% | 59.80% | 69.59% |
| macro | 59.24% | 50.07% | 53.02% | 62.92% | 60.09% | 69.41% |
| overlay_guarded | 58.48% | 50.07% | 52.84% | 62.92% | 60.09% | 69.41% |

{'original': {'equalChartAccuracy': 0.6958552679007847, 'byChart': {'manual/AA': 0.6690082644628099, 'manual/AH': 0.75644375, 'manual/BE': 0.6510489219701853, 'manual/DS': 0.6398873620847826, 'manual/EE': 0.6110863525596655, 'manual/RJ': 0.8476569563272648}, 'falseBreakMs': 72306.0, 'missedBreakMs': 109238.0}, 'macro': {'equalChartAccuracy': 0.6941175811316516, 'byChart': {'manual/AA': 0.6690082644628099, 'manual/AH': 0.75644375, 'manual/BE': 0.6510489219701853, 'manual/DS': 0.6294612414699836, 'manual/EE': 0.6110863525596655, 'manual/RJ': 0.8476569563272648}, 'falseBreakMs': 72306.0, 'missedBreakMs': 113488.5}, 'overlay_guarded': {'equalChartAccuracy': 0.6941175811316516, 'byChart': {'manual/AA': 0.6690082644628099, 'manual/AH': 0.75644375, 'manual/BE': 0.6510489219701853, 'manual/DS': 0.6294612414699836, 'manual/EE': 0.6110863525596655, 'manual/RJ': 0.8476569563272648}, 'falseBreakMs': 72306.0, 'missedBreakMs': 113488.5}}

共同部署选择（仅INNER）：{'original': {'ratio': 1.0, 'ratioId': 7, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5311077336338339, 'macroRecall': 0.5049140548177161, 'localAccuracy': 0.6958552679007846}, 'macro': {'ratio': 0.85, 'ratioId': 5, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5341195591337421, 'macroRecall': 0.5061398515779462, 'localAccuracy': 0.6941175811316515}, 'overlay_guarded': {'ratio': 1.0, 'ratioId': 7, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5311077336338339, 'macroRecall': 0.5049140548177161, 'localAccuracy': 0.6958552679007846}}

全部8相对密度上限×245聚合策略完整联调；0与1分别复现删除／原规则。未改变其它Break条件或类型模型。旧资料上的回顾性评价，不是新盲测。
