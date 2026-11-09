# 无绝对密度上限的规则节奏休息门控：开／关对照

| 处理 | macro P | macro R | macro F1 | micro F1 | 集合全对 |
|---|---:|---:|---:|---:|---:|
| original/retuned | 58.36% | 49.94% | 52.69% | 62.66% | 59.80% |
| original/fixed | 55.15% | 51.35% | 51.98% | 62.23% | 57.78% |
| candidate/retuned | 59.31% | 50.19% | 53.15% | 63.19% | 60.37% |
| candidate/fixed | 55.48% | 51.60% | 52.24% | 62.67% | 58.36% |

{'original': {'equalChartAccuracy': 0.6958552679007847, 'byChart': {'manual/AA': 0.6690082644628099, 'manual/AH': 0.75644375, 'manual/BE': 0.6510489219701853, 'manual/DS': 0.6398873620847826, 'manual/EE': 0.6110863525596655, 'manual/RJ': 0.8476569563272648}, 'falseBreakMs': 72306.0, 'missedBreakMs': 109238.0}, 'candidate': {'equalChartAccuracy': 0.6859945021779946, 'byChart': {'manual/AA': 0.6489669421487604, 'manual/AH': 0.7384229166666667, 'manual/BE': 0.6510489219701853, 'manual/DS': 0.6187849233954248, 'manual/EE': 0.6110863525596655, 'manual/RJ': 0.8476569563272648}, 'falseBreakMs': 71331.0, 'missedBreakMs': 126778.5}}

完整245聚合策略在两处理下各自重调；局部分类器没有新增参数。所有外折选择仅使用对应TRAIN内部资料。F01/F02是主动查询的回归例，不作随机准确率。
