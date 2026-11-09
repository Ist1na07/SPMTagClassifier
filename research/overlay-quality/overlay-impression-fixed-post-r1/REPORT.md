# Overlay明显误判：原参数网格，新增源TRAIN玩家主类监督

| 策略 | macro P | macro R | macro F1 | micro F1 | 整谱全对 | 原六谱局部 | 源留出两条Speed摘录 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original_fixed_post | 55.15% | 51.35% | 51.98% | 62.23% | 57.78% | 69.59% | 79.33% |
| experience_fixed_post | 55.07% | 51.47% | 52.02% | 62.41% | 58.07% | 69.41% | 100.00% |
| shipped_fixed | 55.15% | 51.35% | 51.98% | 62.23% | 57.78% | — | — |

共同部署选择（INNER＋对应TRAIN监督）：{'original_fixed_post': {'configId': 0, 'config': {'ratio': 1, 'rowRateMax': 0}, 'policyId': 42, 'policy': {'dominantShare': 0.52, 'mixShare': 0.07, 'secondRel': 0.75}, 'macroF1': 0.51376192535686, 'macroRecall': 0.5111420682775678, 'localAccuracy': 0.6958552679007846, 'queryTrainPrimaryAccuracy': 0.7932666666666667}, 'experience_fixed_post': {'configId': 45, 'config': {'ratio': 0.85, 'rowRateMax': 16}, 'policyId': 42, 'policy': {'dominantShare': 0.52, 'mixShare': 0.07, 'secondRel': 0.75}, 'macroF1': 0.5141122080522745, 'macroRecall': 0.5123678650377979, 'localAccuracy': 0.6958552679007846, 'queryTrainPrimaryAccuracy': 1.0}}

原17类／六谱指标含义不变。两条玩家摘录是主动选择的回归，不是随机准确率。TRAIN主类符合度不是验证成绩；所有源留出结果按原外折组隔离，摘录标签未复制到整张来源谱。
