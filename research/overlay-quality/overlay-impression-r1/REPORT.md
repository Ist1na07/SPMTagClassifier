# Overlay明显误判：原参数网格，新增源TRAIN玩家主类监督

| 策略 | macro P | macro R | macro F1 | micro F1 | 整谱全对 | 原六谱局部 | 源留出两条Speed摘录 |
|---|---:|---:|---:|---:|---:|---:|---:|
| original | 58.36% | 49.94% | 52.69% | 62.66% | 59.80% | 69.59% | 79.33% |
| macro | 59.24% | 50.07% | 53.02% | 62.92% | 60.09% | 69.41% | 100.00% |
| strict | 58.48% | 50.07% | 52.84% | 62.92% | 60.09% | 69.41% | 100.00% |
| no_action_rate | 59.24% | 50.07% | 53.02% | 62.92% | 60.09% | 69.41% | 100.00% |
| experience | 58.48% | 50.07% | 52.84% | 62.92% | 60.09% | 69.41% | 100.00% |
| shipped_fixed | 55.15% | 51.35% | 51.98% | 62.23% | 57.78% | — | — |

共同部署选择（INNER＋对应TRAIN监督）：{'original': {'configId': 0, 'config': {'ratio': 1, 'rowRateMax': 0}, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5311077336338339, 'macroRecall': 0.5049140548177161, 'localAccuracy': 0.6958552679007846, 'queryTrainPrimaryAccuracy': 0.7932666666666667}, 'macro': {'configId': 41, 'config': {'ratio': 0.85, 'rowRateMax': 0}, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5341195591337421, 'macroRecall': 0.5061398515779462, 'localAccuracy': 0.6941175811316515, 'queryTrainPrimaryAccuracy': 1.0}, 'strict': {'configId': 41, 'config': {'ratio': 0.85, 'rowRateMax': 0}, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5341195591337421, 'macroRecall': 0.5061398515779462, 'localAccuracy': 0.6941175811316515, 'queryTrainPrimaryAccuracy': 1.0}, 'no_action_rate': {'configId': 41, 'config': {'ratio': 0.85, 'rowRateMax': 0}, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5341195591337421, 'macroRecall': 0.5061398515779462, 'localAccuracy': 0.6941175811316515, 'queryTrainPrimaryAccuracy': 1.0}, 'experience': {'configId': 53, 'config': {'ratio': 0.95, 'rowRateMax': 16}, 'policyId': 39, 'policy': {'dominantShare': 0.52, 'mixShare': 0.05, 'secondRel': 1.0}, 'macroF1': 0.5311077336338339, 'macroRecall': 0.5049140548177161, 'localAccuracy': 0.6958552679007846, 'queryTrainPrimaryAccuracy': 1.0}}

原17类／六谱指标含义不变。两条玩家摘录是主动选择的回归，不是随机准确率。TRAIN主类符合度不是验证成绩；所有源留出结果按原外折组隔离，摘录标签未复制到整张来源谱。
