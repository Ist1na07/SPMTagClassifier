# 7K 回炉：原生流程联合校准结果

本阶段保留原专家系数先验，在同一最终任务目标下联合选择原生边界、专家混合/LN 门控、显示策略、相对负荷门控及聚合。每项消融重选全部可用参数；这是预先给定空间和候选预算内的比较，不是所有连续参数的全局最优。

| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |
|---|---:|---:|---:|
| factorial_scale | 52.16% | 65.88% | 64.41% |
| repaired | 53.29% | 62.13% | 60.83% |
| repaired_no_release | 53.84% | 68.33% | 66.41% |
| repaired_no_scale | 54.98% | 64.70% | 63.32% |
| repaired_normalized | 53.76% | 66.56% | 66.36% |
| repaired_soft | 49.16% | 62.15% | 61.17% |
| repaired_no_effort | 54.18% | 64.70% | 63.32% |
| repaired_no_confidence | 54.47% | 65.43% | 64.85% |
| repaired_no_diversity | 53.29% | 62.13% | 60.83% |
| original_fixed | 51.98% | 69.59% | 68.92% |
| normalized_fixed | 51.68% | 69.59% | 68.92% |
| original_aggregation_retuned | 53.11% | 69.59% | 68.92% |
| normalized_aggregation_retuned | 52.13% | 69.59% | 68.92% |

整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。

模型配置由各外层的内层预测联合选择；各方案的 `candidates.json`、`search.json.gz`、`selection.json` 与 `candidate-predictions.npz` 保留完整候选、评分和实际输出。实际分组见 `protocol.json`。

原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。
