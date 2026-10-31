# 7K 回炉：原生流程联合校准结果

本阶段保留原专家系数先验，在同一最终任务目标下联合选择原生边界、专家混合/LN 门控、显示策略、相对负荷门控及聚合。每项消融重选全部可用参数；这是预先给定空间和候选预算内的比较，不是所有连续参数的全局最优。

| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |
|---|---:|---:|---:|
| original | 53.22% | 66.52% | 64.55% |
| area | 52.94% | 66.52% | 64.55% |
| adjacency | 52.20% | 66.65% | 65.77% |
| heldOrder | 53.22% | 66.52% | 64.55% |
| phase | 54.08% | 66.47% | 64.50% |
| sustaining | 52.33% | 66.52% | 64.55% |
| neutralHands | 53.64% | 65.19% | 63.31% |
| coverage | 53.22% | 66.52% | 64.55% |
| transformedScale | 51.29% | 65.47% | 63.82% |
| logical | 55.03% | 65.31% | 64.63% |
| logical_neutral | 54.11% | 69.72% | 69.04% |
| release | 53.78% | 67.27% | 65.19% |
| logical_release | 52.62% | 65.76% | 66.06% |
| full | 52.35% | 62.12% | 61.31% |
| full_legacy | 53.53% | 62.12% | 61.31% |
| full_normalized | 52.35% | 62.12% | 61.31% |
| full_soft | 52.58% | 68.48% | 68.48% |
| full_soft_no_confidence | 52.58% | 68.48% | 68.48% |
| full_soft_no_effort | 52.07% | 66.45% | 65.15% |
| full_soft_no_diversity | 51.64% | 68.48% | 68.48% |
| original_soft_coherent | 51.94% | 67.65% | 65.68% |
| neutral_soft_coherent | 49.53% | 65.19% | 63.31% |
| mirror_ensemble | 50.58% | 67.19% | 65.18% |
| full_share_only | 52.35% | 62.12% | 61.31% |
| full_soft_include_vibro | 52.58% | 68.48% | 68.48% |
| original_fixed | 51.98% | 69.59% | 68.92% |
| normalized_fixed | 51.68% | 69.59% | 68.92% |
| original_aggregation_retuned | 53.11% | 69.59% | 68.92% |
| normalized_aggregation_retuned | 52.13% | 69.59% | 68.92% |

整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。

模型配置由各外层的内层预测联合选择；各方案的 `candidates.json`、`search.json.gz`、`selection.json` 与 `candidate-predictions.npz` 保留完整候选、评分和实际输出。实际分组见 `protocol.json` 和 `supports.json`。

原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。
