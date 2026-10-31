# 7K：嵌套选择流程更新容量

保持父实验全部固定预测不变。在每个外层的内层训练子集上，各容量预算同时选择整条策略；其内层验证输出再选择容量，最后在完整外层训练组上重选策略。主候选为 metric_repaired，其余限制各自重调；固定最小/不限制容量只作机制对照。

| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |
|---|---:|---:|---:|
| metric_repaired | 52.07% | 67.14% | 66.55% |
| capacity_minimum_control | 51.58% | 66.81% | 66.32% |
| capacity_unrestricted_control | 53.47% | 64.89% | 64.52% |
| metric_no_release | 51.98% | 69.72% | 69.04% |
| metric_old_gain | 52.09% | 67.14% | 66.55% |
| metric_full_scale | 52.42% | 66.68% | 66.01% |
| metric_normalized | 52.09% | 67.14% | 66.55% |
| metric_soft | 51.00% | 64.12% | 62.70% |
| metric_no_effort | 52.46% | 67.14% | 66.55% |
| metric_no_confidence | 53.56% | 67.14% | 66.55% |
| metric_no_diversity | 52.07% | 67.14% | 66.55% |
| metric_original_control | 51.79% | 66.47% | 64.50% |
| metric_logical_control | 52.68% | 65.29% | 63.50% |
| original_fixed | 51.98% | 69.59% | 68.92% |
| normalized_fixed | 51.68% | 69.59% | 68.92% |
| original_aggregation_retuned | 53.11% | 69.59% | 68.92% |
| normalized_aggregation_retuned | 52.13% | 69.59% | 68.92% |

整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。


原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。

父候选、全部固定预测和分段缓存见 `../native-metric-r1/`。本轮各方案 `selection.json` 保存实际内层容量成绩、每次整策略选择与最终外层配置，`costs.json` 保存所有候选的机制改变数；实际嵌套划分和父协议保存在 `protocol.json`。
