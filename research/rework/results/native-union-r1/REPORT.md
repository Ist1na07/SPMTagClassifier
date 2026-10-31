# 7K：嵌套选择流程更新容量

保持父实验全部固定预测不变。先合并全部非历史候选，逐项限制从同一并集筛选，不让控制组另抽一套候选。在每个外层的内层训练子集上，各容量预算同时选择整条策略；其内层验证输出再选择容量，最后在完整外层训练组上重选策略。主候选为 pool_repaired_nested，其余限制各自重调；固定最小/不限制容量只作机制对照。

| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |
|---|---:|---:|---:|
| pool_repaired_nested | 52.67% | 62.13% | 60.83% |
| pool_repaired_unrestricted | 53.54% | 64.69% | 63.31% |
| pool_minimum_control | 51.58% | 66.81% | 66.32% |
| pool_no_release_nested | 53.31% | 67.86% | 66.00% |
| pool_no_release_unrestricted | 53.79% | 67.85% | 65.98% |
| pool_old_gain_nested | 52.67% | 62.13% | 60.83% |
| pool_old_gain_unrestricted | 53.69% | 64.70% | 63.32% |
| pool_full_scale_nested | 52.84% | 65.44% | 63.97% |
| pool_full_scale_unrestricted | 53.96% | 66.01% | 64.63% |
| pool_normalized_nested | 52.67% | 62.13% | 60.83% |
| pool_normalized_unrestricted | 53.54% | 64.69% | 63.31% |
| pool_soft_nested | 51.53% | 63.97% | 63.86% |
| pool_soft_unrestricted | 51.79% | 66.54% | 66.35% |
| pool_no_effort_nested | 52.22% | 63.98% | 63.87% |
| pool_no_effort_unrestricted | 53.72% | 65.15% | 64.07% |
| pool_no_confidence_nested | 53.39% | 67.14% | 66.55% |
| pool_no_confidence_unrestricted | 53.83% | 65.15% | 64.07% |
| pool_no_diversity_nested | 52.67% | 62.13% | 60.83% |
| pool_no_diversity_unrestricted | 53.54% | 64.69% | 63.31% |
| pool_logical_control_nested | 53.24% | 67.86% | 66.00% |
| pool_logical_control_unrestricted | 53.94% | 67.86% | 66.00% |
| original_fixed | 51.98% | 69.59% | 68.92% |
| normalized_fixed | 51.68% | 69.59% | 68.92% |
| original_aggregation_retuned | 53.11% | 69.59% | 68.92% |
| normalized_aggregation_retuned | 52.13% | 69.59% | 68.92% |

整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。


原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。

全部联合候选见根目录 `candidates.json.gz`，位打包的完整预测及来源见 `candidate-predictions.npz` / `candidate-origins.npz`；类序与编码、重复逐值核对见 `pool-audit.json`。所有实际训练选择作用域的分数在 `scope-scores/`，每项限制的候选索引与内层容量/策略选择在其目录。根目录 `costs.json` 保存机制改变数。原始父预测及原生分段缓存见 `../native-metric-r1/`，实际划分、源参数与生产源码见 `protocol.json`。
