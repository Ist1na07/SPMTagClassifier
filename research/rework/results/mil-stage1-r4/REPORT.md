# 7K 回炉：固定原分段的联合拟合结果

本阶段固定原自动边界，比较完整表示、先验适配容量、强弱监督及聚合。结果用于决定下一步；分段器与释放尺度尚未联调。

| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |
|---|---:|---:|---:|
| original56 | 40.98% | 56.62% | 57.75% |
| original56_context | 36.33% | 52.98% | 53.00% |
| modern40_context | 38.60% | 54.77% | 54.11% |
| corrected56_context | 37.01% | 56.97% | 56.33% |
| union_context | 41.38% | 56.17% | 55.60% |
| prior_residual | 45.25% | 66.27% | 65.13% |
| frozen_prior | 48.81% | 69.59% | 68.92% |
| strong_only_prior | 42.63% | 68.45% | 67.06% |
| copied_prior | 47.50% | 61.86% | 61.52% |
| context_residual | 46.58% | 67.62% | 66.52% |
| bias_only_prior | 48.92% | 67.98% | 67.27% |
| original_fixed | 51.98% | 69.59% | 68.92% |
| normalized_fixed | 51.68% | 69.59% | 68.92% |
| original_aggregation_retuned | 53.11% | 69.59% | 68.92% |
| normalized_aggregation_retuned | 52.13% | 69.59% | 68.92% |

整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。

模型配置由各外层的内层预测联合选择；完整配置与分数见 `search.json`，实际训练/验证/测试记录见 `protocol.json` 和 `supports.json`。

原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。

## 优化终止审计

所有内层候选及外层拟合都达到预先声明的训练梯度或连续训练目标/预测稳定条件。

| 拟合范围 | 主阶段实际状态 | 独立局部阶段实际状态 |
|---|---|---|
| inner | small_gradient_reused: 1209, train_prediction_stable: 3078, small_gradient: 1617 | small_gradient: 1152 |
| outer | train_prediction_stable: 36, small_gradient: 30 | small_gradient: 12 |

连续预测/目标稳定不等于梯度已趋近于零。逐项未接受记录与实际梯度上界见 `optimizer-audit.json`。

本轮只续训未达条件的内层候选，再重新联合选择；外层新拟合 0，相同配置及划分复用 66。
