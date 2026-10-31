# 7K 回炉：原双专家冷启动联合拟合结果

本阶段保留原 RC/LN 双专家、规则覆盖、wildcard 与 Break 结构，但只用各训练折从零估计标准化与两个专家权重。源风格逐段复制、等谱权重复制和弱整谱 bag 监督分别重训，全部专家配置与最终门控、规则混合、显示和聚合共同选择；不先按局部代理成绩淘汰模型。固定原边界，尚未检验冷启动专家与原生分段联调。

| 方案 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |
|---|---:|---:|---:|
| count_primary_original | 41.32% | 48.93% | 49.18% |
| count_primary_corrected | 44.16% | 49.43% | 49.32% |
| copy_partial_original | 44.85% | 48.84% | 48.80% |
| copy_partial_corrected | 43.54% | 48.20% | 47.67% |
| bag_partial_original | 42.16% | 50.24% | 50.40% |
| bag_partial_corrected | 40.85% | 48.03% | 48.48% |
| rules_original | 44.26% | 44.73% | 43.98% |
| rules_corrected | 39.86% | 44.93% | 44.30% |
| original_fixed | 51.98% | 69.59% | 68.92% |
| normalized_fixed | 51.68% | 69.59% | 68.92% |
| original_aggregation_retuned | 53.11% | 69.59% | 68.92% |
| normalized_aggregation_retuned | 52.13% | 69.59% | 68.92% |

整谱固定 17 类口径，重复物理文件总权重为 1。完整逐类、歌曲组平衡结果、成对组 bootstrap 区间、逐谱分段与 singleton 校准见 `metrics.json`。

每个方案/外层目录保存 `inner-models.json.gz`、`candidates.json.gz`、`search.json.gz`、`inner-predictions.npz`、先写入的 `selection.json` 及 `outer-model.json`。实际划分和源码快照见 `protocol.json`。全零规则输出按家族 wildcard 弃权，不任意采用字典首类；这项明确修正与归档纯规则路径有差别。

原模型固定预测和原系数先验的历史训练暴露未知；纯重拟合方案另列，不能把先验比较写成双方训练资料完全相同的泛化结论。
