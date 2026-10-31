# 完整共享网格：物理充分条件消融

all16 predeclared raw-expert variants, each complete configuration grid reselected independently for all4 fixed support treatments; no outer variant winner

| 专家 | 模板 | 4K macro-F1 | 7K粗层macro-F1 | 分段按谱等权 |
|---|---|---:|---:|---:|
| base40 | none | 94.60% | 89.18% | 62.15% |
| base40 | jack | 94.60% | 89.18% | 62.15% |
| base40 | ln | 94.60% | 89.18% | 62.15% |
| base40 | both | 94.60% | 89.18% | 62.15% |
| sequence44 | none | 94.83% | 92.89% | 61.24% |
| sequence44 | jack | 94.83% | 92.89% | 61.24% |
| sequence44 | ln | 94.83% | 92.89% | 61.24% |
| sequence44 | both | 94.83% | 92.89% | 61.24% |
| full53 | none | 94.69% | 92.31% | 62.41% |
| full53 | jack | 94.69% | 92.31% | 62.41% |
| full53 | ln | 94.69% | 92.31% | 62.41% |
| full53 | both | 94.69% | 92.31% | 62.41% |
| no_geometry | none | 93.09% | 87.16% | 61.46% |
| no_geometry | jack | 93.09% | 87.16% | 61.46% |
| no_geometry | ln | 93.09% | 87.16% | 61.46% |
| no_geometry | both | 93.09% | 87.16% | 61.46% |
| no_rhythm | none | 92.38% | 90.12% | 59.93% |
| no_rhythm | jack | 92.38% | 90.12% | 59.93% |
| no_rhythm | ln | 92.38% | 90.12% | 59.94% |
| no_rhythm | both | 92.38% | 90.12% | 59.94% |
| no_holds | none | 94.11% | 87.82% | 63.15% |
| no_holds | jack | 94.11% | 87.82% | 63.15% |
| no_holds | ln | 94.11% | 87.82% | 63.15% |
| no_holds | both | 94.11% | 87.82% | 63.15% |
| no_sequence | none | 95.01% | 92.20% | 62.52% |
| no_sequence | jack | 95.01% | 92.20% | 62.52% |
| no_sequence | ln | 95.01% | 92.20% | 62.53% |
| no_sequence | both | 95.01% | 92.20% | 62.53% |
| no_release | none | 94.83% | 92.89% | 61.24% |
| no_release | jack | 94.83% | 92.89% | 61.24% |
| no_release | ln | 94.83% | 92.89% | 61.24% |
| no_release | both | 94.83% | 92.89% | 61.24% |
| no_burden | none | 94.43% | 92.67% | 62.10% |
| no_burden | jack | 94.43% | 92.67% | 62.10% |
| no_burden | ln | 94.43% | 92.67% | 62.10% |
| no_burden | both | 94.43% | 92.67% | 62.10% |
| copied | none | 95.24% | 87.96% | 63.00% |
| copied | jack | 95.24% | 87.96% | 63.00% |
| copied | ln | 95.24% | 87.96% | 63.00% |
| copied | both | 95.24% | 87.96% | 63.00% |
| no_class_balance | none | 95.53% | 88.45% | 65.37% |
| no_class_balance | jack | 95.53% | 88.45% | 65.37% |
| no_class_balance | ln | 95.53% | 88.45% | 65.37% |
| no_class_balance | both | 95.53% | 88.45% | 65.37% |
| no_effort | none | 94.36% | 89.72% | 63.80% |
| no_effort | jack | 94.36% | 89.72% | 63.80% |
| no_effort | ln | 94.23% | 89.72% | 63.80% |
| no_effort | both | 94.23% | 89.72% | 63.80% |
| no_confidence | none | 94.47% | 90.81% | 62.90% |
| no_confidence | jack | 94.47% | 90.81% | 62.90% |
| no_confidence | ln | 94.47% | 90.81% | 62.90% |
| no_confidence | both | 94.47% | 90.81% | 62.90% |
| weak4_only | none | 95.42% | 79.98% | 53.73% |
| weak4_only | jack | 95.56% | 79.53% | 52.56% |
| weak4_only | ln | 95.42% | 79.98% | 53.74% |
| weak4_only | both | 95.56% | 79.53% | 52.57% |
| weak4_manual7 | none | 94.58% | 89.26% | 64.41% |
| weak4_manual7 | jack | 94.58% | 89.26% | 64.41% |
| weak4_manual7 | ln | 94.58% | 89.26% | 64.43% |
| weak4_manual7 | both | 94.58% | 89.26% | 64.43% |
| weak7_manual7 | none | 81.41% | 89.42% | 62.17% |
| weak7_manual7 | jack | 81.41% | 89.42% | 62.17% |
| weak7_manual7 | ln | 81.41% | 89.42% | 62.18% |
| weak7_manual7 | both | 81.41% | 89.42% | 62.18% |

主方案固定为 `full53/both`，不在这64项外层结果中挑赢家。各专家目录包含全配置内层选择、逐谱预测、逐类指标、成对区间、图与源拟合状态审计。

所有内层专家系数均复用未改变的原训练目标；支持模板后的最终输出完全重新评价、选参。新增外层专家拟合数量见 `complete.json`。准备矩阵仅在实际有效输入逐值相同时复用，不近似预测或跳过配置。
