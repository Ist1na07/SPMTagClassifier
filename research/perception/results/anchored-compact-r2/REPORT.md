# compact：扩大活跃边界后的保守百分数聚合

同一原349条／源歌曲组。每外折全新规则参数联合选择；局部专家未改变。

| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |
|---|---:|---:|---:|---:|---:|---:|
|shipped_v122|55.07%|51.47%|52.02%|62.41%|65.46%|58.07%|
|all/protected|59.70%|50.72%|53.39%|66.45%|68.34%|61.82%|
|hard_time/protected|57.77%|50.04%|52.40%|64.43%|66.90%|60.37%|
|branch_time/protected|56.48%|50.98%|52.47%|64.46%|67.67%|60.95%|
|difficulty/protected|58.49%|52.81%|54.31%|67.06%|70.12%|62.68%|
|actions/protected|58.16%|52.26%|53.79%|66.62%|69.84%|62.68%|
|combined/protected|59.70%|50.72%|53.39%|66.45%|68.34%|61.82%|
|hybrid_False|55.97%|51.21%|52.16%|62.22%|64.83%|57.20%|
|hybrid_True|59.70%|50.72%|53.39%|66.45%|68.34%|61.82%|

仅INNER选择的共同评分配置：
```json
{
  "representationId": 230,
  "familyId": 20,
  "policyId": 239,
  "representation": {
    "weightId": 114,
    "power": 2.0,
    "hard": false
  },
  "family": {
    "enabled": true,
    "minRc": 0.07,
    "minLn": 0.05
  },
  "policy": {
    "keepFloor": 0.25,
    "addFloor": 1.01,
    "mixOverride": 1.01
  },
  "metrics": {
    "macroPrecision": 0.6434055072463458,
    "macroRecall": 0.5214068014308958,
    "macroF1": 0.5605288365200457,
    "microF1": 0.6966787594383416,
    "sampleF1": 0.7062211100998583,
    "exactMatch": 0.640920246654591
  },
  "objectiveDeficitFromMaximum": 0.0,
  "nearOptimalTolerance": 0.001
}
```

扩大alpha/beta、时间窗与门控边界是同一模型的完整笛卡尔比较。近最优0.1个百分点内优先线性物理解释和较少修正，是明确先验，不是把平地伪装成数据井。条件剖面记录实际标签向量和百分数的变化；未通过保护的固定坐标明确标为不可行。

常规输入不需再次评分；公开评分输入复用同次SPMRating原始曲线。局部反传是独立对照，不能把此处整谱增益算成局部分段准确率提高。所有结果是同批反复研究资料上的回顾性检验。
