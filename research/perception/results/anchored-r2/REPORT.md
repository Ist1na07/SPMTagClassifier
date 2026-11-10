# detail：扩大活跃边界后的保守百分数聚合

同一原349条／源歌曲组。每外折全新规则参数联合选择；局部专家未改变。

| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |
|---|---:|---:|---:|---:|---:|---:|
|shipped_v122|55.07%|51.47%|52.02%|62.41%|65.46%|58.07%|
|all/protected|59.56%|51.20%|53.57%|66.62%|68.87%|62.39%|
|hard_time/protected|57.77%|50.04%|52.40%|64.43%|66.90%|60.37%|
|branch_time/protected|56.48%|50.98%|52.47%|64.46%|67.67%|60.95%|
|difficulty/protected|58.67%|53.06%|54.47%|67.32%|70.31%|62.68%|
|actions/protected|58.16%|52.26%|53.79%|66.62%|69.84%|62.68%|
|combined/protected|59.56%|51.20%|53.57%|66.62%|68.87%|62.39%|
|hybrid_False|55.97%|51.21%|52.16%|62.22%|64.83%|57.20%|
|hybrid_True|59.56%|51.20%|53.57%|66.62%|68.87%|62.39%|

仅INNER选择的共同评分配置：
```json
{
  "representationId": 233,
  "familyId": 20,
  "policyId": 235,
  "representation": {
    "weightId": 116,
    "power": 1.0,
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
    "mixOverride": 0.5
  },
  "metrics": {
    "macroPrecision": 0.5987941540980115,
    "macroRecall": 0.5441933479199339,
    "macroF1": 0.5537619010152247,
    "microF1": 0.6920395433696765,
    "sampleF1": 0.7235240191949911,
    "exactMatch": 0.6437805596597027
  },
  "objectiveDeficitFromMaximum": 0.0002175444957935868,
  "nearOptimalTolerance": 0.001
}
```

扩大alpha/beta、时间窗与门控边界是同一模型的完整笛卡尔比较。近最优0.1个百分点内优先线性物理解释和较少修正，是明确先验，不是把平地伪装成数据井。条件剖面记录实际标签向量和百分数的变化；未通过保护的固定坐标明确标为不可行。

常规输入不需再次评分；公开评分输入复用同次SPMRating原始曲线。局部反传是独立对照，不能把此处整谱增益算成局部分段准确率提高。所有结果是同批反复研究资料上的回顾性检验。
