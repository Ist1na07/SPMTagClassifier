# 保留成熟判断，用感知百分数修正

> 同一 v1.2 迭代的过程报告。`shipped_v122` 是中间对照，研究总基线为 v1.1.0；最终选型与统一比较见 [复核结论](../selection-review-r4/REPORT.md)。
原349条、原歌曲／物理源分组；新全局权重和门控完整联合选择。

| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |
|---|---:|---:|---:|---:|---:|---:|
|shipped_v122|55.07%|51.47%|52.02%|62.41%|65.46%|58.07%|
|all/protected|59.03%|53.99%|55.34%|68.09%|71.46%|63.83%|
|hard_time/protected|58.14%|52.63%|54.14%|66.10%|69.06%|61.82%|
|branch_time/protected|57.42%|52.78%|54.01%|66.10%|69.25%|61.82%|
|difficulty/protected|58.77%|53.39%|54.86%|67.49%|70.79%|63.54%|
|actions/protected|58.15%|52.93%|54.35%|66.45%|69.54%|62.10%|
|combined/protected|59.03%|53.99%|55.34%|68.09%|71.46%|63.83%|
|hybrid_False|56.61%|51.47%|52.58%|62.82%|65.50%|58.07%|
|hybrid_True|59.03%|53.99%|55.34%|68.09%|71.46%|63.83%|

共同评分候选（仅INNER）：
```json
{
  "representationId": 49,
  "familyId": 1,
  "policyId": 80,
  "representation": {
    "weightId": 24,
    "power": 1.0,
    "hard": false
  },
  "family": {
    "enabled": true,
    "minRc": 0.1,
    "minLn": 0.1
  },
  "policy": {
    "keepFloor": 0.25,
    "addFloor": 0.5,
    "mixOverride": 0.65
  },
  "metrics": {
    "macroPrecision": 0.5869407415565937,
    "macroRecall": 0.5391398340239358,
    "macroF1": 0.5487322767997934,
    "microF1": 0.6825065882072473,
    "sampleF1": 0.716317971963749,
    "exactMatch": 0.6380271422849435
  }
}
```

原有整谱输出是冻结先验，不是新的逐段标签。低占比剔除、高占比补充、强原型替换Mix、两大家族Hybrid判定都直接读取新积分百分数。全部关闭时精确回到已发布v1.2.2。

这批规则没有TRAIN数值拟合：每外折在其完整TRAIN上选全局参数，等价于汇总原3INNER验证库存；最终OUTER仍独立于本折选参。报告是重复使用原资料的回顾性检验。INNER保护不保证OUTER无退步。

mixOverride也可以替换旧Hybrid；没有新家族触发不单独取消旧Hybrid。此处澄清原协议一句过宽的描述，代码／结果没有改变。
