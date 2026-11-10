# 直接用感知百分数门控整谱tag

> 同一 v1.2 迭代的过程报告。`shipped_v122` 是中间对照，研究总基线为 v1.1.0；最终选型与统一比较见 [复核结论](../selection-review-r4/REPORT.md)。
原349条标注、6×3歌曲分组，全部新权重／门控／收缩参数联合选择。

| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |
|---|---:|---:|---:|---:|---:|---:|
|shipped_v122|55.07%|51.47%|52.02%|62.41%|65.46%|58.07%|
|all/balanced|56.59%|50.42%|51.69%|64.92%|67.09%|57.49%|
|hard_time/balanced|57.61%|51.03%|52.73%|65.18%|67.71%|58.65%|
|branch_time/balanced|56.77%|48.45%|51.15%|63.75%|64.12%|54.61%|
|difficulty/balanced|56.36%|50.12%|52.27%|64.73%|66.23%|56.34%|
|actions/balanced|55.19%|50.07%|51.41%|63.67%|65.71%|55.76%|
|combined/balanced|56.61%|49.59%|52.12%|64.12%|65.42%|55.76%|
|hybrid_False|50.20%|42.55%|44.32%|54.29%|53.93%|47.69%|
|hybrid_True|56.59%|50.42%|51.69%|64.92%|67.09%|57.49%|

仅INNER选择的共同评分候选：
```json
{
  "representationId": 54,
  "policyId": 20,
  "shrinkId": 7,
  "representation": {
    "weightId": 26,
    "power": 2.0,
    "hard": false
  },
  "policy": {
    "hybrid": true,
    "minRc": 0.1,
    "minLn": 0.1,
    "mixFloor": 0.05,
    "dominant": 0.35
  },
  "shrink": {
    "reference": 0.25,
    "strength": 0.1
  },
  "metrics": {
    "macroPrecision": 0.577258015147568,
    "macroRecall": 0.5071649145794533,
    "macroF1": 0.5299965998248265,
    "microF1": 0.6576134912222575,
    "sampleF1": 0.6702101520394614,
    "exactMatch": 0.5661149543617904
  }
}
```

每个TRAIN门限对应一个实际分区区间。reference即使在strength=0也会选取平地区间中的不同代表，故可能改变未见谱预测；没有把正则化选出的点叫作数据识别的唯一井。条件剖面同时记录标签向量与百分数变化。

局部支持在此对照中冻结；这不是局部分类改进。SPM组件与局部反传由单独匹配实验检验。全部指标均为同批反复研究资料上的回顾性对照。
