# 分离整谱Mix与局部成分：配对结果

同一349条整谱、原歌曲分组、回顾性嵌套比较；局部头仍冻结。

| 方案 | macro P | macro R | macro F1 | micro F1 | sample F1 | exact |
|---|---:|---:|---:|---:|---:|---:|
|shipped_v122|55.07%|51.47%|52.02%|62.41%|65.46%|58.07%|
|hard_time/balanced|55.16%|49.09%|50.87%|63.44%|66.29%|59.51%|
|branch_time/balanced|55.99%|49.28%|50.67%|63.58%|66.46%|57.64%|
|difficulty/balanced|52.41%|47.58%|49.02%|62.01%|64.54%|57.06%|
|actions/balanced|53.31%|47.91%|49.30%|62.28%|64.88%|56.34%|
|combined/balanced|54.50%|48.17%|50.02%|62.70%|65.36%|57.20%|
|components/balanced|51.94%|48.72%|49.68%|62.52%|64.89%|56.05%|
|all/balanced|50.81%|46.68%|47.61%|60.50%|63.31%|55.76%|
|no_rating/balanced|53.92%|47.99%|49.56%|62.07%|64.90%|58.21%|
|nonconditional/balanced|50.81%|46.68%|47.61%|60.50%|63.31%|55.76%|
|conditional/balanced|47.02%|45.53%|45.39%|57.43%|60.13%|50.00%|
|no_fallback/balanced|55.36%|47.43%|49.86%|62.46%|64.35%|56.92%|
|with_fallback/balanced|50.81%|46.68%|47.61%|60.50%|63.31%|55.76%|
|public_rating|54.50%|48.17%|50.02%|62.70%|65.36%|57.20%|
|component_rating|52.05%|48.98%|49.90%|62.70%|65.03%|56.05%|
|physical_only|53.31%|47.91%|49.30%|62.28%|64.88%|56.34%|

共同component_rating候选（按INNER选择，不是OUTER最优）：
```json
{
  "configId": 2271,
  "fitId": 1135,
  "config": {
    "arm": "components",
    "weightId": 19,
    "power": 2.0,
    "hard": false,
    "components": true,
    "ridge": 0.03,
    "balance": 0.5,
    "conditional": false,
    "fallback": true
  },
  "metrics": {
    "macroF1": 0.497792929975098,
    "macroRecall": 0.49086357869947833,
    "macroPrecision": 0.5288464239881766,
    "microF1": 0.6381865549271093,
    "sampleF1": 0.6650674394810855,
    "exactMatch": 0.5714336296553758
  }
}
```

完整保留不可支持目标为误判；拟合时其截断NLL常数与零梯度分离。详细数值与错误身份见analysis.json。物理／公开评分／组件三个消费入口分别调参，缺失组件不补零。反传与实际消费者验收仍待完成。
