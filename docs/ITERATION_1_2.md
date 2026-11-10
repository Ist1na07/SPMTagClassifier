# 一轮研究：v1.1.0 基线 → v1.2 系列

本轮任务从原插件分类器的独立提取开始，到 4K／6K 泛化、7K 机制研究、overlay 修补和整谱聚合，**全部属于同一个 v1.2 迭代**。发行次数、实验次数与研究迭代次数不是一回事。

| 标识 | 实际含义 |
|---|---|
| v1.1.0 | 原始提取算法，本轮唯一研究基线 |
| v1.2.0／v1.2.1／v1.2.2 | 同轮开发过程中已发布的检查点 |
| 曾发布的 v1.3.0 | 执行中的错误编号，归入本轮；保留 tag／资产用于历史核对，不再推荐 |
| v1.2.3 | 纠正编号、基线与 Detailed 选型的检查点 |
| v1.2.4 | 当前交付；逐手假切换／稳定变速修正，联合分段学习未采用 |
| 实验 R1／R2／R3／R4 | 实验修订号，不是产品版本或新研究迭代 |

## 基线、内部先验和过程对照必须分开

原349条7K整谱，以相同标签、权重和类顺序重新运行未修改的 v1.1.0，P／R／macro-F1／完全匹配率为 **55.15%／51.35%／51.98%／57.78%**。

聚合实验复用 v1.2.2 的局部专家与旧整谱判断作为冻结先验；一些历史协议把这个过程对照写在 `baseline` 或 `shipped_v122` 字段中。字段的实际计算不改写，**它们不再被称为本轮研究基线**。新汇总同时列出原始基线和过程对照。

4K／6K 是相对原始7K提取版新增的能力；不能虚构原版不存在的原生6K准确率比较。已有4K、6K、7K分段研究的适用范围与局限分别保留。

## 成果与证据入口

- [完整主研究与4K／6K泛化](https://github.com/Ist1na07/SPMTagClassifier/blob/v1.2.3/research/rework/CONCLUSIONS.md)
- [overlay 的 Break／BPM 修补](https://github.com/Ist1na07/SPMTagClassifier/blob/v1.2.3/research/overlay-quality/CONCLUSIONS.md)
- [整谱聚合、消融与失败方案](https://github.com/Ist1na07/SPMTagClassifier/blob/v1.2.3/research/perception/CONCLUSIONS.md)
- [R1／R3 同口径复核与选型纠正](https://github.com/Ist1na07/SPMTagClassifier/blob/v1.2.3/research/perception/results/selection-review-r4/REPORT.md)
- [评分复用与插件更新](perception.md)
- [分段续研、反传对照及窄范围采用决定](../research/local_feedback/CONCLUSIONS.md)

已发布的历史代码和实验结果保留；整理的是当前编号、推荐入口和解释口径，不把历史快照伪装成未曾发布。
