# 本轮外部方法审查：借鉴表示，不套用标签

本次只为限时 overlay 修复寻找新表示，不扩成外部算法排行榜。固定源码版本与文件清单另见 protocol.json 的 externalSources。下载的第三方程序未执行，也未直接复制进产品。

## 连续行语法：Interlude / YAVSRG

[原始识别器](https://github.com/YAVSRG/YAVSRG/blob/a7c78e77733595749e63da26b5300e8962a4e111/prelude/src/Calculator/Patterns/Patterns.fs)通过连续若干行的押数、重复键和空间关系匹配形态。这提示旧分类器的整段平均押数和切换比例会遗漏行排列；本轮只独立实现连续多押支持、单／多押转换两个轻量统计，没有移植整套语法或认定外部标签等于本项目标签。

## 移植和扩展：LeoBlackMT 与 XiL

[LeoBlackMT 行匹配](https://github.com/LeoBlackMT/osumania_map_analyser/blob/ccd9ac9c0a9367fa5fb5ab5e18b787f3fbe73326/ManiaMapAnalyser%20by%20Leo_Black/js/patterns/findPatterns.js)和[XiL 移植](https://github.com/Xiaollull/OSU-Mania_MapPatterns-Analyzer-XiL/blob/d65884484aef0dac24755d76b3c37168dc2fc076/patterns_analyzer/patterns/find_patterns.py)都沿用该方法家族，不能算三个独立模型的交叉验证。它们支持多种局部结构识别，但类别、聚合和本项目玩家标注口径不同，不能直接替换主类作为真值。

## 逐手时序：Etterna

[ChaosMod](https://github.com/etternagame/etterna/blob/2d706147cc4bcf95a8ceb24dc4f4129be207f803/src/Etterna/MinaCalc/Dependent/HD_PatternMods/Chaos.h)在逐手序列中考察相邻打击间隔关系。这启发本轮比较全局和逐手时钟：全局均匀并不保证每手动作均匀。新实现使用各手对数间隔标准差及间隔变化比例，不复制其权重、分段或阈值；变化判据1.35沿用本项目已有实现。Etterna 的技术难度指标也不是7K Tech标签，不能将其数值当分类真值。

## 本轮怎样验证这些启发

原专家与v1.2.2门控固定；原整段／局部2秒上下文分别比较bias、全局时序、逐手时序、连续行、组合表示。每项完整重调声明参数与原245聚合策略，按来源歌曲分组。仅有表示差异或构造例变化不够采用，还须改善真实显示错例并检查整谱P/R/F1及其它键型。
