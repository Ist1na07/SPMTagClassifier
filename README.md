# SPMTagClassifier

独立的 osu!mania **4K / 6K / 7K** 键型分析库；Node 与浏览器均可用，推理零第三方依赖。

- [v1.1.0](https://github.com/Ist1na07/SPMTagClassifier/releases/tag/v1.1.0)：从 SPM Map Analyser **tag v1.2.0** 提取的原版 7K 引擎，782 次逐值一致性检查通过。
- **v1.2.0**：独立特征与分类模块、分段证据、4K 粗分类、6K 迁移，以及可复现的完整调参/消融资料。
- **[研究报告与全部结果](research/REPORT.md)** · [缺陷审查](research/AUDIT.md) · [来源](UPSTREAM.md)

## 先运行一张谱面

```sh
npm install github:Ist1na07/SPMTagClassifier#v1.2.0
```

```js
const fs = require('node:fs');
const { analyzeMap } = require('spm-tag-classifier');
const r = analyzeMap(fs.readFileSync('map.osu', 'utf8'), { speedRate: 1 });
console.log(r.overall.tags);
console.log(r.segments.map(s => ({
  start: s.start, end: s.end, tag: s.tag,
  distribution: s.distribution ?? s.scores,
  occupancy: s.evidence.occupancy,
  entropy: s.evidence.normalizedEntropy,
})));
```

从源码或 Release 源码压缩包运行，无需先安装任何依赖：

```sh
node bin/classify.cjs map.osu
node bin/classify.cjs map.osu 1.5 modern
```

浏览器直接引入 `dist/tag-engine.js`，通过 `TagClassifier.analyzeMap(osuText, options)` 使用。原版归档在 `src/legacy/tag-engine.cjs`。

## 模式：稳定结果与研究候选分开

| 调用 | 7K | 4K / 6K |
|---|---|---|
| 默认，或 `algorithm: 'stable'` | 保留原版键型决定；修正时间线首尾覆盖，增加区间测量及分数分布 | 使用新模型 |
| `algorithm: 'modern'` | 完整新特征、新分段、新整谱研究头 | 使用新模型 |
| `algorithm: 'legacy'` | 返回未经包装的 v1.1.0 行为 | 原实现未在这些键数上验证，不推荐 |

**为什么 7K 没有强制替换？** 新候选没有胜过现有模型：分组测试 macro-F1 为 40.20%，原模型 47.72%；六谱分段评估为 48.03% 对 68.92%。原模型的历史训练样本未公开，这些数字不是双方完全相同训练暴露下的泛化结论。生产默认保持成熟决定，研究候选保留供后续改进。

4K 按验证集选中的模型，整谱五类分组测试 macro-F1 为 **94.87%**。6K 只有难度标注，没有人工键型真值；**83.57%** 是经过跨域歌曲排除后的 4K/7K→6K 投影代理测试 macro-F1，不是实际 6K 准确率。6K 粗分类的验证选择退化为 7K 支持；4K/7K 融合未获稳定收益，细分段落仍提供明确标为外推的混合分布。

## 输出怎么读

所有时间、窗口长度及 LN 尾时间均为 **经过 speedRate 换算后的毫秒**。等效 BPM 使用相邻行间隔对应的十六分音符速度；不稳定的窗口不输出该值。

- `segments`：起止、主键型、次候选、family、BPM、confidence、完整 scores、evidence。
- 稳定 7K 的 `confidence` 和 `scores` 保留旧含义；新增 `distribution` 是归一化 affinity，不是经验校准概率。`evidence.legacyConfidence` 同时保留原置信度。
- 新模型 `scores` 是局部 softmax 分布；`confidence` 是平滑后显示标签的支持。时间平滑可能使显示标签不再是当前最大分数。温度只在六谱验证范围内调节，不能当作对任意谱面的正确率保证。
- `evidence`：归一熵、前两名 margin、负荷、LN 占用面积比、特征窗口数。`contextNoteObservations` 是上下文观测计数，重叠窗口会重复看到同一物件，**不是去重物量**。
- `overall.tags`：整谱印象标签。新整谱模型使用独立多标签支持 `overall.scores`，它们不要求和为 1。
- 稳定 7K `overall.weights` 保留旧的加权时长口径，新增 `overall.composition` 才是归一组成；新模型 `overall.weights` 已经是归一组成。组成是证据摘要，不等于多标签预测概率。
- `descriptors`：anchor、trill、roll、repetition、occupancy、releaseIndependence 连续描述量，不伪装成已经有真值验证的新类别。
- `modelInfo`：实际算法、键数与监督级别。6K 的 `overall.transfer` 明示迁移权重及证据边界。

4K 与 6K 的整谱标签为 jack / ln / speed / stamina / tech。7K 使用原有细分类；6K 分段借用 7K 细类，但尚无真人细分标签验证。course 是整谱容器，不训练成局部键型。

## API 与错误

`analyzeMap`、`extractIntermediate` 保留原入口。`buildCachedMap` **始终保留原缓存 schema**，用于原版算法；新特征缓存使用 `buildEvidenceMap`。TypeScript 类型在 `src/index.d.ts`。

仅接受 Mode 3、4/6/7K、有限正 speedRate。空谱、非法 LN 尾、未知键数会明确报错。非有限 timing point 被忽略，并在新解析器 `map.warnings` 中记录；不修改原文件。全量数据中 `bm6k_0778` 有两个早于头 2 ms 的尾，被明确拒绝。

## 验证与复现

```sh
npm test
npm run build
node research/runtime-parity.cjs
```

前两项离线运行；第三项需要本地 dataset，检查 Python 拟合结果与实际 JS 推理。完整实验：

```powershell
python -m pip install -r research/requirements.txt
./scripts/reproduce.ps1
```

数据按当前工作目录的 `dataset/` 布局提供，未随仓库或 npm 包再分发。每个消融重新调参；固定歌曲组划分、全部候选、逐谱/逐时间结果及局限均在 `research/`。原版提取逐值检查另需 `.upstream/` 为插件 v1.2.0 checkout：`npm run test:parity`。

## English summary

Dependency-free standalone osu!mania classifier. Default 7K retains mature predictions with additive evidence; opt into the new research head with `{ algorithm: 'modern' }`. 4K uses five benchmark coarse classes. 6K is cross-key extrapolation, not human-validated pattern accuracy. Browser global: `TagClassifier`. See the research report for independently retuned ablations and negative findings.

MIT · original upstream attribution retained.
