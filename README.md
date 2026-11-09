# SPMTagClassifier

独立的 osu!mania **4K / 6K / 7K** 键型分析库。Node 与浏览器均可用，推理零第三方依赖。

- **7K 默认**：沿用原专家，修正高速规则段的 Break 门控和均匀节奏 BPM 显示，附带区间占用、分数分布及 anchor / release 几何证据。
- **4K**：原生五类模型，同一局部分布参与分段、证据加权及整谱组成。
- **6K**：在真实六列上运行 4K/7K 共同粗分类器，不再使用虚拟中键或投影专家输出混合。
- **7K `modern`**：完整联合选择后的 LN 小幅更新候选，显式启用；没有充分证据替换默认。

**[回炉研究结论与证据](research/rework/CONCLUSIONS.md)** · [完整研究记录](research/rework/README.md) · [上游来源](UPSTREAM.md)

已发布的 [v1.1.0](https://github.com/Ist1na07/SPMTagClassifier/releases/tag/v1.1.0) 是从插件 **tag v1.2.0** 提取的原版算法；v1.2.0 的分阶段调参完整性与部分归因结论已修正，历史 tag 不改写。本说明对应 overlay 修补版 v1.2.2。

### v1.2.2：先修正可确定的 overlay 错误

7K 默认及 `modern` 对近似均匀到源毫秒精度的打击行直接测量等效 BPM，避免旧网格把 300 BPM 显示成 200；原测值保存在 `segment.evidence.cadence.legacyEquivBpm`。Jack 保留多押行／同列重复的动作时钟，不把交错单键速度当 Jack 速度。不规则段保留原估计，`legacy` 路径不变。

默认 7K 另收紧“小押数＋规则节奏”的休息判据：高速段不再只因物量略低于整谱中位数就判 Break。57 种门控与 245 种聚合策略分别完整对照后，采用相对物量 0.85、实际行速率 16 rows/s，并保留原聚合参数。F01/F02 两条源歌曲留出的 Speed 回归主类覆盖由79.33%到100%；它们是主动选取的错例，不是总体准确率。

整谱17类 macro P/R/F1 从55.15%／51.35%／51.98%到55.07%／51.47%／52.02%，基本持平；原六张人工谱局部准确率69.59%到69.41%，留出 DS 有4.25秒休息误报。**Tech→Dense CS 尚未解决**，没有采用未改善该错例的新分类头。完整取舍、逐类错误和参数范围见 [本轮结论](research/overlay-quality/CONCLUSIONS.md)。实际插件渲染器离线检查已通过；实时游戏／tosu 链路尚未验收。

11月10日的[限时定点跟进](research/overlay-quality/targeted-sequence-r1/DECISION.md)参考GitHub上的连续行识别及逐手时序方法，完成原段／局部上下文的联合对照，但没有减少真实Tech→Dense错误；新头未采用，产品仍为v1.2.2。

## 运行一张谱面

```sh
npm install https://github.com/Ist1na07/SPMTagClassifier/releases/download/v1.2.2/spm-tag-classifier-1.2.2.tgz
```

```js
const fs = require('node:fs');
const { analyzeMap } = require('spm-tag-classifier');
const r = analyzeMap(fs.readFileSync('map.osu', 'utf8'), { speedRate: 1 });
console.log(r.overall.tags);
console.log(r.segments.map(s => ({
  start: s.start,
  end: s.end,
  tag: s.tag,
  support: s.distribution,
  weight: s.aggregationWeight,
  occupancy: s.evidence.occupancy,
  geometry: s.evidence.geometry,
})));
```

发布包只含运行所需文件；完整仓库另含研究证据。源码或发布包无需安装推理依赖即可运行：

```sh
node bin/classify.cjs map.osu
node bin/classify.cjs map.osu 1.5 modern
```

浏览器引入 `dist/tag-engine.js`，通过 `TagClassifier.analyzeMap(osuText, options)` 调用；不需要 Node 全局、运行时 `eval` 或动态编译。

### 替换现有 overlay 的分类引擎

插件 v1.2.0 自带一份 `tag_engine.js`；安装本库不会自动更新那份文件。先备份插件目录中的原文件，再将本库 `dist/tag-engine.js` 复制到同一位置并命名为 **`tag_engine.js`**，最后重新加载 overlay。无需替换难度引擎、设置或谱面文件；需要回退时恢复备份并重新加载。

当前已用原插件渲染器离线检查调用及显示，实时游戏／tosu 链路尚未验收。原插件只为7K调用分类器；替换引擎本身不会为插件界面增加4K/6K支持。

## 默认与研究候选

| 选项 | 7K | 4K / 6K |
|---|---|---|
| 默认或 `algorithm: 'stable'` | 原专家及 confidence / scores；修正 Break 门控、BPM 显示，附加新证据 | 原生粗分类 |
| `algorithm: 'modern'` | 保留原 RC/LN 专家先验，仅适配 Release 相对证据；LN 保护门控与聚合联调 | 同一新原生粗分类 |
| `algorithm: 'legacy'` | 未经包装的 v1.1.0 行为 | 原实现未验证这些键数 |

4K / 6K 的公开标签是 `jack / stamina / speed / tech / ln`。`stamina` 保留源数据的兼容名称，主要指 jump/handstream；内部形态名为 **chordstream**，可从 `r.coarse` 读取。6K 不再输出缺少真值验证的 7K 细分类。

4K / 6K 的 `overall.tags` 是**单一主导形态**，`sort` 是该形态的 family；连续 `familyMass` 另行保留 LN/RC 混合情况。7K 保留多标签 / Mix 印象口径。

## 分数、权重和时间怎么读

- `distribution` 是归一化类型支持，**不是识别正确率**。4K / 6K 的 `scores` 与它相同；结构充分条件可能输出单位支持，其来源不是经验概率校准。
- 稳定 7K 的 `confidence`、`scores` 保持原含义。新 7K 候选的 `confidence` 是显示标签的归一支持，旧 affinity 保存在 `evidence.legacyConfidence`。
- 4K / 6K 使用同一局部支持计算 `时长 × 相对物理负荷^a × 集中度^b`，集中度为 `Σp²`。a、b 与局部头一起拟合；最终两个模型均选到 a=1、b=1。`aggregationWeight` 给出实际加权贡献。
- 7K 候选保留其联调选中的原生后处理。当前 effort / confidence 指数均为 0；没有为了“使用置信度”而强制非零加权。具体参数在 `modelInfo.aggregationPolicy`。
- 4K / 6K 的 `overall.weights`、`scores`、`composition` 均是同一归一组成。7K 的 `weights` 保留加权秒数，`composition` 才是归一组成；候选的 normalized 模式以已知具体标签为条件，unknown 不被伪装成某个细类。
- `evidence` 包括熵、margin、负荷、占用、结构规则窗口数及特征截断比例。`contextNoteObservations` 是上下文观测次数，重叠上下文可能重复看到同一物件，不是去重物量。
- 新 `evidence.geometry.anchor` 描述单指持续 run、集中度及规律性；`geometry.release` 描述头部隔离、其他指持键、同手顺序等。它们是连续物理量，不是新增硬类别概率。旧 `descriptors.anchor` 等字段保留旧含义。

所有时间、LN 尾时间和窗口长度均为 **speedRate 换算后的毫秒**。`equivBpm` 为对应动作间隔的十六分音符等效速度：Speed／Tech／CS等使用行节奏，Jack优先使用多押行间隔、否则使用同列重复。不能用交错单键的全行速度替代Jack动作速度；不稳定或不适用的段落可以不提供。7K 候选的 `sourceStart/sourceEnd` 保留模型实际评分区间，显示首尾覆盖修复不改变原聚合。

## 当前证据支持什么

- **4K**：歌曲分组嵌套评价的五类 macro-F1 为 **95.56%**；桶名不是逐窗口真值。
- **共同粗层**：7K 可确定粗类整谱 macro-F1 为 **92.31%**；与 7K 细类指标不是同一口径。
- **6K**：2,343 张有效真实谱、180,009 窗口及 32 张镜像通过运行检查。没有人工 6K 键型真值，**没有六键准确率结论**。
- **7K 候选**：整谱 17 类 macro-F1 为 54.32%，细类分段按谱等权为 68.75%；原固定算法为 51.98% / 69.59%。差值区间不足以支持替换默认。

研究按歌曲及重复物理谱分组，每项消融重新选择其完整声明参数空间；弱整谱标签不复制成局部真值。归档 7K 先验的历史训练暴露未知。完整逐类、逐谱、联合配置、优化状态、正负结果及参数定型方法见 [研究结论](research/rework/CONCLUSIONS.md)。

## API 与缓存

`analyzeMap`、`extractIntermediate` 保留原入口，类型在 `src/index.d.ts`。

- `buildCachedMap` **保留原 v1.1.0 缓存 schema**，供原算法使用。
- `buildEvidenceMap` **保留 v1.2.0 的 40 项诊断缓存合同**，不是新 53 项粗层缓存。
- 当前 4K / 6K 的原生窗口在 `result.debug.windows`；特征顺序在 `result.modelInfo.featureKeys`。不要混用这三种表示。

非 legacy 路径仅接受 Mode 3、4/6/7K、有限正 speedRate。空谱、非法 LN 尾、未知键数会明确报错。非有限 timing point 被忽略并记录在 `map.warnings`。源六键集中的 `bm6k_0778` 有两个早于头 2 ms 的尾，被明确拒绝。

## 开发与复现

```sh
npm ci
npm run build
npm test
npm run test:types
```

构建会从验证过的共同源码生成静态引擎；生成文件无需手工维护。原版提取逐值检查另需 `.upstream/` 是插件 v1.2.0 checkout：`npm run test:parity`。

完整主研究需要本地 `dataset/`、Python 和 PowerShell 7，不会上传原谱；运行时间按小时计：

```powershell
python -m pip install -r research/rework/requirements.txt
./scripts/reproduce.ps1 -Workers 4 `
  -OutputRoot research/rework/results/reproduced `
  -FitRoot .cache/rework-reproduced-fits
```

使用新结果目录，程序会拒绝混合不同协议的续跑。已有有效输入经来源检查后复用；全数据定型只读实际内层留出预测，不从外层模型挑赢家。复现不覆盖库内模型；要明确采用新定型结果，再执行：

```sh
node scripts/export-research-models.cjs --root research/rework/results/reproduced
npm run build
```

早期 v1.2.0 的分阶段复现保留为 `./scripts/reproduce.ps1 -Legacy`，不再是当前主流程。其他原生分段、冷启动和容量研究的独立协议、命令与全部证据位于 `research/rework/`。

MIT · 原始上游署名与归档保留。
