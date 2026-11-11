# 在 overlay 中显示整谱负担占比

本轮以 v1.1.0 为研究基线，全部成果归入 v1.2 系列。稳定 7K 使用多分支支持与连续时间负担积分决定整谱 tag。v1.2.4 增加逐手动作的假切换排除及同类内部稳定变速显示；其余局部专家、4K／6K 和 `modern` 路径不变。联合学习权重没有通过来源留出，未进入默认路径。

仓库未发布更新已采用原56维专家的联合重调 W/b，倍率仍为1.4；保持训练约定中的旧参考预测器作为积分聚合先验。新分段与分支支持由新 W/b 计算，并保留 LN Wildcard 的低贴合休息豁免。`aggregation:'classic'` 保留替换前局部与原聚合路径；已发布 v1.2.4 tag／资产不变。开发成绩、来源留出风险和已接受的退步见 [参数采用报告](../research/core_retune/REFIT_FINDINGS.md)。

## 已有 SPM Rating 结果时，直接复用

```js
const {analyzeMap, ratingEvidenceFromSPM} = require('spm-tag-classifier');
// spmResult 是同一全文、键数和 speedRate 的 whole/include_curve=True 结果。
const ratingEvidence = ratingEvidenceFromSPM(spmResult);
const result = analyzeMap(osuText, {speedRate: spmResult.rate, ratingEvidence});
console.log(result.overall.tags, result.overall.percentages);
```

分类器不启动 Python、不调用评分器、不重复评分。支持 SPMRating v1.2.0 的 Detailed／Compact 已发布模型身份。`query` 必须逐项等于当前谱面原生头部行时钟，`demand` 是未校准的原始需求场；不能传入整谱星级、400 ms 显示桶或平滑后的曲线。

直接提供的 `ratingEvidence` 包含 `schemaVersion:1`、`keyCount:7`、相同 `speedRate`、`clock:'post-rate-ms'`、`modelIdentity`、`query` 和 `demand`。错误模型、裁剪曲线或倍率会明确报错。所有时间都是倍率换算后的毫秒，保留小数。

不提供曲线时，使用单独联调的动作物量配置；`overall.perception.difficultyAvailable` 为 `false`。不会把缺失难度补成零。使用替换前局部专家及原整谱聚合策略（完整 v1.1.0 行为用 `algorithm:'legacy'`）：

```js
analyzeMap(osuText, {aggregation: 'classic'});
```

`algorithm:'legacy'` 仍是原 v1.1.0；`modern` 仍是原显式研究候选。只有稳定原生 7K 支持新聚合；对新功能显式传入不兼容模式会报错。

## 更新插件调用和显示

只替换 `tag_engine.js` 会得到动作版聚合，但不会自动接入评分或改变旧界面的“时间占比”。完整更新需要本包中的适配器。先关闭该 overlay 页面，在插件目录外运行：

```powershell
# 先检查支持的布局与完整评分参数，不写文件。
node path/to/package/bin/update-overlay.cjs 'E:/path/to/plugin'
# 确认输出是预期插件目录后应用。
node path/to/package/bin/update-overlay.cjs 'E:/path/to/plugin' --apply
```

适配器只修改 `index.html`、`spm_algorithm.js` 和 `tag_engine.js`；写入前把三份原文件及清单保存在插件内 `.spm-tag-backups/<时间>/`。设置、音乐和谱面不变。回退时关闭页面，从对应备份复制这三个文件到插件目录，再重新加载。首次更新后也需重新加载页面。旧 schema1 及误编号 v1.3.0 集成均可原位迁移并保留备份。npm 用户应显式安装 v1.2.4 的包，普通版本升级不会自动从 1.3.0 降版。

适配器针对 SPM Map Analyser v1.2.0 的已验证布局，拒绝未知参数和布局。同版本重复运行无变化。评分函数只附带同一次计算已产生的原始 `query/D`；评分数值不变。切换 Detailed／Compact 会同时更新整谱 tag；平滑和显示宽度不影响 tag。插件本来的 4K／6K 难度显示保留，不借此增加细分类界面。

实心 chip 是通过门控的整谱 tag，轮廓 chip 是尚未通过门控的次要分支。二者百分数均为负担份额。Hybrid 显示 RC／LN 家族份额，避免把家族并集的约 100% 误看成 Hybrid 的识别置信度。没有达到门控的 tag 时显示“主要键型未定”，不伪造主类，也不覆盖评分器原有的家族判断。

已检查真实插件渲染器的离线调用、430 px 布局、模型切换、平滑不变性和 DT 时钟；尚无实时游戏／tosu 验收。

## 百分数的确切含义

对具体键型 `c`：

```
P[c] = 100 × ∫ support[c,t] × burden[t] dt / ∫ burden[t] dt
burden[t] = log(1 + D[t])^alpha × log(1 + A[t])^beta
A[t] = (1000/windowMs × (窗口头数 + 并集尾数) + holdCredit × 持键列数) / 7
```

每个动作使用以其时间为中心的方窗。每列 LN 先取区间并集；重叠持键不重复计价。`D` 沿头部行时钟右连续，最后样本延拓到最后物理事件；延拓毫秒数单列在元数据中。积分在所有分支、曲线、动作窗和持键端点的并集上精确求和，而不是逐头平均。

`support` 来自未平滑的当前专家分支：非负分数除以 `max(1, 最大分数)`，端点上限 0.999；Break 为零。它表示原型贴合，保留一阶意义，不平方、不强迫各类和为一。困难但短的段落可以比时间占比更突出；Break 的物理负担仍保留在分母中。

纯 Rice 且两种拇指归属下都无手内指组变化时，排除不可能的 Dense/Fast CS 支持；分母仍用排除前原最大值，不把删除的支持分给其它类型。旧参考预测器的整谱判断继续作为固定先验，不随新局部参数一起移动。RC Wildcard 没有对应的整谱细分支，局部百分数使用其分类支持并明确标注来源，不假报为 0%。

最终动作版参数为 `alpha=0,beta=2,windowMs=1000,holdCredit=1`；Detailed 恢复 R1 的 `alpha=2,beta=1,windowMs=1000,holdCredit=1`；Compact 保留 `alpha=4,beta=1.5,windowMs=1000,holdCredit=1`。指数是在压缩后的量上联调得到；仍有活跃边界，不能视为已找出唯一最优参数。

- `overall.percentages`：14 个具体分支的 0–100 份额；允许重叠，总和可以超过 100。
- `overall.labelPercentages`：选中标签的份额。Mix／Hybrid 是对应家族的负担覆盖，不是虚构的局部混合类概率。
- `overall.familyMass`：RC／LN 路由的物理负担比例，Break 不进入这两个家族。
- `overall.weights`：未归一的支持加权负担积分，**不再是时间秒数**。
- `overall.composition`：为旧调用保留的类型组成：有支持时和为一，无支持时全零，不能当作新百分数。
- `segment.branches`：在可见合并段上按实际负担积分得到的多分支贴合度。
- `segment.aggregationWeight`：该可见段对全谱负担分母的比例。

整谱门控使用这些新份额保留／剔除旧 tag，必要时用强原型替换旧 Mix；两个具有实质负担的家族可触发 Hybrid，不再要求三个细类。具体门槛见 `modelInfo.perception`。

研究结论、完整消融、逐类错误和没有采用的方案见 [本轮研究](https://github.com/Ist1na07/SPMTagClassifier/blob/v1.2.3/research/perception/CONCLUSIONS.md)。
