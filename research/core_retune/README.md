# 原专家核心重调：先诊断表达能力，再做来源隔离选参

本次实验已完成，先读 [采用结论与完整比较](CONCLUSIONS.md)。采用 LN Wildcard 的低贴合休息误判修复；未采用新拟合权重。仓库构建含修复，已发布 v1.2.4 不变，未另发 release。

研究基线仍为 v1.1.0；当前产品对照为 v1.2.4，不新增产品版本。

## 第一关：沿实际执行路径审计

`extract.cjs` 重建既有 411 输入的自然区间、原专家 56 维向量和两个家族的规则／模型分数。逐项核对原始分类与已有输入；人工边界不进入特征提取。原始归档、现有协议和研究缓存不改写。

`audit.cjs` 在当前 model 模式下，对全部 RC/LN 默认规则参数作单参数扰动，同时记录规则分数、最终混合分数、原始主类及平滑输出的变化。没有变化分为：代码上不在有效路径、输入饱和／未激活、改变了支持但未跨主类门槛。不把单参数无变化当作无用特征的证明。

后续只搜索真实有效的原模型参数。先做标注训练范围的表达能力检查，再用既有歌曲来源分组完成选参。训练拟合不计作验证成绩。分段改变必须经过实际主类／平滑及固定三条整谱消费路径，不以替代 loss 的下降作采用证据。

## 比较顺序

1. 原专家核心参数直接重调，先固定聚合；优先 Tech／Dense CS／Speed。
2. 若训练错例仍不可分，定位表示或公式冲突后作最小机制修改。
3. 仅有希望的候选进入支持强度约束 × 局部／局部加整谱监督对照；不自动启动全部网格。
4. 报告逐类 P/R/F1、原六图与真实来源局部、重点混淆、整谱三路径、显示及开销。历史数据不称盲测。

执行：`node research/core_retune/extract.cjs`，随后 `node research/core_retune/audit.cjs`。输入保存在 `.cache/core-retune-parents-r1.jsonl.gz`；结果位于本目录 `results/`。

## 复核与续跑入口

- `prepare_gate.cjs`、`extract_context.cjs`、`extract_rc_context.cjs`：保留原生分段、早晚休息状态及短物理上下文，人工边界不进入提取。
- `feasibility.py`、`context_feasibility.py`：TRAIN 表达能力诊断，不能作采用成绩。
- `local_study.py`、`rc_context_study.py`：完整局部来源隔离对照；旧协议内有当时的源码快照。
- `native_parameter_study.py`、`wildcard_gate_study.py`：原生参数联调与休息机制完整消融；原臂有效指标复用，不重复拟合。
- `joint_study.py`：原核心四臂对照；`joint-source-r1/execution-repair.json` 保留稳定算术执行修复，`numeric-reuse-check.json` 验证旧检查点。源码更改时拒绝冒用旧协议；复现已完成的旧流程应使用协议中的源码快照。
- `joint_oof_report.py`：完整349池化指标。`joint-source-r1/oof.json` 的整谱字段为 OUTER 逐折均值，不能代替 `oof-full349.json`。
- `verify_candidate.cjs`、`product_report.py`：实际当前默认构建、有效先验和三条评分消费路径的411重放；不覆盖旧产品证据。
- `native_check.cjs`：非零原 W/b 模型与真实原生分类器的数值及决策一致性。
- `test_core.py`：研究合同、导出、来源隔离、指标计数、梯度及原生参数可达性检查。

本地重放：

```powershell
npm run build
npm test
npm run test:types
npm run test:parity
python -X utf8 -m unittest discover -s research/core_retune -p test_core.py
node research/core_retune/verify_candidate.cjs
python -X utf8 research/core_retune/product_report.py
```

需要原始 `dataset/`、只读 `.upstream/` 和先前提取的物理／评分缓存；不把数据集、音频或评分重算封装进产品。所有声明的网格均已完成，不需要为了复核再做一轮大搜索。可选 sample-F1 的旧中间字段已作废，正确最终值见重放报告；其它指标和选模保持。
