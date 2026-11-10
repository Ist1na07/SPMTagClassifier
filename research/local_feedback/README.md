# 分段反馈与实际读出续研

先读 [采用决定](CONCLUSIONS.md)，再按问题查 [过程证据](RESULTS.md)。本轮唯一研究基线为 v1.1.0；实验中的固定产品对照为 v1.2.3。v1.2.4 没有采用新的联合学习权重。

## 直接检查本次产品

```powershell
npm run build
npm test
npm run test:types
npm run test:parity
```

`test/segment-readout.test.cjs` 是自包含的逐手动作、负向亲和守恒、变速、倍率及旧长边界回归；不依赖用户标注包。实际插件仍先 dry run，再使用 `bin/update-overlay.cjs --apply`。本研究没有修改实时插件。

## 完整实验与参数

- `joint_activity.py`：路由＋类型的固定亲和预算联合目标；TRAIN 白化、类别对比及全部有效参数正则化。
- `joint_study.py`：冻结的 102 项完整配置、六乘三来源分组及选择规则。
- `joint_report.py`：先保存公共 INNER 选择，再消费 OUTER；分别报告原六图、八个真实局部、三条整谱消费路径和对照。
- `results/joint-r7-physical/`：正式物理候选约束研究。`protocol.json` 保存声明和源码快照；各分组 `inner-grid.json`／`selection.json`／`outer-*.json` 保存完整选择及预测；`oof.json` 是主报告。`product` 允许退回原产品，不把回退称作新头收益。
- `results/activity-r6/`：只学活动路由的前序完整负面实验。
- `results/static-cut-r1/`：参数自由的动作证书、六个规则／窗口消融和 H 对照。新产品不把 Anchor/Vibro 推作替代基础类。
- `product_readout_check.cjs`／`product_readout_report.py`：实际静态消费器的全411原生对照、原349标签集合与六图报告。`product-readout-display-baseline.json` 的349 UID集合与这些报告相同。

大体积结果、试算源码、原始拟合参数和两个起点的收敛记录在本 release 的 `spm-tag-classifier-1.2.4-research.zip`。解压到仓库根目录，保留目录结构。正式 R7 检查点位于 `.cache/local-feedback-joint-r7-physical/`，不要与早期未约束 R7 混用。每个正式拟合接收的原坐标梯度须不超过 `1e-5`。原始未完成／未约束试算单列，没有混入1,836个正式拟合。

拟合依赖本地原始数据及前序物理/评分缓存；这些不是 npm 包内容。保留当前项目 `dataset/` 和 `.cache/` 时可续跑：

```powershell
python -X utf8 research/local_feedback/joint_study.py --workers 6
python -X utf8 research/local_feedback/joint_report.py
node research/local_feedback/product_readout_check.cjs
python -X utf8 research/local_feedback/product_readout_report.py
```

`joint_study.py` 会校验冻结协议并复用有效检查点；更改模型或网格后不能冒用相同研究身份。新机器须先按前序 `research/perception/`、本目录的提取／准备脚本重建数据缓存；仅下载拟合检查点不能替代物理输入。主动补标、构造机制和历史接触数据各自保留原作用范围，不当作新的随机测试集。

## 研究消费器与采用代码分开

`activity_overlay_probe.cjs` 等动态编译仅用于隔离研究；生产是 `src/segment-readout.cjs`、维护构建器和 CSP 静态浏览器包。完成正式研究后，研究探针改用生产同义的内部 `_branchSupport` 通道；旧探针源码仍保存在 `report-protocol.json`。此兼容修复不改变训练或已保存的选择。

首349输入显示诊断包含未标注谱，不能与精确的349条标注 UID 指标混算。R7 型号、读出原型修订号都不是新增产品迭代。
