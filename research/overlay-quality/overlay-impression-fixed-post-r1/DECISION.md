# 采用高速规则段门控，保留原聚合

采用共同INNER选择 ratio=0.85、rowRateMax=16，原聚合dominantShare=.52、mixShare=.07、secondRel=.75。专家系数、分段与其它休息规则不变。

两条源留出玩家Speed摘录覆盖79.33%→100%。整谱P/R/F1 55.15/51.35/51.98%→55.07/51.47/52.02%；原六谱69.59%→69.41%，DS留出4.2505秒休息误报。采用以明确错例修复为理由，不称整体大幅进步。完整245聚合对照F1较高但召回下降，故不用其聚合参数。

旧无查询监督选择、严格重调F1保护和macro-first对照均保留。新产品目标使用同57门控配置及来源TRAIN监督，没有把查询TRAIN符合度叫验证准确率。共同参数来自INNER平均表而非外折赢家。原协议文字中“original arm retunes245”指完整研究；本控制的originalPostControl=true与aggregationDomain明确只允许原聚合。

Tech→Dense仍3,012ms，未解决。条件范围、可识别性、原生及实际渲染验收见上层CONCLUSIONS.md。`analysis.json`中adopted=false是报告生成时状态；本文件记录随后完成运行验收后的部署决定，不改写当时协议或数值。
