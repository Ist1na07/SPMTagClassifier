"""Run every declared raw-model ablation through all four support treatments."""
import argparse,concurrent.futures,subprocess
from types import SimpleNamespace
from coarse_support_search import evaluate,MODES
from coarse_search import read,write
from coarse_model import SPECS,Path
from coarse_report import run as report


def job(arguments):
    variant,output,fits,outer=arguments;directory=Path(output)/variant
    if not(directory/'complete.json').exists():
        evaluate(SimpleNamespace(variant=variant,output=str(directory),fits=fits,outer_fits=outer))
    report(directory)
    return variant


def run(args):
    root=Path(args.output);variants=list(SPECS)
    protocol={'id':'coarse-supported-all-r1','primary':{'expert':'full53','support':'both'},
      'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'scope':'all16 predeclared raw-expert variants, each complete configuration grid reselected independently for all4 fixed support treatments; no outer variant winner',
      'variants':variants,'modes':MODES,'sourceFits':args.fits,
      'sourceSnapshot':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['coarse_support_grid.py','coarse_support_search.py','coarse_supported.py','physical-support.cjs','coarse-runtime.cjs','coarse_model.py','coarse_data.py']}}
    if(root/'protocol.json').exists()and read(root/'protocol.json')!=protocol:raise RuntimeError('Full supported study protocol changed')
    write(root/'protocol.json',protocol)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers)as pool:
        for variant in pool.map(job,[(v,str(root),args.fits,args.outer_fits)for v in variants]):print('support variant complete',variant,flush=True)
    metrics={v:read(root/v/'metrics.json')for v in variants};counts=[read(root/v/'complete.json')for v in variants]
    if not all(m['optimizerAudit']['accepted']for m in metrics.values()):raise RuntimeError('Unaccepted source fit')
    write(root/'metrics.json',metrics)
    write(root/'complete.json',{'variants':variants,'modes':MODES,'uniqueInnerExpertFitsReused':sum(c['uniqueInnerExpertFitsReused']for c in counts),
      'innerConfigurationEvaluations':sum(c['innerConfigurationEvaluations']for c in counts),'newOuterExpertFits':sum(c['newOuterExpertFits']for c in counts),
      'outerSelections':sum(c['outerSelections']for c in counts),'primary':protocol['primary']})
    lines=['# 完整共享网格：物理充分条件消融','',protocol['scope'],'',
      '| 专家 | 模板 | 4K macro-F1 | 7K粗层macro-F1 | 分段按谱等权 |','|---|---|---:|---:|---:|']
    for v,report_ in metrics.items():
        for mode,m in report_['metrics'].items():lines.append(f"| {v} | {mode} | {m['4']['macroF1']:.2%} | {m['7']['macroF1']:.2%} | {m['segment']['chartMeanAccuracy']:.2%} |")
    lines+=['','主方案固定为 `full53/both`，不在这64项外层结果中挑赢家。各专家目录包含全配置内层选择、逐谱预测、逐类指标、成对区间、图与源拟合状态审计。',
      '', '所有内层专家系数均复用未改变的原训练目标；支持模板后的最终输出完全重新评价、选参。新增外层专家拟合数量见 `complete.json`。准备矩阵仅在实际有效输入逐值相同时复用，不近似预测或跳过配置。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',default='research/rework/results/coarse-supported-all-r1');p.add_argument('--fits',default='.cache/rework-coarse-shared-r3-fits')
    p.add_argument('--outer-fits',default='.cache/rework-coarse-supported-all-r1-outer');run(p.parse_args())
