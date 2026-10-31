"""Separate manual-chart exposure and source-style early stopping.

The weak corpus is FIXED in every condition and disjoint from all six manual
song groups. Compare training on all six manual charts with leave-one-manual
training, at one prespecified historical-style configuration. This diagnostic
does not select a release model or claim the archive's exact training batch.
"""
from family_train import *
from family import classify,smooth
from analyse import score_segments
import concurrent.futures,argparse,subprocess

CONFIG={'localL2':.0005,'strongWeight':1.,'balance':'cap6'}
SPEC={'recipe':'count_primary','representation':'original'}
_MAIN=_MANUAL=_META=_ROOT=None

def init(root):
    global _MAIN,_MANUAL,_META,_ROOT
    _MAIN,_MANUAL,_=load();_META=load_metadata();_ROOT=Path(root)

def source_gd(weak,strong,family):
    data=HeadData(weak,strong,_META,family,'original','count_primary',CONFIG)
    W=torch.nn.Parameter(torch.zeros(len(data.classes),data.x.shape[1]));b=torch.nn.Parameter(torch.zeros(len(data.classes)))
    for epoch in range(500):
        W.grad=b.grad=None;loss=data.objective(W,b,CONFIG);loss.backward()
        with torch.no_grad():W-=.08*W.grad;b-=.08*b.grad
    W.grad=b.grad=None;loss=data.objective(W,b,CONFIG);loss.backward()
    fitted={'classes':data.classes,'keys':SCHEMA['original'],'mean':data.mean.tolist(),'std':data.std.tolist(),
      'W':W.detach().numpy().ravel().tolist(),'b':b.detach().numpy().tolist()}
    trace={'policy':'prespecified source-style GD500/lr.08','epochs':500,'loss':float(loss.detach()),
      'gradientMax':max(float(W.grad.abs().max()),float(b.grad.abs().max())),**data.scope}
    return fitted,trace

def job(task):
    optimizer,held=task;strong=_MANUAL if held<0 else[r for i,r in enumerate(_MANUAL)if i!=held]
    path=_ROOT/'models'/(optimizer+'-'+('full'if held<0 else str(held))+'.json');train_uids=[r['uid']for r in _MAIN+strong]
    if path.exists():
        saved=json.loads(path.read_text(encoding='utf8'));models,trace=saved['models'],saved['fit']
        if saved['trainUids']!=train_uids:raise RuntimeError('Exposure diagnostic scope changed')
    else:
        models={};trace={}
        if optimizer=='converged':models,trace=fit_pair(_MAIN,strong,_META,SPEC,CONFIG)
        else:
            for family in ['rc','ln']:models[family],trace[family]=source_gd(_MAIN,strong,family)
        dump(path,{'models':models,'fit':trace,'trainUids':train_uids,'weakCorpusFixed':True,'config':CONFIG})
    evaluation=_MANUAL if held<0 else[_MANUAL[held]];rows=[]
    for r in evaluation:
        raw=classify(r,_META[r['uid']],models,'original',.45,'model')
        rows.extend({'uid':r['uid'],**s}for s in smooth(raw))
    return {'optimizer':optimizer,'held':held,'segments':rows,'fit':trace,'metrics':score_segments(evaluation,rows)}

def run(args):
    main,manual,_=load();root=Path(args.output)
    if {r['group']for r in main}&{r['group']for r in manual}:raise RuntimeError('Weak/manual song overlap')
    protocol={'id':'manual-exposure-early-stop-r1','config':CONFIG,'spec':SPEC,
      'sourceBaseCommit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'sourceSnapshot':{n:(Path(__file__).parent/n).read_text(encoding='utf8')for n in ['family_exposure.py','family.py','family_train.py']},
      'weakUidsFixedInAllConditions':[r['uid']for r in main],'manualUids':[r['uid']for r in manual],
      'scope':'retrospective diagnostic; fixed weak training corpus; full-manual resubstitution versus leave-one-manual accuracy; fixed original boundaries and historical-style config, no hyperparameter selection',
      'policies':['GD500/lr.08 at recorded source-style settings','same scalar objective optimized with the family convergence policy'],
      'archive':'nearby historical source recipe, not a verified reconstruction of the archived training batch'}
    p=root/'protocol.json'
    if p.exists()and json.loads(p.read_text(encoding='utf8'))!=protocol:raise RuntimeError('Exposure protocol changed; use a new revision')
    dump(p,protocol)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,initializer=init,initargs=(str(root),))as pool:
        results=list(pool.map(job,[(policy,held)for policy in ['gd500','converged']for held in range(-1,6)]))
    metrics={};segments=[]
    for policy in ['gd500','converged']:
        full=next(r for r in results if r['optimizer']==policy and r['held']==-1)
        loo=[s for r in results if r['optimizer']==policy and r['held']>=0 for s in r['segments']]
        metrics[policy]={'fullManualResubstitution':full['metrics'],'leaveOneManual':score_segments(manual,loo)}
        segments.extend({'policy':policy,'condition':'full',**s}for s in full['segments'])
        segments.extend({'policy':policy,'condition':'leave_one',**s}for s in loo)
    rng=np.random.default_rng(20261031);draw=rng.multinomial(6,np.full(6,1/6),size=2000)/6
    a=np.array([r['accuracy']for r in metrics['gd500']['leaveOneManual']['charts']]);b=np.array([r['accuracy']for r in metrics['converged']['leaveOneManual']['charts']])
    delta={'convergedMinusGd500':float((b-a).mean()),'pairedChart95':np.quantile(draw@(b-a),[.025,.975]).tolist()}
    dump(root/'metrics.json',{'scope':protocol['scope'],'metrics':metrics,'optimizerEffect':delta})
    dump(root/'segments.json',segments);dump(root/'fits.json',[{k:v for k,v in r.items()if k!='segments'}for r in results])
    lines=['# 固定弱训练集，分开人工谱暴露与训练停止策略','',
      '同一批弱整谱、同一原结构/边界及同一预定历史式配置。全六谱列是训练集重代入，留一列的该张人工谱未参与拟合；不以此诊断选择发布参数。',
      '', '| 优化策略 | 全六谱重代入 | 留一人工谱 |','|---|---:|---:|']
    for policy,m in metrics.items():lines.append(f"| {policy} | {m['fullManualResubstitution']['chartMeanAccuracy']:.2%} | {m['leaveOneManual']['chartMeanAccuracy']:.2%} |")
    lines+=['',f"留一时，充分优化减 GD500 的分段差值为 {100*delta['convergedMinusGd500']:+.2f} 个百分点，六谱成对区间为 [{100*delta['pairedChart95'][0]:+.2f}, {100*delta['pairedChart95'][1]:+.2f}]。",
      '', '附近历史源码采用 500 轮、学习率 0.08；归档精确训练批次仍未定位。结果解释当前数据上的暴露和优化策略效应，不将历史源码的相似性当作归档训练身份。完整模型、实际输出和训练梯度见本目录。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    dump(root/'complete.json',{'conditions':14,'headFits':28,'hyperparameterSelections':0})
    print('\n'.join(lines),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=3);p.add_argument('--output',default='research/rework/results/family-exposure-r1');run(p.parse_args())
