"""Score complete coarse outer predictions, source by source; no refitting."""
import argparse
from coarse_data import load_headers,CLASSES
from coarse_model import FINE
from coarse_search import read,write,source_metric
from mil import np,Path,json,Counter,defaultdict
from sklearn.metrics import classification_report,accuracy_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def macro(tp,fp,fn):return np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp,dtype=float),where=(2*tp+fp+fn)>0).mean(-1)

def run(directory):
    root=Path(directory);complete=read(root/'complete.json');protocol=read(root/'protocol.json')
    weak,manual,_=load_headers();rows=read(root/'outer-predictions.json');segments=read(root/'outer-segments.json');byid={r['uid']:r for r in weak}
    metrics={};groups={};counts={};boots={};rng=np.random.default_rng(20261031)
    for K in[4,7]:groups[K]=sorted({r['group']for r in weak if r['keys']==K})
    draws={K:rng.multinomial(len(gs),np.full(len(gs),1/len(gs)),size=2000)for K,gs in groups.items()}
    manual_draw=rng.multinomial(6,np.full(6,1/6),size=2000)/6
    for name in complete['variants']:
        selected=[r for r in rows if r['variant']==name];predicted={r['uid']:r for r in selected}
        if len(selected)!=len(weak)or set(predicted)!=set(byid):raise RuntimeError('Incomplete/repeated coarse outer predictions: '+name)
        metrics[name]={};boots[name]={}
        for K in[4,7]:
            data=[r for r in weak if r['keys']==K];labels=[FINE.index(predicted[r['uid']]['predicted'])if predicted[r['uid']]['predicted']in FINE else 5 for r in data]
            y=np.array([FINE.index(r['coarseTags'][0])for r in data]);p=np.array(labels);w=np.array([r['recordWeight']for r in data])
            truth=np.eye(5)[y];output=np.zeros_like(truth);active=p<5;output[np.flatnonzero(active),p[active]]=1.
            index={g:i for i,g in enumerate(groups[K])};gi=np.array([index[r['group']]for r in data]);block=[]
            for x in[truth*output,(1-truth)*output,truth*(1-output)]:
                b=np.zeros((len(groups[K]),5));np.add.at(b,gi,x*w[:,None]);block.append(b)
            boot=macro(*(draws[K]@a for a in block));boots[name][str(K)]=boot
            group_mass=np.bincount(gi,weights=w);gw=w/group_mass[gi]
            group_score=macro(*[(a*gw[:,None]).sum(0)for a in[truth*output,(1-truth)*output,truth*(1-output)]])
            result=source_metric(data,labels,K)
            result.update(macroF1_95=np.quantile(boot,[.025,.975]).tolist(),songBalancedMacroF1=float(group_score),groups=len(groups[K]),
              perClass=classification_report(y,p,labels=list(range(5)),target_names=FINE,sample_weight=w,output_dict=True,zero_division=0))
            metrics[name][str(K)]=result
        bychart=defaultdict(list)
        for s in segments:
            if s['variant']==name:bychart[s['uid']].append(s)
        sm=[]
        for r in manual:
            pred=sorted(bychart[r['uid']],key=lambda s:s['start']);correct=total=0.;confusion=defaultdict(float)
            for a in r['annotations']:
                if not a['tags']:continue
                total+=a['end']-a['start']
                for s in pred:
                    overlap=max(0,min(a['end'],s['end'])-max(a['start'],s['start']))
                    if s['tag']in a['tags']:correct+=overlap
                    else:confusion['|'.join(a['tags'])+' -> '+s['tag']]+=overlap
            sm.append({'uid':r['uid'],'accuracy':correct/max(1,total),'correctMs':correct,'labelledMs':total,
                       'largestErrors':sorted(confusion.items(),key=lambda x:-x[1])[:8]})
        acc=np.array([s['accuracy']for s in sm]);boot=manual_draw@acc;boots[name]['segment']=boot
        metrics[name]['segment']={'chartMeanAccuracy':float(acc.mean()),'durationAccuracy':sum(s['correctMs']for s in sm)/sum(s['labelledMs']for s in sm),
          'chartMeanAccuracy95':np.quantile(boot,[.025,.975]).tolist(),'charts':sm}
    reference='none'if'none'in metrics else'weak4_only'
    delta_field='pairedDeltaVsNone95'if reference=='none'else'pairedDeltaVsWeak4Only95'
    if reference in metrics:
        for name in metrics:
            for K in['4','7','segment']:metrics[name][K][delta_field]=np.quantile(boots[name][K]-boots[reference][K],[.025,.975]).tolist()
    # Audit actual optimizer exits; successful process completion is not an
    # optimizer criterion. Keep inner/outer accounting separate.
    statuses=Counter();maxgradient=0.;exhausted=[];pool_bounds=Counter();inner_fits=0
    support='uniqueInnerExpertFitsReused'in complete
    sources=[]
    if support:
        for p in root.glob('inner/*/*/expert-reuse.json'):
            for row in read(p):sources.append((Path(row['source']),None))
        if len(sources)!=len({str(p)for p,_ in sources}):raise RuntimeError('Repeated expert reuse identity')
    else:sources=[(p,rows)for p in root.glob('*/*/*/fits.json.gz')for rows in [read(p)]]
    for p,rows_ in sources:
        for row in ([read(p)]if rows_ is None else rows_):
            t=row['fit'];inner_fits+=1;statuses[t['status']]+=1;g=t['checks'][-1]['projectedGradientMax'];maxgradient=max(maxgradient,g)
            if t['status']=='budget_exhausted':exhausted.append({'file':str(p),'configId':row['configId'],'gradient':g})
            for k,v in row.get('pool',row.get('model',{}).get('pool',{})).items():
                if v<=1e-9 or v>=1-1e-9:pool_bounds[k+(':lower'if v<.5 else':upper')]+=1
    expected=complete['uniqueInnerExpertFitsReused']if support else complete['innerFits']
    if inner_fits!=expected:raise RuntimeError('Coarse complete-marker fit count differs from actual traces')
    outer_statuses=Counter();outer_unaccepted=[];outer_maxgradient=0.;outer_models=0
    for p in root.glob('*/*/outer-model.json'):
        row=read(p);trace=row['fit'];outer_models+=1;outer_statuses[trace['status']]+=1
        gradient=trace['checks'][-1]['projectedGradientMax'];outer_maxgradient=max(outer_maxgradient,gradient)
        if trace['status'] not in ['small_projected_gradient','train_prediction_stable']:
            outer_unaccepted.append({'file':str(p),'status':trace['status'],'gradient':gradient})
    if outer_models!=complete['outerSelections']:raise RuntimeError('Outer model trace count differs from complete marker')
    audit={'innerFits':inner_fits,'statuses':dict(statuses),'maxFinalProjectedGradient':maxgradient,'budgetExhausted':exhausted,'poolBoundCounts':dict(pool_bounds),
      'outerSelections':outer_models,'outerStatuses':dict(outer_statuses),'outerMaxFinalProjectedGradient':outer_maxgradient,
      'outerUnaccepted':outer_unaccepted,'accepted':not exhausted and not outer_unaccepted,
      'outerCountMeaning':'one trace per selected artifact; identical source/configuration fits may be reused, not independent training runs'}
    if support:
        outer_traces=read(root/'outer-fits.json')
        new_outer=sum(not r['reused']for r in outer_traces)
        if new_outer!=complete['newOuterExpertFits']:raise RuntimeError('New outer fit count differs')
        audit.update(innerFits=0,uniqueInnerExpertFitsReused=inner_fits,innerConfigurationEvaluations=complete['innerConfigurationEvaluations'],newOuterExpertFits=new_outer,
          innerCountMeaning='all inner raw-expert fits reused; their actual source traces are audited, not counted as newly executed fits')
    write(root/'metrics.json',{'scope':protocol['scope'],'mapLabelMeaning':'weak source bucket or determinate whole impression; not precise per-window truth',
      'metrics':metrics,'optimizerAudit':audit,'sixKeyAccuracy':None})
    lines=['# '+('固定专家加物理充分条件：完整外层评价'if support else'共用粗层：完整外层评价'),'',
      '4K 与 7K 分别按固定五类口径评分，分段采用六张 7K 人工谱的可接受粗类集合。所有数字来自外层实际预测；4K 桶不能作为逐段真值，6K 没有真人键型准确率。',
      '', '| 方案 | 4K macro-F1 | 7K粗层 macro-F1 | 分段按谱等权 |','|---|---:|---:|---:|']
    for n,m in metrics.items():lines.append(f"| {n} | {m['4']['macroF1']:.2%} | {m['7']['macroF1']:.2%} | {m['segment']['chartMeanAccuracy']:.2%} |")
    lines+=['',f"内层审计 {inner_fits:,} 份配置/折拟合轨迹；优化退出：`{dict(statuses)}`。这不是新增拟合次数或统计样本数；有效旧拟合可复用。完整逐类、歌曲等权、成对区间、逐谱分段和优化状态见 `metrics.json`。",
      '', '局部头与两个聚合指数使用同一训练目标联合拟合，离散配置在内层最终目标上交叉选择。`weak4_only` 只用 4K 弱训练及 4K 内层目标（包括同分时的选择）；`weak4_only_shared_selection` 复用这些拟合，但用共享目标另选配置。二者在同一外层、同一配置下不重复训练。',
      '', '完整配置和记录划分见 `protocol.json`；各方案/外层保存内层全配置评分、先写入的选择、外层模型和逐谱/逐段输出。内层每折保存全部实际硬预测和训练轨迹；完整逐次模型的断点缓存留在本地，可由同一数据与源码重建。']
    lines+=['',f"外层 {outer_models} 项选择的模型轨迹：`{dict(outer_statuses)}`；同一实际拟合被两个选择口径复用时不算独立训练。连续训练预测稳定是声明的停止条件，不等于驻点证明。"]
    if support:
        lines[lines.index(next(s for s in lines if s.startswith('内层审计')))]=f"复用 {inner_fits:,} 份内层专家拟合轨迹，重新计算 {complete['innerConfigurationEvaluations']:,} 项最终输出配置评价；新增内层拟合为0，新增外层专家拟合为 {new_outer}。复用训练轨迹退出状态：`{dict(statuses)}`。"
        lines=[s for s in lines if not s.startswith('局部头与两个聚合指数')]
        lines+=['','`both` 是预先声明的主方案，`none` 是相同训练专家、相同配置搜索空间的对照；`jack`/`ln` 为充分条件消融。每种方案都从完整专家配置集合重新做内层选择，不沿用原配置。充分条件本身不拟合参数；专家仍以原始辅助训练目标联合拟合局部头和聚合指数，不冒称在硬模板目标上重新训练。',
          '',f"本轮内层选择口径为 `{protocol['criterion']}`。`4k` 只使用4K指标及4K同分处理，7K整谱和人工分段仅作外推评价。成对区间以 `none` 为参照，见 `metrics.json` 中 `{delta_field}`。",
          '', '99个定义级反例和跨语言一致性检查只验证充分条件的含义与实现，不是玩家标注的准确率。']
    if not audit['accepted']:lines+=['','**存在未达到停止条件的内层或外层拟合，当前评价仅为暂定诊断；必须处理上述优化状态再作最终模型选择。**']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    names=list(metrics);y=np.arange(len(names));fig,axes=plt.subplots(1,3,figsize=(14,max(4,.35*len(names)+2)),sharey=True)
    for ax,key,field,ci,title in zip(axes,['4','7','segment'],['macroF1','macroF1','chartMeanAccuracy'],['macroF1_95','macroF1_95','chartMeanAccuracy95'],['4K five-class macro-F1','7K coarse macro-F1','7K segment accuracy']):
        for j,n in enumerate(names):
            v=metrics[n][key];lo,hi=np.array(v[ci])*100;ax.hlines(j,lo,hi,color='#a1b5ba',lw=2);ax.plot(v[field]*100,j,'o',color='#126878')
        ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.15);ax.set_xlabel('Percent; paired-group/chart bootstrap 95%');ax.spines[['top','right','left']].set_visible(False)
    axes[0].set_yticks(y,names);axes[0].invert_yaxis();fig.suptitle(('Fixed raw expert + sufficient physical support'if support else'Native-key shared morphology: actual outer predictions')+(' | PROVISIONAL optimizer'if not audit['accepted']else''),x=.03,ha='left',fontsize=14,fontweight='bold')
    fig.text(.03,.025,'Retrospective cross-key song groups; 6K has no human pattern labels. Full optimizer audit saved beside this figure.',fontsize=9,color='#47555a')
    fig.subplots_adjust(left=.24,right=.98,top=.82,bottom=.18,wspace=.2);fig.savefig(root/'outer-results.png',dpi=160,facecolor='white');plt.close(fig)
    print('\n'.join(lines[:10]),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');run(p.parse_args().directory)
