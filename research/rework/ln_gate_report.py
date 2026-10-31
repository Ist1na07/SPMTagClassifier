"""Audit reused fits and distinguish gate effects from LN-head effects."""
from pathlib import Path
from collections import Counter
import argparse
from mil import np,json,load,encode,MAP_CLASSES
from analyse import main_run,encode_prediction,f1_counts
from ln_adapter_search import read,write,VARIANTS
from ln_adapter_report import definite_release
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def paired(parent_rows,current_rows,parent_segments,current_segments,weak,manual):
    gs=sorted({r['group']for r in weak});index={g:i for i,g in enumerate(gs)};gi=np.array([index[r['group']]for r in weak])
    y=encode(weak);w=np.array([r['recordWeight']for r in weak]);rng=np.random.default_rng(20261101)
    draw=rng.multinomial(len(gs),np.full(len(gs),1/len(gs)),size=2000);md=rng.multinomial(6,np.full(6,1/6),size=2000)/6
    values=[]
    for rows,segments in [(parent_rows,parent_segments),(current_rows,current_segments)]:
        byid={r['uid']:r for r in rows};assert len(rows)==len(byid)==len(weak)
        p=encode_prediction([byid[r['uid']]for r in weak]);blocks=[]
        for v in [y*p,(1-y)*p,y*(1-p)]:
            b=np.zeros((len(gs),len(MAP_CLASSES)));np.add.at(b,gi,v*w[:,None]);blocks.append(b)
        from analyse import score_segments
        acc=np.array([r['accuracy']for r in score_segments(manual,segments)['charts']])
        values.append((float(f1_counts(*(b.sum(0)for b in blocks))),f1_counts(*(draw@b for b in blocks)),float(acc.mean()),md@acc))
    a,b=values
    return {'mapDelta':b[0]-a[0],'mapDelta95':np.quantile(b[1]-a[1],[.025,.975]).tolist(),
      'segmentDelta':b[2]-a[2],'segmentDelta95':np.quantile(b[3]-a[3],[.025,.975]).tolist()}


def run(directory):
    root=Path(directory);complete=read(root/'complete.json');protocol=read(root/'protocol.json');parent=Path(complete['parentAllGateControl'])
    weak,manual,_=load();parent_rows=read(parent/'outer-predictions.json');parent_segments=read(parent/'outer-segments.json')
    source_fits={};inner_status=Counter();outer_status=Counter();unaccepted=[];gradients=[];new_outer=set();outer_sources=set();changes={};release={};metrics={}
    for mode in complete['modes']:
        main_run(root/mode);metrics[mode]=read(root/mode/'metrics.json');rows=read(root/mode/'outer-predictions.json');segments=read(root/mode/'outer-segments.json');changes[mode]={};release[mode]={}
        for variant in VARIANTS:
            old=[r for r in parent_rows if r['variant']==variant];cur=[r for r in rows if r['variant']==variant]
            old_s=[r for r in parent_segments if r['variant']==variant];cur_s=[r for r in segments if r['variant']==variant]
            changes[mode][variant]=paired(old,cur,old_s,cur_s,weak,manual);release[mode][variant]=definite_release(manual,cur_s)
            for o,p in enumerate(protocol['plans']):
                directory=root/mode/variant/str(o)
                for r in read(directory/'inner-expert-reuse.json.gz'):
                    key=(variant,o,r['inner'],r['configId']);trace=r['fit']
                    if key in source_fits:
                        assert source_fits[key]==r
                        continue
                    source_fits[key]=r;inner_status[trace['status']]+=1
                    allowed={weak[i]['uid']for i in p['inner'][r['inner']]['train']}|{manual[i]['uid']for i in p['inner'][r['inner']]['strongTrain']}
                    assert set(trace['trainUids'])<=allowed
                    gradient=trace['checks'][-1]['gradientMax'];gradients.append(gradient)
                    if trace['status']!='small_gradient'or gradient>1e-5:unaccepted.append(key)
                profile=read(directory/'outer-model.json');trace=profile['fit'];outer_status[trace['status']]+=1
                allowed={weak[i]['uid']for i in p['train']}|{manual[i]['uid']for i in p['strongTrain']};assert set(profile['trainUids'])==allowed
                if trace['status']!='frozen_archived_prior':
                    assert set(trace['trainUids'])<=allowed;gradient=trace['checks'][-1]['gradientMax'];gradients.append(gradient)
                    if trace['status']!='small_gradient'or gradient>1e-5:unaccepted.append((mode,variant,o))
                    outer_sources.add(profile['expertFitSource'])
                    if not profile['expertFitReused']:new_outer.add(profile['expertFitSource'])
    assert len(source_fits)==complete['uniqueInnerExpertFitsReused']
    audit={'uniqueInnerExpertFitsReused':len(source_fits),'newInnerFits':0,'innerStatuses':dict(inner_status),'outerSelectionStatuses':dict(outer_status),
      'uniqueOuterExpertSources':len(outer_sources),'newUniqueOuterExpertFits':len(new_outer),'maxGradient':max(gradients),'unaccepted':unaccepted,'accepted':not unaccepted}
    write(root/'optimizer-audit.json',audit);write(root/'gate-effects.json',changes);write(root/'definite-release.json',release)
    lines=['# LN 头与额外负荷门控的独立效应','',
      '固定原始自动边界和旧版Break/家族判定。仅改变研究流程新增的相对头部负荷门控：ln_protected不再用该门控删除已经路由为LN的窗口，off则完全移除此新增门控。原始all门控对照来自同系数、同划分的完整父研究。',
      '', '预定主方案geometry3/ln_protected；不在外层挑表示或门控赢家。每项方案都用完整30项拟合配置及精确零头，与共同后处理集合重新联合选择；off投影至零门控后去重为488项，ln_protected保留512项。',
      '', '| 门控 | 表示 | 整谱macro-F1 | 按谱分段 | Release召回 | Release精确率 |','|---|---|---:|---:|---:|---:|']
    for mode in complete['modes']:
        for variant in VARIANTS:
            m=metrics[mode];r=release[mode][variant];precision='—'if r['precision']is None else f"{r['precision']:.2%}"
            lines.append(f"| {mode} | {variant} | {m['map'][variant]['macroF1']:.2%} | {m['segment'][variant]['chartMeanAccuracy']:.2%} | {r['recall']:.2%} | {precision} |")
    fmt=lambda ci:f'[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]'
    lines+=['','## 门控修订与同表示all门控对照的成对差','', '| 门控 | 表示 | Δ整谱百分点［95%］ | Δ分段百分点［95%］ |','|---|---|---:|---:|']
    for mode,variants in changes.items():
        for v,d in variants.items():lines.append(f"| {mode} | {v} | {100*d['mapDelta']:+.2f} {fmt(d['mapDelta95'])} | {100*d['segmentDelta']:+.2f} {fmt(d['segmentDelta95'])} |")
    lines+=['','## 小幅更新与同门控零更新对照','']
    for mode in complete['modes']:
        m=metrics[mode];a=m['map']['geometry3'];b=m['segment']['geometry3']
        lines.append(f"- {mode}：主更新相对no_update的整谱差 {100*(a['macroF1']-m['map']['no_update']['macroF1']):+.2f}pp {fmt(a['pairedGroupBootstrapDeltaVsNoUpdate95'])}；分段差 {100*(b['chartMeanAccuracy']-m['segment']['no_update']['chartMeanAccuracy']):+.2f}pp {fmt(b['pairedChartBootstrapDeltaVsNoUpdate95'])}。")
    lines+=['',f"实际复用 {len(source_fits):,} 项不同内层专家拟合，新增内层拟合0；新外层专家拟合 {len(new_outer)} 项。全部源训练作用域及所选外层作用域核对，优化状态见 `optimizer-audit.json`。重复消费相同专家不算新训练证据。",'',
      'Release为确定单主标签评价，阳性只有两个谱来源；门控修正后的完整任务指标、误报、逐谱和成对区间必须一起看。归档先验的历史训练暴露仍未知。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    plot(root,changes)
    print(audit,flush=True)


def plot(root,changes):
    fig,axes=plt.subplots(1,2,figsize=(12,7),sharey=True);colors={'ln_protected':'#126878','off':'#b06335'}
    for j,v in enumerate(VARIANTS):
        for mode,offset in [('ln_protected',-.12),('off',.12)]:
            d=changes[mode][v]
            for ax,key,title in zip(axes,['map','segment'],['Whole-chart macro-F1','Segment accuracy (equal-chart)']):
                ci=np.array(d[key+'Delta95'])*100;ax.hlines(j+offset,*ci,color=colors[mode],alpha=.55,lw=2)
                ax.plot(d[key+'Delta']*100,j+offset,'o',color=colors[mode],label=mode if j==0 else None);ax.set_title(title,loc='left')
    for ax in axes:ax.axvline(0,color='#555',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_xlabel('Difference from matched all-gate control (pp)');ax.spines[['top','right','left']].set_visible(False)
    axes[0].set_yticks(range(len(VARIANTS)),VARIANTS);axes[0].invert_yaxis()
    handles,labels=axes[1].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,loc='upper right',bbox_to_anchor=(.98,.97),ncol=2)
    fig.suptitle('Extra head-burden gate: all parameters reselected',x=.03,ha='left',weight='bold')
    fig.text(.03,.025,'Paired 95% bootstrap: song groups / six annotated charts. Reused unchanged auxiliary experts; historical prior exposure unknown.',fontsize=9)
    fig.subplots_adjust(left=.19,right=.98,top=.87,bottom=.14,wspace=.2);fig.savefig(root/'gate-effects.png',dpi=160,facecolor='white');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='research/rework/results/ln-extra-gate-r1');run(p.parse_args().directory)
