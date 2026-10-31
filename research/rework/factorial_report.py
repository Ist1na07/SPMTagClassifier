"""Paired 2x2 followup from four COMPLETE retuned cells; no fitting."""
from analyse import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run():
    parent=Path('research/rework/results/native-joint-r1');follow=Path('research/rework/results/native-followup-r1');out=follow/'factorial'
    cells={'base':parent/'logical_neutral','release':parent/'logical_release','scale':follow/'factorial_scale','both':parent/'full'}
    main,manual,_=load();true=encode(main);weight=np.array([r['recordWeight']for r in main])
    groups=sorted({r['group']for r in main});index={g:i for i,g in enumerate(groups)};gi=np.array([index[r['group']]for r in main])
    metrics={};counts={};segments={};choices={}
    for name,path in cells.items():
        if not(path/'complete.json').exists():raise RuntimeError('Factorial cell incomplete: '+name)
        rows=json.loads((path/'outer-predictions.json').read_text(encoding='utf8'));byid={r['uid']:r for r in rows}
        if len(rows)!=len(main)or set(byid)!={r['uid']for r in main}:raise RuntimeError('Factorial population differs')
        p=encode_prediction([byid[r['uid']]for r in main]);metrics[name]=hard_measures(true,p,weight)
        counts[name]=[]
        for x in [true*p,(1-true)*p,true*(1-p)]:
            a=np.zeros((len(groups),len(MAP_CLASSES)));np.add.at(a,gi,x*weight[:,None]);counts[name].append(a)
        segments[name]=score_segments(manual,json.loads((path/'outer-segments.json').read_text(encoding='utf8')))
        choices[name]=json.loads((path/'selection.json').read_text(encoding='utf8'))
    rng=np.random.default_rng(20261031);draw=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=2000)
    md=rng.multinomial(6,np.full(6,1/6),size=2000)/6
    mb={n:f1_counts(*(draw@a for a in c))for n,c in counts.items()}
    sb={n:md@np.array([a['accuracy']for a in s['charts']])for n,s in segments.items()}
    definitions={'release_minus_base':{'release':1,'base':-1},'scale_minus_base':{'scale':1,'base':-1},
      'both_minus_base':{'both':1,'base':-1},'scale_given_release':{'both':1,'release':-1},
      'release_given_scale':{'both':1,'scale':-1},'interaction':{'both':1,'release':-1,'scale':-1,'base':1}}
    effects={}
    for name,coeff in definitions.items():
        effects[name]={}
        for key,actual,boot in [('map',{n:m['macroF1']for n,m in metrics.items()},mb),
                                ('segment',{n:s['chartMeanAccuracy']for n,s in segments.items()},sb)]:
            point=sum(c*actual[n]for n,c in coeff.items());samples=sum(c*boot[n]for n,c in coeff.items())
            effects[name][key]={'delta':point,'paired95':np.quantile(samples,[.025,.975]).tolist()}
    out.mkdir(parents=True,exist_ok=True)
    dump(out/'metrics.json',{'scope':'four completed retuned factorial cells; fixed historical experts; retrospective followup, unknown original exposure',
      'cellSources':{n:str(p)for n,p in cells.items()},'map':metrics,'segment':segments,'effects':effects,'selections':choices})
    lines=['# 分段信号归一化与 Release 替换：四格重调比较','',
      '每格均在同一六折歌曲组/精标留一协议下，重新选择完整原生分段、门控、显示与聚合。Release 的 20/40/80 ms 只在内层选。三格复用已有有效结果，补做唯一缺失的“只改分段信号归一化”格。',
      '', '| 条件 | 整谱 macro-F1 | 分段按谱等权 | 分段按时长 |','|---|---:|---:|---:|']
    labels={'base':'逻辑修正＋中性手＋覆盖','release':'再加 Release 替换','scale':'再改分段信号归一化','both':'两项都加'}
    for n in cells:lines.append(f"| {labels[n]} | {metrics[n]['macroF1']:.2%} | {segments[n]['chartMeanAccuracy']:.2%} | {segments[n]['durationAccuracy']:.2%} |")
    lines+=['','| 成对差值 | 整谱百分点［95%区间］ | 分段百分点［95%区间］ |','|---|---:|---:|']
    for n,effect in effects.items():
        def fmt(v):return f"{100*v['delta']:+.2f} [{100*v['paired95'][0]:+.2f}, {100*v['paired95'][1]:+.2f}]"
        lines.append(f"| {n} | {fmt(effect['map'])} | {fmt(effect['segment'])} |")
    lines+=['','归一化修改作用于变化点使用的分段信号，不直接替换固定边界上的 56 项分类向量；输出差异因此包含边界改变及重新选择的门控/聚合。Release 替换改变一个 LN 表示的含义，冻结原系数的比较不能代替重新训练该表示。',
      '', '这是两个方案经重调后的差值，不是把其他阈值锁死的单一微分效应。交互为“两项都加−只加释放−只改归一化＋基础”；整谱区间按歌曲组、分段区间按六张人工谱成对抽样。完整逐谱、混淆与所选配置见 `metrics.json`。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    names=list(effects);fig,axes=plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax,key,title in zip(axes,['map','segment'],['Whole-chart macro-F1','Segment accuracy (equal-chart)']):
        for j,n in enumerate(names):
            v=effects[n][key];lo,hi=np.array(v['paired95'])*100
            ax.hlines(j,lo,hi,color='#a1b5ba',lw=2);ax.plot(v['delta']*100,j,'o',color='#126878')
        ax.axvline(0,color='#25363c',ls='--',lw=1);ax.set_title(title,loc='left');ax.grid(axis='x',alpha=.15)
        ax.set_xlabel('Retuned difference (percentage points)');ax.spines[['top','right','left']].set_visible(False)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis()
    fig.suptitle('Release replacement x segment-signal normalization',x=.03,ha='left',fontweight='bold',fontsize=14)
    fig.text(.03,.025,'Four complete cells. Paired 95% bootstrap. Frozen historical experts; retrospective followup.',fontsize=9,color='#47555a')
    fig.subplots_adjust(left=.19,right=.98,top=.83,bottom=.16,wspace=.14)
    fig.savefig(out/'paired-factorial.png',dpi=160,facecolor='white');plt.close(fig)
    print('\n'.join(lines[:13]),flush=True)

if __name__=='__main__':run()
