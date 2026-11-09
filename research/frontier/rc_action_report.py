"""Multimetric whole/local and concept acceptance for the restricted RC test."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from rc_action_model import World,CONCEPTS,mil,read
from rc_action_runtime import policy_predictions,POST
from whole_reference import write
from landscape_audit import multilabel_metrics,paired_group_intervals


def unpack(z,head):
    out=[]
    for order,weights,total,family,sort in zip(z['order'][head],z['weights'][head],z['total'][head],z['familyCount'][head],z['sort'][head]):
        names=[mil.MAP_CLASSES[i]for i in order if i>=0];v=[weights[i]for i in order if i>=0]
        out.append({'ranked':names,'shares':np.array(v)/total if total else np.zeros(len(v)),'weights':dict(zip(names,v)),'total':total,'familyCount':family,'sort':['rc','ln','hb'][sort]})
    return out


def run():
    root=Path('research/frontier/results/rc-action-r1');protocol=read(root/'protocol.json');complete=read(root/'complete.json');native=read(root/'native-runtime-audit.json');world=World();main=world.main;uids=[r['uid']for r in main];y=mil.encode(main);w=np.array([r['recordWeight']for r in main]);groups=[r['group']for r in main]
    if uids!=[r['uid']for r in protocol['records']]:raise ValueError('RC report inventory differs')
    rows=read(root/'outer-predictions.json');local=read(root/'outer-local.json');concept=read(root/'concept-predictions.json');metrics={};hard={};local_metrics={};concepts={};profiles={};count=0;starts=0;maximum_gradient=0.;maximum_start_spread=0.
    references={'original_fixed':[r for r in read('research/rework/results/aggregation/predictions.json')if r['variant']=='original_fixed'],
        'v1.2.1_LN_candidate':[r for r in read('research/rework/results/ln-extra-gate-r1/ln_protected/outer-predictions.json')if r['variant']=='geometry3']}
    variants=list(references)+[kind+'/'+v for kind in protocol['kinds']for v in ['full','no_concept','no_strong','adaptive','frozen']]
    for name in variants:
        if name in references:cases=references[name]
        else:
            kind,v=name.split('/');cases=[r for r in rows if r['kind']==kind and r['variant']==v]
        lookup={r['uid']:r for r in cases}
        if len(lookup)!=len(cases)or set(lookup)!=set(uids):raise ValueError('Incomplete RC whole outputs')
        q=np.array([[c in lookup[uid]['predicted']for c in mil.MAP_CLASSES]for uid in uids],int);hard[name]=q;metrics[name]=multilabel_metrics(y,q,w,mil.MAP_CLASSES,groups)
        if name in references:continue
        samples=[r for r in local if r['kind']==kind and r['variant']==v]
        if len(samples)!=6 or len({r['uid']for r in samples})!=6:raise ValueError('Manual source folds differ')
        accuracy=np.array([r['rawPrimaryIntervalAccuracy']for r in sorted(samples,key=lambda r:r['uid'])]);local_metrics[name]={'equalChartRawPrimaryAccuracy':float(accuracy.mean()),'perChartAccuracy':accuracy.tolist(),'uids':[r['uid']for r in sorted(samples,key=lambda r:r['uid'])]}
        concepts[name]={}
        for uid,allowed in CONCEPTS.items():
            cases=[r for r in concept if r['kind']==kind and r['variant']==v and r['uid']==uid]
            if len(cases)!=6:raise ValueError('Missing concept predictions')
            concepts[name][uid]={'allowedSet':allowed,'overallTagsByOuter':[r['overall']['tags']for r in cases],
                'rawFramesByOuter':[[{'start':f['start'],'end':f['end'],'tag':f['tag'],'confidence':f['confidence']}for f in r['frames']]for r in cases],
                'allowedPrimaryWholeCount':sum(bool(set(r['overall']['tags'])&set(allowed))for r in cases)}
    pairs=[('hand/full','frequency/full'),('hand/full','hand/frozen'),('hand/full','hand/no_concept'),('hand/full','hand/no_strong'),('hand/full','original_fixed'),('hand/full','v1.2.1_LN_candidate')]
    paired={a+' vs '+b:paired_group_intervals(y,{b:hard[b],a:hard[a]},w,groups)for a,b in pairs};draw=np.random.default_rng(20261102).integers(0,6,size=(4000,6))
    for a,b in pairs:
        if a in local_metrics and b in local_metrics:
            delta=np.array(local_metrics[a]['perChartAccuracy'])-np.array(local_metrics[b]['perChartAccuracy']);paired[a+' vs '+b]['localEqualChartDelta95']=np.quantile(delta[draw].mean(1),[.025,.975]).tolist()
    for kind in protocol['kinds']:
        profiles[kind]=[]
        for o in range(6):
            folder=root/kind/str(o);search=read(folder/'search.json');choices=read(folder/'selection.json')['selections'];fits=read(folder/'fits.json');count+=len(fits)
            for trace in fits:
                starts+=len(trace['starts']);maximum_gradient=max(maximum_gradient,trace['maximumGradient']);losses=[s['loss']for s in trace['starts']];maximum_start_spread=max(maximum_start_spread,max(losses)-min(losses))
            with np.load(folder/'inner-bases.npz')as saved:z={k:saved[k]for k in saved.files}
            ids=z['validationIds'];iw=w[ids];ref=choices['full'];reference=policy_predictions(unpack(z,ref['headId']),[ref['policy']])[0];curves={}
            eligible=[s for s in search if s['headId']>0 and s['config']['conceptWeight']>0]
            for coordinate in ['localL2','strongWeight','conceptWeight','effortPower','weakBalance']:
                points=[]
                for value in dict.fromkeys(s['config'][coordinate]for s in eligible):
                    best=max((s for s in eligible if s['config'][coordinate]==value),key=lambda h:(h['macroF1'],h['macroRecall'],h['segmentAccuracy'],-h['headId'],-h['policyId']))
                    q=policy_predictions(unpack(z,best['headId']),[best['policy']])[0];points.append({'value':value,'selected':best,'changedTagSetMass':float(iw@np.any(q!=reference,axis=1)/iw.sum())})
                curves[coordinate]=points
            # Post-coordinate profiles fully reselect the local head as well
            # as the other two policy coordinates, using the saved full tensor.
            headids=[0]+[s['headId']for s in eligible]
            for coordinate in ['dominantShare','mixShare','secondRel']:
                points=[]
                for value in dict.fromkeys(p[coordinate]for p in POST):
                    policies=[i for i,p in enumerate(POST)if p[coordinate]==value];best=max(itertools_product(headids,policies),key=lambda hp:(z['macroF1'][hp],z['macroRecall'][hp],float(np.mean(z['manualAccuracy'][hp[0]])),hp[0]==0,-hp[0],-hp[1]))
                    hid,pid=best;q=policy_predictions(unpack(z,hid),[POST[pid]])[0];points.append({'value':value,'headId':int(hid),'policyId':int(pid),'macroF1':float(z['macroF1'][best]),'macroRecall':float(z['macroRecall'][best]),'changedTagSetMass':float(iw@np.any(q!=reference,axis=1)/iw.sum())})
                curves[coordinate]=points
            profiles[kind].append({'outer':o,'selected':ref,'fullyRetunedProfiles':curves,'frozenMatchedPost':choices['frozen']})
    if count!=complete['innerFits']or starts!=complete['innerTrainingStarts']or maximum_gradient>1e-5:raise ValueError('RC fit accounting or convergence failed')
    write(root/'metrics.json',{'whole':metrics,'local':local_metrics,'fitConceptSensitivity':concepts,'paired':paired,'maximumTrainingGradient':maximum_gradient,'maximumStartLossSpread':maximum_start_spread,
        'scope':protocol['scope'],'AP':'not assigned to final hard legacy tag decisions; raw local affinities are not whole-chart calibrated probabilities'})
    write(root/'parameter-profiles.json',profiles)
    names=['original_fixed','v1.2.1_LN_candidate','hand/frozen','frequency/full','hand/full','hand/no_concept','hand/no_strong'];fig,axes=plt.subplots(2,1,figsize=(14,8),gridspec_kw={'height_ratios':[1,2]},constrained_layout=True)
    for key,marker in [('precision','o'),('recall','s'),('f1','D')]:axes[0].plot(range(len(names)),[metrics[v]['macro'][key]for v in names],marker=marker,label='macro '+key)
    axes[0].plot(range(len(names)),[metrics[v]['exactMatch']for v in names],marker='^',label='exact tag set');axes[0].set_xticks(range(len(names)),names);axes[0].set_ylim(0,1);axes[0].grid(alpha=.15);axes[0].legend(ncol=4,fontsize=8);axes[0].spines[['top','right']].set_visible(False)
    matrix=np.array([[metrics[v]['perClass'][c]['f1']for c in mil.MAP_CLASSES]for v in names]);im=axes[1].imshow(matrix,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto');axes[1].set_yticks(range(len(names)),names);axes[1].set_xticks(range(y.shape[1]),mil.MAP_CLASSES,rotation=45,ha='right',fontsize=8)
    for i in range(len(names)):
        for j in range(y.shape[1]):axes[1].text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=7,color='white'if matrix[i,j]>.65 else'#172e42')
    fig.colorbar(im,ax=axes[1],label='per-class F1',shrink=.8);fig.suptitle('Roll-gated action residual: matched complete head/post selection',fontweight='bold');fig.savefig(root/'metrics.png',dpi=160,facecolor='white');plt.close(fig)
    lines=['# 局部roll动作修正：联调结果尚未满足真人语义','',protocol['scope'],'',
        '只重新分配Speed/Tech/Minijack的原有归一化affinity质量；其他类的分数不变，LN/Break路由不变。其他类仍可能因三类重新分配而成为硬主类，不能将“分数保留”误说成“其他硬标签必然不变”。99张单RC细类谱中仅44张单三类谱提供这项可变弱损失，整谱标签没有复制到每个窗口；全部349谱仍参与完整选参和留出评价。', '',
        '| 处理 | macro P | macro R | macro F1 | micro F1 | 集合全对 | 六谱局部准确率 |','|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        m=metrics[name];lines.append('| '+name+' | '+' | '.join(f'{v:.2%}'for v in [m['macro']['precision'],m['macro']['recall'],m['macro']['f1'],m['micro']['f1'],m['exactMatch']])+' | '+(f"{local_metrics[name]['equalChartRawPrimaryAccuracy']:.2%}"if name in local_metrics else'—')+' |')
    lines+=['', 'frozen不是旧默认参数：它也接受完全相同的245后处理组合联调。因此hand/full与hand/frozen的比较才隔离新增局部学习的收益。no_concept、no_strong均重新选择其他全部有效参数，no_hand由frequency/full提供。', '',
        f"完整声明网格：每表示672个局部配置及冻结头；每头交叉245个dominantShare/mixShare/secondRel组合。实际{count:,}项内层拟合、{starts:,}个训练初值、{complete['jointHeadPostChoices']:,}个头×后处理组合，{complete['outerDecisions']}项外层决策。最大梯度{maximum_gradient:.3g}；最大初值间训练loss差{maximum_start_spread:.3g}。其他旧策略坐标为了归因保持不变，这不是全部旧策略坐标的完整重调。", '',
        f"实际JS消费保存的外层模型，并调用原始平滑/聚合函数，核对{native['wholeOutputs']}份整谱输出、{native['manualOutputs']}份人工谱输出和{native['conceptOutputs']}份构造谱输出；最大局部分数差{native['maximumLocalScoreDifference']:.3g}。", '',
        '## 已用于训练的机制对照仍没有被稳定满足','',
        '| 对照 | frequency/full六折整谱主类 | hand/full六折整谱主类 |','|---|---|---|']
    for uid in CONCEPTS:lines.append('| '+uid+' | '+' / '.join('+'.join(t)or'empty'for t in concepts['frequency/full'][uid]['overallTagsByOuter'])+' | '+' / '.join('+'.join(t)or'empty'for t in concepts['hand/full'][uid]['overallTagsByOuter'])+' |')
    lines+=['', '这些是共享训练的机制对照，不是六次独立泛化测试。C24应为Tech，却在hand/full六折仍为Speed；C48部分成Vibro单类，是未调整的原规则分数在三类重分配后胜出，不能宣称学到了用户的Minijack/Anchor组合。Anchor不在本模型可学习分支中。候选未采用。', '',
        '## 成对歌曲组2000次bootstrap','']
    for name,effect in paired.items():
        lines.append('- '+name)
        for key in ['macroPrecision','macroRecall','macroF1','microF1','exactMatch']:
            ci=effect['delta95'][key];lines.append(f'  - {key}: [{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]pp')
        if'localEqualChartDelta95'in effect:lines.append('  - 六谱局部准确率: ['+', '.join(f'{100*v:+.2f}'for v in effect['localEqualChartDelta95'])+']pp')
    lines+=['', 'parameter-profiles.json保留完整条件重选择及实际tag变化。下一步首先检验弱bag损失的含义：局部时长份额不是“主要整谱slot”的置信度，直接−log平均份额会持续奖励把次要窗口也改成整谱类。主类置信映射需要单独匹配控制，不靠继续加大概念样本权重强压。', '',
        '整谱最终输出仍为硬决策，未伪造校准概率或AP。逐类P/R/F1、Jaccard、混淆取舍和全部实际预测保留。外层资料是回顾性数据；本轮4K/6K没有新结果。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    for name in names:print(name,metrics[name]['macro'],'exact',metrics[name]['exactMatch'],flush=True)


def itertools_product(a,b):
    for x in a:
        for y in b:yield(x,y)


if __name__=='__main__':run()
