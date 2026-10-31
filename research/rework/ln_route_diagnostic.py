"""Locate Release errors after a completed study, without fitting or selection.

Forced conditional LN decisions are counterfactuals, not deployable predictions
or extra validation results. Keep actual boundary overlaps and actual routes.
"""
from ln_adapter import np,json,Path,load,predict,ARCHIVE,CLASSES
from family import classify,load_metadata,model_scores
from native_search import pack
from collections import defaultdict


def conditional_ln(probability,rule,scorer):
    if scorer=='rules':scores=dict(rule)
    else:
        mdl=dict(zip(CLASSES,probability));scores={}
        for tag in dict.fromkeys(list(mdl)+list(rule)):
            if tag=='vibro':scores[tag]=min(1.,rule.get(tag,0)*2)
            elif tag in ['wildcard','ln wildcard']:scores[tag]=rule.get(tag,0)*.6
            elif scorer=='model':scores[tag]=mdl.get(tag,0)*1.4
            else:scores[tag]=.5*(mdl.get(tag,0)+rule.get(tag,0))
    if sum(scores.values())<=0:return {'tag':'ln wildcard','confidence':0.,'scores':{'ln wildcard':1.}}
    ranked=sorted(scores.items(),key=lambda pair:-pair[1]);tag,confidence=ranked[0]
    if tag in ['wildcard','ln wildcard']:
        for other,value in ranked[1:]:
            if other not in ['wildcard','ln wildcard']and value>.3:tag,confidence=other,value;break
    return {'tag':tag,'confidence':min(1.,float(confidence)),'scores':scores}


def overlap(a,b):return max(0.,min(a['end'],b['end'])-max(a['start'],b['start']))


def initial_break_reasons(record,metadata):
    features=[m['gates']for m in metadata['original']];nps=sorted(g['nps']for g in features if g['rowCount'])
    median=nps[len(nps)//2]if nps else 0.;reasons=[]
    for s,m,g in zip(record['segments'],metadata['original'],features):
        low=(s['end']-s['start']>=2500 and g['nps']<median and
             (g['nps']<18.5 and g['segIvLogStd']>.41 or
              g['chordMean']<1.9 and g['vibroFraction']>.3 and g['segRhythmChangeRate']<.1 or
              g['nps']<22.5 and g['segIvLogStd']>.25 and g['vibroFraction']<.45 and g['lnWeightRatio']<.5))
        reasons.append('head_only_markBreaks'if m['family0']=='break'else'no_head_rows'if not g['rowCount']else'lowStakes'if low else None)
    return reasons


def run():
    root=Path('research/rework/results/ln-adapter-r1');protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    main,manual,_=load();meta=load_metadata();saved=json.loads((root/'outer-segments.json').read_text(encoding='utf8'))
    result={};verified=0;details=[]
    for variant in ['geometry3','no_update']:
        totals=defaultdict(float);bychart=[];routes=defaultdict(lambda:defaultdict(float));break_reasons=defaultdict(float)
        for o,plan in enumerate(protocol['plans']):
            profile=json.loads((root/variant/str(o)/'outer-model.json').read_text(encoding='utf8'));policy=profile['policy']
            records=[manual[i]for i in plan['strongTest']];ln=predict(profile['model'],records)
            for r,prob in zip(records,ln):
                rc=model_scores(ARCHIVE['rc'],np.array([s['original']for s in r['segments']]))
                rows=classify(r,meta[r['uid']],ARCHIVE,ln_ratio=policy['lnRatio'],scorer=policy['scorer'],precomputed={'rc':rc,'ln':prob})
                actual=pack([{'partitions':[{'cases':[{'raw':rows}]}]}],policy)['chunks'][0]
                expected=[s for s in saved if s['variant']==variant and s['uid']==r['uid']]
                assert [(s['start'],s['end'],s['tag'],s['family'])for s in actual]==[(s['start'],s['end'],s['tag'],s['family'])for s in expected]
                assert len(actual)==len(r['segments']),'Diagnostic requires raw coherent intervals'
                counts=defaultdict(float)
                annotations=[a for a in r['annotations']if len(a['tags'])==1 and a['tags'][0]not in ['wildcard','ln wildcard','rc mix','ln mix','hybrid']]
                reasons=initial_break_reasons(r,meta[r['uid']])
                for j,(s,p,m)in enumerate(zip(actual,prob,meta[r['uid']]['original'])):
                    forced=conditional_ln(p,m['ruleLn'],policy['scorer'])
                    if s['family']=='ln':
                        assert forced['tag']==s['tag']
                        assert abs(forced['confidence']-s['confidence'])<1e-12
                        assert forced['scores']==s['scores'];verified+=1
                    positive=sum(overlap(a,s)for a in annotations if a['tags']==['release'])
                    negative=sum(overlap(a,s)for a in annotations if a['tags']!=['release'])
                    if positive:
                        route=s['family'];routes[route]['positiveMs']+=positive
                        if route=='break':break_reasons[reasons[j]or('post_low_confidence_or_sparse'if rows[j]['family']=='break'else'extra_burden_gate')]+=positive
                        routes[route]['actualReleaseMs']+=positive*(s['tag']=='release')
                        routes[route]['forcedConditionalReleaseMs']+=positive*(forced['tag']=='release')
                        counts['positiveMs']+=positive;counts['actualReleaseMs']+=positive*(s['tag']=='release')
                        counts['forcedConditionalReleaseMs']+=positive*(forced['tag']=='release')
                        counts['rawReleaseArgmaxMs']+=positive*(CLASSES[int(p.argmax())]=='release')
                        counts['releaseMajorityRawIntervalMs']+=positive*(positive>.5*(s['end']-s['start']))
                        details.append({'variant':variant,'uid':r['uid'],'start':s['start'],'end':s['end'],'positiveMs':positive,'definiteOtherMs':negative,
                          'actualFamily':route,'actualTag':s['tag'],'initialBreakReason':reasons[j],'forcedConditionalLnTag':forced['tag'],'conditionalReleaseProbability':float(p[CLASSES.index('release')]),
                          'conditionalArgmax':CLASSES[int(p.argmax())],'positiveFractionOfRawInterval':positive/(s['end']-s['start'])})
                    if negative:
                        counts['definiteOtherMs']+=negative;counts['actualFalseReleaseMs']+=negative*(s['tag']=='release')
                        counts['forcedConditionalFalseReleaseMs']+=negative*(forced['tag']=='release')
                for k,v in counts.items():totals[k]+=v
                bychart.append({'uid':r['uid'],**dict(counts)})
        result[variant]={'totals':dict(totals),'actualRouteOnReleaseTime':{k:dict(v)for k,v in routes.items()},'breakReasonPositiveMs':dict(break_reasons),'charts':bychart}
    output=root/'route-diagnostic.json';output.write_text(json.dumps({'scope':'post-study mechanical counterfactual on unchanged raw boundaries; no fitting/selection; forced LN ignores all Break and family-route constraints',
      'verifiedActualLnRows':verified,'variants':result,'positiveOverlaps':details},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    lines=['# Release 错误卡在何处','', '本诊断不选参数、不重新训练。强制LN条件判定跳过Break及家族路由，仅用于定位阻碍；不是可部署方案的验证分数。仅统计确定单主标签。',
      '', '| 方案 | 实际Release召回 | 强制LN条件判定召回 | 条件专家argmax召回 | 阳性时间落在Release占多数的原始窗 |','|---|---:|---:|---:|---:|']
    for v,r in result.items():
        t=r['totals'];den=t['positiveMs'];lines.append(f"| {v} | {t['actualReleaseMs']/den:.2%} | {t['forcedConditionalReleaseMs']/den:.2%} | {t['rawReleaseArgmaxMs']/den:.2%} | {t['releaseMajorityRawIntervalMs']/den:.2%} |")
    lines+=['',f'条件LN辅助实现已与实际消费者的 {verified} 个LN窗口逐标签、分数、置信度核对；实际路由输出也与已保存外层输出核对。分家族时长、强制判定误报、逐原始窗混合比例见 `route-diagnostic.json`。']
    (root/'ROUTE_DIAGNOSTIC.md').write_text('\n'.join(lines)+'\n',encoding='utf8');print('\n'.join(lines),flush=True)


if __name__=='__main__':run()
