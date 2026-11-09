"""Correct the legacy source-coordinate/standalone-clock diagnostic mismatch.

This does not fit, select, change a metric or modify any training input.
"""
from targeted_sequence_model import World,frames
from targeted_sequence_study import ROOT,PLANS
from pressure_data import read
from rc_action_runtime import smooth
from whole_reference import write

def run():
    choices=read(ROOT/'joint-inner-selection.json')['choices'];ann=read('research/frontier/annotations-r5.json')
    source=read('.cache/targeted-natural-queries-r1.json');reference=World('native');groups={r['uid']:r['group']for r in reference.main}
    worlds={};retained=[];out=[]
    for q in source['rows']:
        group=groups[ann['cases'][q['name']]['sourceUid']]
        folds=[o for o,p in enumerate(PLANS)if group in {reference.main[i]['group']for i in p['test']}]
        if len(folds)!=1:raise ValueError('Source group not held out exactly once')
        o=folds[0]
        if group in {reference.main[i]['group']for i in PLANS[o]['train']}:raise ValueError('Source query group leaked into TRAIN')
        for name,cs in choices.items():
            c=cs[o];context=c['context'];uid=q['uid']
            if context not in worlds:worlds[context]=World(context)
            w=worlds[context];row=q['local']if context=='local2s'else dict(frames=q['parents'],values=q['originalValues'],parentIds=list(range(len(q['parents']))))
            w.inputs[uid]={**row,'intervals':[[s['start'],s['end']]for s in row['frames']],'parents':q['parents']}
            m=read(ROOT/context/c['kind']/str(o)/'outer-models.json')[str(c['headId'])]['model'];retained.append(m)
            fs=smooth(frames(w,uid,m));a,b=9000,16500
            accuracy=sum(max(0,min(b,s['end'])-max(a,s['start']))for s in fs if s['tag']=='speed')/(b-a)
            profile=dict(context=context,model=m,policy=c['policy'])
            out.append(dict(query=q['name'],uid=uid,source=q['source'],name=name,outer=o,sourceGroup=group,choice=c,profile=profile,speedPrimaryCoverage=accuracy,expected=fs))
    write(ROOT/'query-diagnostics-correction.json',dict(
        correction='Producer regressions.json used9-16.5s on historical local/Fxx original-song coordinates109942/79946ms; those zero coverages are invalid diagnostics. No query enters fitting or selection, so no fit or INNER/OUTER whole/manual result changes. This file supersedes that diagnostic only.',
        effectiveScope='Full standalone excerpt natural segmentation; evaluate9-16.5s only with the model whose source-song group is held out. No label input or forced target boundary in extraction.',
        cases=out,sourceSnapshot=__import__('pathlib').Path(__file__).read_text(encoding='utf8')))
    # Repair the effective handoff, rather than leaving known-invalid zeros
    # in its query fields. Frozen producer per-scope diagnostics remain intact.
    analysis=read(ROOT/'analysis.json')
    for name,result in analysis['joint'].items():result['regressions']=[r for r in out if r['name']==name]
    analysis['queryDiagnosticSource']='query-diagnostics-correction.json; natural excerpt clock, source-held-out models. Whole/manual metrics and selections unchanged.'
    write(ROOT/'analysis.json',analysis)
    report=ROOT/'REPORT.md';text=report.read_text(encoding='utf8');marker='\n## 正确的摘录回归口径\n'
    text=text.split(marker)[0]+marker+'\nF01/F02按完整摘录的自然分段，在9–16.5秒计算主Speed覆盖；只使用来源歌曲组留出的模型。所有策略的两条覆盖均为100%，没有新改善或退步。历史local/Fxx缓存使用原谱坐标，不能套用摘录时钟；修复依据、来源组和实际输出见query-diagnostics-correction.json。\n'
    report.write_text(text,encoding='utf8')
    print('Corrected source-held-out natural-excerpt queries',[(r['name'],r['query'],r['speedPrimaryCoverage'])for r in out],flush=True)

if __name__=='__main__':run()
