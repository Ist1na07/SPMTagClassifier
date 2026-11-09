"""Export replay expectations; no fitting or statistical re-evaluation."""
from pathlib import Path
from pressure_data import mil, read
from whole_reference import write
from rc_action_runtime import aggregate, sufficient, smooth


def run():
    root=Path('research/overlay-quality/overlay-impression-fixed-post-r1')
    a=read(root/'analysis.json');data=read('research/overlay-quality/break-regularity-r3/frames.json')
    rows={r['uid']:r for r in data};main,manual,_=mil.load();plans=read('research/rework/results/mil-stage1-r4/protocol.json')['plans']
    def expected(uid,choice,phase):
        source=rows[uid]['frames'][choice['configId']]
        return dict(uid=uid,phase=phase,config=choice['config'],policy=choice['policy'],segments=smooth(source),overall=aggregate(sufficient(source),choice['policy']))
    choices=[r for r in a['choices']if r['name']=='experience_fixed_post'];cases=[]
    for selected in choices:
        plan=plans[selected['outer']];c=selected['choice']
        cases.extend(expected(main[i]['uid'],c,'outer')for i in plan['test'])
        cases.extend(expected(manual[i]['uid'],c,'outer-manual')for i in plan['strongTest'])
    final=a['final']['experience_fixed_post'];cases.extend(expected(r['uid'],final,'final')for r in data)
    queries=[]
    for q in a['sourceHeldOutQueries']['experience_fixed_post']:
        queries.append(dict(uid=q['uid'],config=q['choice']['config'],policy=q['choice']['policy'],
            targetMs=[9000,16500],segments=smooth(q['frames']),expectedPrimaryAccuracy=q['primaryAccuracy']))
    file=Path('.cache/overlay-native-expected-r1.json')
    if file.exists():raise ValueError('Preserve numerical evidence; use a new fixture name')
    write(file,dict(cases=cases,queries=queries,role='Pure replay fixture; final399is runtime parity, not held-out accuracy.'))
    print('Native replay cases',len(cases),'source-held-out queries',len(queries),flush=True)


if __name__=='__main__':run()
