"""Attach real local supervision to original song groups, never all-fold."""
from pathlib import Path
from pressure_data import mil,read
from whole_reference import write


def eligible_targets(targets,train_groups):
    return [r for r in targets if r['sourceGroup']in set(train_groups)]


def run():
    records,_,_=mil.load();lookup={r['uid']:r for r in records};rows=[]
    for round_ in ['r4','r5','r6']:
        annotation=read('research/frontier/annotations-'+round_+'.json')
        for name,a in annotation['cases'].items():
            original=lookup[a['sourceUid']];rows.append({'uid':'local/'+name,'id':name,'round':round_,'sourceUid':a['sourceUid'],'sourceGroup':original['group'],'source':original['source'],'intervals':[a['sourceTargetMs']],
                'primary':a.get('primaryCategory',a['localCategory']),'component':a.get('additionalLocalComponent'),'confidenceQualitative':a['confidenceQualitative'],'numericConfidence':None})
    plans=read('research/rework/results/mil-stage1-r4/protocol.json')['plans'];scopes=[]
    for o,p in enumerate(plans):
        for split,train in [('outer',p['train'])]+[('inner'+str(j),q['train'])for j,q in enumerate(p['inner'])]:
            groups={records[i]['group']for i in train};scopes.append({'outer':o,'split':split,'eligibleLocalUids':[r['uid']for r in eligible_targets(rows,groups)]})
        heldout={records[i]['group']for i in p['test']};train={records[i]['group']for i in p['train']}
        if heldout&train:raise ValueError('Original whole splits leak groups')
    output=Path('.cache/frontier-local-targets-r1.json')
    if output.exists():raise ValueError('Preserve local input manifest; use a new revision')
    write(output,{'scope':'Exact local targets only; main349 inventory unchanged; annotations restricted to original song TRAIN groups; primary and additional component are separate, no invented probability','records':rows,'trainingScopes':scopes,'sourceSnapshot':Path(__file__).read_text(encoding='utf8')});print('local target manifest',len(rows),len(scopes),flush=True)


if __name__=='__main__':run()
