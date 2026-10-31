"""Reuse only fits whose actual training inputs survive the release-order fix.

No optimization is rerun. Cache migration separately verifies every other
coordinate over the entire corpus; this also checks packed whole-fold losses
and gradients for each allowed recipe/window/gate combination.
"""
import shutil
from coarse_model import np,torch,json,Path,Packed,SPECS,training_rows
from coarse_data import load_headers,plans
from coarse_search import read,write


def run():
    old=Path('.cache/rework-coarse-r1');new=Path('.cache/rework-coarse-r2')
    audit=read(new/'repair-audit.json')
    assert not any(r['keys']==4 for r in audit['changedCharts'])
    weak,manual,_=load_headers();split=plans(weak,manual)[0]['inner'][0]
    headers=[weak[i]for i in split['train']]+[manual[i]for i in split['strongTrain']]
    def materialize(root):
        return [{**r,'windows':read(root/(r['uid'].replace('/','__')+'.json'))['windows']}for r in headers]
    before,after=materialize(old),materialize(new);checks=[]
    for variant in ['base40','weak4_only']:
        spec=SPECS[variant]
        for window in [2000,4000,8000]:
            for gate in [0.,.2]:
                cfg={'windowMs':window,'burdenGate':gate,'localL2':.003,'strongWeight':.5,'source4Weight':.5}
                a=Packed(training_rows(before,spec),spec,cfg);b=Packed(training_rows(after,spec),spec,cfg)
                for key in ['mean','scale']:assert np.array_equal(getattr(a,key),getattr(b,key)),key
                for key in ['x','eligible','effort','duration','bag','labels','weak_weight','strong_index','mask','strong_weight']:
                    assert torch.equal(getattr(a,key),getattr(b,key)),key
                torch.manual_seed(713);theta=torch.randn(5*(a.d+1)+2)*.03;theta[-2:]=torch.tensor([.4,.6])
                left=theta.clone().requires_grad_(True);right=theta.clone().requires_grad_(True)
                la=a.objective(left,cfg);lb=b.objective(right,cfg);la.backward();lb.backward()
                assert torch.equal(la,lb) and torch.equal(left.grad,right.grad)
                checks.append({'variant':variant,'windowMs':window,'gate':gate,'records':len(a.records),'windows':len(a.x),
                               'packedInputsLossAndGradientBitwiseEqual':True})
    source=Path('.cache/rework-coarse-shared-r1-fits');target=Path('.cache/rework-coarse-shared-r3-fits')
    four={r['uid']for r in weak if r['keys']==4};copied=[];rejected=[]
    for p in source.rglob('*.json.gz'):
        relative=p.relative_to(source);parts=relative.parts;variant=parts[1]if parts[0]=='outer'else parts[0]
        saved=read(p);keys=saved['model']['keys'];spec=saved['spec']
        if variant=='weak4_only':
            assert spec.get('weak4Only') and set(saved['trainUids'])<=four
            # This coordinate was identically zero in actual four-key input.
            k=keys.index('release:sameHandOrder');D=len(keys)
            assert saved['model']['mean'][k]==0 and all(saved['model']['theta'][c*D+k]==0 for c in range(5))
        elif variant=='base40':assert 'release:sameHandOrder'not in keys and spec==SPECS['base40']
        else:rejected.append(str(relative));continue
        destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert read(destination)==saved
        else:shutil.copy2(p,destination)
        copied.append(str(relative))
    result={'scope':'reuse fitted coefficients only; all inner/outer validation predictions regenerated against repaired caches',
      'corpusCoordinateAudit':audit['unchanged'],'wholeTrainingFoldChecks':checks,
      'reusedFits':len(copied),'files':copied,'notReused':rejected,
      'conditions':'base40 excludes the only changed coordinate; pure4 training has no changed input at all. Source config and trainUids still checked by fit_or_load.'}
    write('research/rework/results/coarse-shared-r3/reuse-audit.json',result)
    print({k:v for k,v in result.items()if k not in ['files','wholeTrainingFoldChecks','notReused']},flush=True)


if __name__=='__main__':run()
