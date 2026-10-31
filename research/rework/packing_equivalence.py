"""Check the runtime-only packing repair against the actual producing source.

No optimization is repeated. The prior cold fits remain valid if all train
inputs, scalar objective and its gradient agree bitwise on this complete fold.
"""
import coarse_model as current
from coarse_data import load_headers,plans
from coarse_search import read,write
from mil import Path,np,torch,time

def run():
    protocol=read('research/rework/results/coarse-4k-baseline-r1/protocol.json')
    source=protocol['sourceSnapshot']['coarse_model.py'];namespace={'__name__':'recorded_coarse_reference'}
    exec(compile(source,'recorded-coarse-model.py','exec'),namespace)
    weak,manual,_=load_headers();split=plans(weak,manual)[0]['inner'][0]
    rows=current.materialize([weak[i]for i in split['train']]+[manual[i]for i in split['strongTrain']])
    config={'windowMs':4000,'burdenGate':.2,'localL2':.003,'strongWeight':.5,'source4Weight':.5};spec=current.SPECS['full53']
    start=time.perf_counter();a=namespace['Packed'](rows,spec,config);old_seconds=time.perf_counter()-start
    start=time.perf_counter();b=current.Packed(rows,spec,config);new_seconds=time.perf_counter()-start
    for key in ['mean','scale']:assert np.array_equal(getattr(a,key),getattr(b,key)),key
    for key in ['x','eligible','effort','duration','bag','labels','weak_weight','strong_index','mask','strong_weight']:
        assert torch.equal(getattr(a,key),getattr(b,key)),key
    for key in ['keys','lengths','n','d','weak','missing','scope']:assert getattr(a,key)==getattr(b,key),key
    torch.manual_seed(913);theta=torch.randn(len(current.FINE)*(a.d+1)+2)*.03;theta[-2:]=torch.tensor([.4,.6])
    left=theta.clone().requires_grad_(True);right=theta.clone().requires_grad_(True)
    la=a.objective(left,config);lb=b.objective(right,config);la.backward();lb.backward()
    assert torch.equal(la,lb);assert torch.equal(left.grad,right.grad)
    evidence={'scope':'actual complete inner-training fold, shared full53 with manual annotations and nonzero gate; recorded baseline producing code vs runtime-only guard repair',
      'records':len(rows),'windows':len(a.x),'packedInputsBitwiseEqual':True,'scalarObjectiveBitwiseEqual':True,'gradientBitwiseEqual':True,
      'oldPackingSeconds':old_seconds,'newPackingSeconds':new_seconds,'validColdFitReuse':'same scalers, tensors, loss and optimizer; no refit or retuning required for these existing coefficients',
      'guardChange':'only active-window validation is vectorized; atomic file publication does not affect numerical fitting'}
    write('research/rework/results/coarse-shared-r2/packing-equivalence.json',evidence)
    print(evidence,flush=True)

if __name__=='__main__':run()
