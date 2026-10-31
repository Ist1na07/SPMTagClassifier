"""Exercise both structural branches through fresh parse JS and Python math."""
import subprocess
from coarse_supported import predict,np,FINE
import json
from pathlib import Path


def run():
    scratch=Path('.cache/coarse-supported-primitives.json')
    subprocess.run(['node','research/rework/coarse-primitives-parity.cjs',str(scratch)],check=True)
    value=json.loads(scratch.read_text(encoding='utf8'));maximum={};counts={'jack':0,'ln':0};windows=0
    for case in value['cases']:
        got=predict(value['model'],[case['record']],case['mode']);expected=case['expected'];windows+=len(expected['local'])
        for key in ['local','weights','composition']:
            actual=got[key][0]
            error=float(np.abs(np.array(expected[key])-actual).max());maximum[key]=max(maximum.get(key,0),error)
            assert error<1e-9,(case['record']['uid'],case['mode'],key,error)
        if case['mode']!='none':
            rules=[None if k<0 else FINE[int(k)]for k in got['rules'][0]];assert rules==expected['rules']
            for k in counts:counts[k]+=rules.count(k)
    assert counts['jack']>0 and counts['ln']>0
    out=Path('research/rework/results/coarse-supported-primitives-parity-r1');out.mkdir(exist_ok=True)
    result={'scope':'definition-level arithmetic parity, not human accuracy or model selection','cases':len(value['cases']),'windows':windows,
       'ruleWindowsByType':counts,'modes':['none','jack','ln','both'],'maximumAbsoluteErrors':maximum,
       'sources':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['coarse_primitives_parity.py','coarse-primitives-parity.cjs','physical-support.cjs','coarse_supported.py','coarse-runtime.cjs']}}
    (out/'REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print({k:v for k,v in result.items()if k!='sources'},flush=True)


if __name__=='__main__':run()
