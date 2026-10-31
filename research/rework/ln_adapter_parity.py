"""Selected real trained LN updates through fresh JS and actual Python caller."""
from ln_adapter import np,json,Path,load,predict,ARCHIVE
from family import classify,load_metadata,model_scores
from native_search import pack,summaries,decisions,FINE
from mil import MAP_CLASSES
from ln_gate import pack_gated
import subprocess,argparse


def run(directory='research/rework/results/ln-adapter-r1',output='research/rework/results/ln-adapter-runtime-parity-r1'):
    root=Path(directory);protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    if 'plans'not in protocol:protocol=json.loads(Path(protocol['parent']).read_text(encoding='utf8'))
    main,manual,_=load();metadata=load_metadata();requests=[];expected=[];gate_changes=0
    for o,plan in enumerate(protocol['plans']):
        if not(root/'geometry3'/str(o)/'complete.json').exists():raise RuntimeError('Selected main outer profile not complete')
        profile=json.loads((root/'geometry3'/str(o)/'outer-model.json').read_text(encoding='utf8'));policy=profile['policy']
        records=[main[i]for i in plan['test'][:2]]+[manual[i]for i in plan['strongTest']]
        ln=predict(profile['model'],records)
        for r,prob in zip(records,ln):
            x=np.array([s['original']for s in r['segments']]);rc=model_scores(ARCHIVE['rc'],x)
            rows=classify(r,metadata[r['uid']],ARCHIVE,ln_ratio=policy['lnRatio'],scorer=policy['scorer'],precomputed={'rc':rc,'ln':prob})
            value={'partitions':[{'cases':[{'raw':rows}]}]};data=pack_gated([value],policy,profile.get('extraGate','all'));summary=summaries(data,policy);p=decisions(summary,policy)[0]
            old=pack([value],policy)['chunks'][0];gate_changes+=sum(a['tag']!=b['tag']for a,b in zip(old,data['chunks'][0],strict=True))
            expected.append({'uid':r['uid'],'rows':data['chunks'][0],'tags':[c for c,b in zip(MAP_CLASSES,p)if b],
                             'shares':summary['shares'][0],'distribution':summary['distribution'][0],'diversity':summary['diversity'][0]})
            requests.append({'uid':r['uid'],'source':r['source'],'profile':profile})
    scratch=Path('.cache/ln-adapter-parity');scratch.mkdir(exist_ok=True);(scratch/'requests.json').write_text(json.dumps(requests),encoding='utf8')
    subprocess.run(['node','research/rework/ln-adapter-parity-driver.cjs',str(scratch/'requests.json'),str(scratch/'actual.json')],check=True)
    actual=json.loads((scratch/'actual.json').read_text(encoding='utf8'));errors={};intervals=0
    def near(key,a,b):
        error=float(np.abs(np.array(a)-np.array(b)).max());errors[key]=max(errors.get(key,0),error)
        assert error<1e-9,(key,error)
    for a,b in zip(actual,expected,strict=True):
        assert a['uid']==b['uid']and a['tags']==b['tags'];assert len(a['rows'])==len(b['rows']);intervals+=len(a['rows'])
        near('shares',[a['shares'].get(tag,0)for tag in FINE],b['shares']);near('diversity',a['diversity'],b['diversity'])
        if a['distribution']is not None:near('distribution',a['distribution'],b['distribution'])
        for x,y in zip(a['rows'],b['rows'],strict=True):
            for key in ['start','end','tag','family']:assert x[key]==y[key],(a['uid'],key,x[key],y[key])
            for key in ['confidence','probability','burden']:near(key,x[key],y[key])
    out=Path(output);out.mkdir(exist_ok=True,parents=True)
    result={'scope':'actual learned main outer profiles on held-out real charts; fresh JS parse/geometry/expert replacement/rules/aggregation vs actual Python research caller',
       'charts':len(actual),'rawIntervals':intervals,'tagsFamiliesTimes':'exact','maximumAbsoluteErrors':errors,'extraGateChangedIntervalsVsAllGate':gate_changes,
       'sources':{n:Path(__file__).with_name(n).read_text(encoding='utf8')for n in ['ln-adapter-direct.cjs','ln-adapter-parity-driver.cjs','ln_adapter_parity.py','native-policy.cjs','ln_gate.py','native_search.py']}}
    (out/'REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print({k:v for k,v in result.items()if k!='sources'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',default='research/rework/results/ln-adapter-r1');p.add_argument('--output',default='research/rework/results/ln-adapter-runtime-parity-r1');a=p.parse_args();run(a.directory,a.output)
