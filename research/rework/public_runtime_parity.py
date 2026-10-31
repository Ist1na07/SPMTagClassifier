"""Final all-training profiles: actual Node/browser API vs Python consumers."""
from coarse_data import load_headers
from coarse_model import materialize,np,Path,json,FINE
from coarse_supported import predict as coarse_predict
from ln_adapter import load,predict,ARCHIVE
from family import load_metadata,model_scores,classify
from ln_gate import pack_gated
from native_search import summaries,decisions,MAP_CLASSES,CLASSES
import subprocess


def run():
    weak,_,_=load_headers();records=[];expect=[];requests=[];errors={};counts={}
    selected=[]
    for tag in FINE:
        candidates=sorted([r for r in weak if r['keys']==4 and r['coarseTags']==[tag]],key=lambda r:(r['notes'],r['uid']))
        selected.extend([candidates[0],candidates[-1]])
    four=materialize(selected);six_root=Path('.cache/rework-six-physical-r2');inventory=json.loads((six_root/'inventory.json').read_text())['records'];ordered=sorted(inventory,key=lambda r:(r['notes'],r['uid']))
    six=[json.loads((six_root/(r['uid'].replace('/','__')+'.json')).read_text())for r in [ordered[round(i*(len(ordered)-1)/4)]for i in range(5)]]
    for K,rs in [(4,four),(6,six)]:
        profile=json.loads(Path(f'research/rework/results/coarse-final-r1/{K}k/profile.json').read_text());p=coarse_predict(profile['model'],rs,'both')
        for j,r in enumerate(rs):
            requests.append({'uid':r['uid'],'keys':K,'source':r['source']});expect.append({'kind':'coarse','record':r,'prediction':p,'at':j,'model':profile['model']})
    main,manual,_=load();rs=main[::max(1,len(main)//6)][:6]+manual;meta=load_metadata();profile=json.loads(Path('research/rework/results/ln-final-r1/profile.json').read_text());ln=predict(profile['model'],rs);policy=profile['policy']
    for r,local in zip(rs,ln):
        rc=model_scores(ARCHIVE['rc'],np.array([s['original']for s in r['segments']]))
        rows=classify(r,meta[r['uid']],ARCHIVE,ln_ratio=policy['lnRatio'],scorer=policy['scorer'],precomputed={'rc':rc,'ln':local})
        data=pack_gated([{'partitions':[{'cases':[{'raw':rows}]}]}],policy,profile['extraGate']);summary=summaries(data,policy);p=decisions(summary,policy)[0]
        requests.append({'uid':r['uid'],'keys':7,'source':r['source']});expect.append({'kind':'ln','data':data,'summary':summary,'tags':[t for t,b in zip(MAP_CLASSES,p)if b]})
    scratch=Path('.cache/public-runtime-parity');scratch.mkdir(exist_ok=True);(scratch/'requests.json').write_text(json.dumps(requests))
    subprocess.run(['node','research/rework/public-parity-driver.cjs',str(scratch/'requests.json'),str(scratch/'actual.json')],check=True)
    actual=json.loads((scratch/'actual.json').read_text());windows=intervals=0
    def near(k,a,b,tol=1e-9):
        delta=float(np.abs(np.array(a)-np.array(b)).max());errors[k]=max(errors.get(k,0.),delta);assert delta<tol,(k,delta)
    for request,a,e in zip(requests,actual,expect,strict=True):
        assert request['uid']==a['uid'];counts[str(a['keys'])]=counts.get(str(a['keys']),0)+1
        assert a['segments'][0]['start']==a['first']and a['segments'][-1]['end']==a['last']+1
        intervals+=len(a['segments'])
        if e['kind']=='coarse':
            p=e['prediction'];j=e['at'];near('coarseComposition',list(a['coarse']['composition'].values()),p['composition'][j]);near('coarseLocal',[s['probability']for s in a['windows']],p['local'][j]);near('coarseWeight',[s['weight']for s in a['windows']],p['weights'][j],1e-8);windows+=len(a['windows'])
            raw_labels=p['local'][j].argmax(1);rows=e['record']['windows'][str(e['model']['config']['windowMs'])];expected=[]
            for w,label in zip(rows,raw_labels,strict=True):
                tag=(FINE+['break'])[label];tag='stamina'if tag=='chordstream'else tag
                if expected and expected[-1][2]==tag:expected[-1][1]=w['end']
                else:expected.append([w['start'],w['end'],tag])
            assert expected==[[s['start'],s['end'],s['tag']]for s in a['segments']]
        else:
            # Python's binary label vector is in taxonomy order; the public
            # API deliberately ranks multiple labels by actual support.
            assert set(a['overall']['tags'])==set(e['tags']),(a['uid'],a['overall']['tags'],e['tags'])
            if len(a['overall']['tags'])==2:
                first,second=a['overall']['tags'];assert a['overall']['shares'][first]>=a['overall']['shares'][second]
            near('lnShares',[a['overall']['shares'].get(t,0)for t in FINE_LN],e['summary']['shares'][0]);near('lnDiversity',a['overall']['diversity'],e['summary']['diversity'][0])
            rows=e['data']['chunks'][0];assert len(rows)==len(a['segments']);near('lnLocal',[s['probability']for s in a['segments']],[s['probability']for s in rows])
            for b,c in zip(a['segments'],rows,strict=True):
                assert (b['sourceStart'],b['sourceEnd'],b['tag'],b['family'])==(c['start'],c['end'],c['tag'],c['family'])
                near('lnConfidence',b['confidence'],c['probability'][CLASSES.index(c['tag'])]);near('lnLegacyConfidence',b['legacyConfidence'],c['confidence'])
    output=Path('research/rework/results/public-runtime-parity-r1');output.mkdir(exist_ok=True)
    result={'scope':'final all-training profiles, actual shipped Node API and standalone browser IIFE vs actual Python research consumers; numeric/contract verification, not independent accuracy',
      'cases':len(actual),'byKeys':counts,'coarseWindows':windows,'displayIntervals':intervals,'browserVsNode':'exact complete JSON equality','maximumAbsoluteErrors':errors,
      'lnIntervalMeaning':'original sourceStart/sourceEnd match fitted research intervals; display envelope extends first/last only',
      'multiLabelOrdering':'same label sets as Python taxonomy vector; public API order additionally checked against descending actual support',
      'sources':{n:Path(__file__).with_name(n).read_text()for n in ['public_runtime_parity.py','public-parity-driver.cjs']}}
    (output/'REPORT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print({k:v for k,v in result.items()if k!='sources'})


FINE_LN=[c for c in CLASSES if c not in ['break','wildcard','ln wildcard']]
if __name__=='__main__':run()
