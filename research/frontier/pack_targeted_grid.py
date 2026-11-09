"""Publish the complete joint INNER metric grid compactly; no new fits."""
import gzip,itertools,json
from targeted_sequence_study import ROOT,CONFIGS,POST,PLANS
from targeted_sequence_model import FEATURES,World
from pressure_data import read
from whole_reference import write

def run():
    contexts=['native','local2s'];tables=[];models=[];w=World('native')
    for context,kind,outer in itertools.product(contexts,FEATURES,range(6)):
        folder=ROOT/context/kind/str(outer);rows=read(folder/'inner-search.json')
        if len(rows)!=(len(CONFIGS)+1)*len(POST):raise ValueError('Joint metric grid incomplete')
        block=[]
        for hid,cfg in enumerate([None]+CONFIGS):
            rs=rows[hid*len(POST):(hid+1)*len(POST)]
            if any(r['headId']!=hid or r['policyId']!=i or r['config']!=cfg for i,r in enumerate(rs)):raise ValueError('Grid axis identity changed')
            if any(r['manualAccuracy']!=rs[0]['manualAccuracy']or r['pressureConfusion']!=rs[0]['pressureConfusion']for r in rs):raise ValueError('Local metric unexpectedly depends on whole post policy')
            block.append(dict(manualAccuracy=rs[0]['manualAccuracy'],pressureConfusion=rs[0]['pressureConfusion'],macroF1=[r['macroF1']for r in rs],macroRecall=[r['macroRecall']for r in rs]))
        tables.append(dict(context=context,kind=kind,outer=outer,heads=block))
        models.append(dict(context=context,kind=kind,outer=outer,trainUids=[w.main[i]['uid']for i in PLANS[outer]['train']],strongTrainUids=[w.manual[i]['uid']for i in PLANS[outer]['strongTrain']],heads=read(folder/'outer-models.json')))
    value=dict(schemaVersion=1,contexts=contexts,representations=list(FEATURES),configs=[None]+CONFIGS,postPolicies=POST,tables=tables,scope='Complete actual INNER joint grid. Local metrics are displayed manual accuracy and pressure-weighted symmetric Tech/CS error, not renamed probability accuracy.')
    raw=json.dumps(value,separators=(',',':'),allow_nan=False).encode('utf8');data=gzip.compress(raw,mtime=0)
    (ROOT/'inner-grid.json.gz').write_bytes(data);write(ROOT/'selected-outer-heads.json',models)
    print('Complete joint-grid cells',len(tables)*(len(CONFIGS)+1)*len(POST),'gzip bytes',len(data))

if __name__=='__main__':run()
