"""Check actual maintained consumers against the independently replayed gate grid."""
from study import *
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'local_feedback'))
from product_readout_report import local


def run():
    st=Study();rows={r['uid']:r for r in map(json.loads,Path('.cache/break-pressure-product-r1.jsonl').read_text(encoding='utf8').splitlines())};classes=st.w.manifest['classes'];fine=st.w.manifest['fine'];results={};error=0.;changed=[]
    if set(rows)!=set(st.d.records):raise ValueError('Incomplete native output')
    for name in ['actions','detail','compact']:
        from model import metrics
        selection=read(ROOT/name/'selection.json')['off'if name=='actions'else'pressure']['config'];frames=st.predict(selection,name);vectors=st.w.evaluate(frames,return_vectors=True)[name];y=np.array([[c in r['tags']for c in classes]for r in st.w.records]);q=np.array([[c in rows[r['uid']]['consumers'][name]['tags']for c in classes]for r in st.w.records])
        if not np.array_equal(q,vectors):raise ValueError('Native whole prediction differs '+name)
        support=np.zeros((len(frames),len(fine)))
        for j,f in enumerate(frames):
            if f['family']=='break':continue
            scores=f['scores'];den=max(1.,max(scores.values()));support[j]=[min(.999,max(0,scores.get(c,0))/den)for c in fine]
        mass=st.w.mass[st.w.consumers.index(name)];share=np.zeros((len(st.w.records),len(fine)));np.add.at(share,st.w.bag,mass[:,None]*support)
        for i,r in enumerate(st.w.records):
            actual=np.array([rows[r['uid']]['consumers'][name]['percentages'][c]for c in fine]);error=max(error,float(abs(actual-100*share[i]).max()))
        details=[local(st.d.records[u],rows[u]['consumers'][name]['segments'])for u in st.d.manual];manual=float(np.mean([r['accuracy']for r in details]));expected=st.d.local(frames)['accuracy']
        if abs(manual-expected)>1e-12:raise ValueError('Native local mismatch '+name)
        whole=metrics(y[st.w.main],q[st.w.main],st.w.weight[st.w.main]);breaks=st.breaks(frames,st.d.manual)
        results[name]=dict(config=selection,whole=whole,manual=details,localAccuracy=manual,breaks=breaks,real=st.d.real(frames),displaySegments=sum(len(rows[st.w.records[i]['uid']]['consumers'][name]['segments'])for i in st.w.main),subsecondSegments=sum(s['end']-s['start']<1000 for i in st.w.main for s in rows[st.w.records[i]['uid']]['consumers'][name]['segments']))
        for u,r in rows.items():
            p=r['consumers'][name]['breakPolicy']
            if p['protectedParents']:changed.append(dict(uid=u,consumer=name,**p))
        print(name,manual,breaks,'whole',[(k,whole[k])for k in ['macroPrecision','macroRecall','macroF1']],flush=True)
    if error>1e-8:raise ValueError('Native burden percentage mismatch '+str(error))
    write('research/break_pressure/results/product-report.json',dict(scope='Actual native consumer with existing Detailed/Compact fields; fixed W/b and old reference predictor. Retrospective repair, not new blind evaluation.',consumers=results,maxPercentagePointError=error,protectedInputs=changed))


if __name__=='__main__':run()
