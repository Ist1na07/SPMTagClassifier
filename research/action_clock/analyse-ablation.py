"""Report actual four native arms and observable tempo coverage, not guessed BPM accuracy."""
from report import *
from collections import defaultdict

def read_arm(name):
 with gzip.open(ROOT/f'ablation-{name}.jsonl.gz','rt',encoding='utf-8')as f:return{r['uid']:r for r in map(json.loads,f)}

def tempo_summary(rows):
 by=defaultdict(lambda:dict(ms=0.,numberMs=0.,segments=0,numberSegments=0))
 for u in main:
  for n in ['actions','detail','compact']:
   for s in rows[u]['consumers'][n]['segments']:
    if s['tag']=='break':continue
    d=by[n+'/'+s['tag']];dt=s['end']-s['start'];d['ms']+=dt;d['segments']+=1
    if s.get('equivBpm',0)>0:d['numberMs']+=dt;d['numberSegments']+=1
 return {k:dict(**v,numberDurationFraction=v['numberMs']/max(1,v['ms']))for k,v in by.items()}

def run():
 arms={n:read_arm(n)for n in ['before','action-only','music-only','combined']};reports={n:report(list(rows.values()))for n,rows in arms.items()};changes=[];by=defaultdict(lambda:dict(commonMs=0.,changedMs=0.,oldFinerOctaveMs=0.,newFinerOctaveMs=0.,oldOnlyMs=0.,newOnlyMs=0.))
 for u in main:
  for n in ['actions','detail','compact']:
   a=arms['before'][u]['consumers'][n]['segments'];b=arms['action-only'][u]['consumers'][n]['segments'];i=j=0
   while i<len(a)and j<len(b):
    x,y=a[i],b[j];dt=max(0,min(x['end'],y['end'])-max(x['start'],y['start']));tag=x['tag'];old=x.get('equivBpm');new=y.get('equivBpm');d=by[n+'/'+tag]
    if dt>0 and tag!='break':
     if old and new:
      d['commonMs']+=dt;d['changedMs']+=dt*(old!=new);d['oldFinerOctaveMs']+=dt*(old/new>1.8);d['newFinerOctaveMs']+=dt*(new/old>1.8)
      if old/new>1.8 or new/old>1.8:changes.append(dict(uid=u,consumer=n,tag=tag,ms=dt,old=old,new=new))
     elif old:d['oldOnlyMs']+=dt
     elif new:d['newOnlyMs']+=dt
    if x['end']<=y['end']:i+=1
    if y['end']<=x['end']:j+=1
 result=dict(scope='Full349 retrospective development metrics, same truth/weighting in all four arms. Cadence statistics are coverage/change audits, NOT correctness rates; only mechanistic constructions and the user example have numeric movement-clock constraints.',
  baseline='b311c408ef02d6705d30b51a4ea789d3e86d2976',researchBaseline='v1.1.0',classes=classes,consumers=reports,
  scalarTempoCoverage={n:tempo_summary(r)for n,r in arms.items()},actionOnlyTempoChanges=dict(by),octaveExamples=sorted(changes,key=lambda x:-x['ms'])[:80],
  metricMeaning='Break durations use final displayed intervals in both arms, including uncovered labelled time as missed. Earlier parent-only Break reports differ slightly; do not subtract cross-definition values.')
 (ROOT/'ablation-report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
 for arm,out in reports.items():print(arm,{n:dict(local=v['manualAccuracy'],P=v['whole']['macroPrecision'],R=v['whole']['macroRecall'],F=v['whole']['macroF1'],real=v['real']['accuracy'],breaks=v['breaks'],display=v['displaySegments'],short=v['subsecondSegments'])for n,v in out.items()},flush=True)
 for n in ['actions','detail','compact']:
  for arm in ['before','action-only']:
   t=result['scalarTempoCoverage'][arm];ms=sum(v['ms']for k,v in t.items()if k.startswith(n+'/'));num=sum(v['numberMs']for k,v in t.items()if k.startswith(n+'/'));print('coverage',n,arm,num/max(1,ms))
 # Self-contained names for the parameter-grid per-class arrays.
 for f in ['joint-report.json','joint-protocol.json']:
  p=ROOT/f;d=json.loads(p.read_text(encoding='utf-8'));d['classes']=classes;p.write_text(json.dumps(d,indent=2),encoding='utf-8')
 grid=json.loads((ROOT/'joint-full-grid.json').read_text(encoding='utf-8'));profiles={}
 for name in ['actions','detail','compact']:
  profiles[name]=[dict(config=r['config'],score=r['consumers'][name]['score'],local=r['consumers'][name]['manualAccuracy'],wholeF1=r['consumers'][name]['whole']['macroF1'])for r in grid if r['config']['radius']==1]
 (ROOT/'parameter-profiles.json').write_text(json.dumps(profiles,indent=2),encoding='utf-8')

if __name__=='__main__':run()
