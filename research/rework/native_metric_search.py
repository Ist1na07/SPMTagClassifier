"""Jointly select low-capacity segmentation-metric shrinkage and the pipeline.

An explicit catalogue configures the existing one-source-at-a-time evaluator;
the exact wrapper, evaluator, cache producer and parameter mapping are saved.
"""
import argparse,json
from pathlib import Path
import native_followup as study

def configure():
    parameters={};sources=[]
    for release in [None,20,40,80]:
        for strength in [0,.05,.2,1]:
            if strength==0:source='logical_neutral'if release is None else'logical_release_'+str(release)
            elif strength==1:source='logical_neutral_scale'if release is None else'full_'+str(release)
            else:source='metric_'+str(release if release is not None else'none')+'_'+str(round(strength*100))
            sources.append(source);parameters[source]={'releaseMs':release,'metricStrength':strength,'logical5NeutralCoverage':True}
            if strength in[.05,.2]:study.CATALOGUE[source]={'root':'.cache/rework-native-metric-r1','id':'native-metric-r1'}
    parameters['original']={'releaseMs':None,'metricStrength':0.,'logical5NeutralCoverage':False,'role':'matched historical representation control only'}
    study.GROUPS={
      'metric_repaired':{'sources':sources},
      'metric_no_release':{'sources':[s for s in sources if parameters[s]['releaseMs']is None]},
      'metric_old_gain':{'sources':[s for s in sources if parameters[s]['metricStrength']==0]},
      'metric_full_scale':{'sources':[s for s in sources if parameters[s]['metricStrength']==1]},
      'metric_normalized':{'sources':sources,'fixed':{'mode':'normalized'}},
      'metric_soft':{'sources':sources,'fixed':{'mode':'soft'}},
      'metric_no_effort':{'sources':sources,'fixed':{'effortPower':0.}},
      'metric_no_confidence':{'sources':sources,'fixed':{'confidencePower':0.}},
      'metric_no_diversity':{'sources':sources,'fixed':{'minimumDiversity':0.}},
      'metric_original_control':{'sources':['original'],'role':'matched historical representation, not a repaired final candidate'},
      'metric_logical_control':{'sources':['logical_neutral'],'role':'matched fixed old-gain logical representation'},
    }
    study.STUDY_ID='native-metric-repaired-r1'
    study.SCOPE='retrospective refinement after the factorial regression; joint metric-strength/release/boundary/gating/evidence selection; frozen historical coefficients, unknown exposure'
    study.SOURCE_FILES=['native_metric_search.py','native_followup.py','native_search.py','metric.cjs','metric-extract.cjs','mil.py']
    study.SOURCE_PARAMETERS=parameters
    study.EXTRA_CACHE_SPECIFICATIONS={'metric':json.loads(Path('.cache/rework-native-metric-r1/spec.json').read_text(encoding='utf8'))}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups');p.add_argument('--candidates',type=int,default=16384)
    p.add_argument('--output',default='research/rework/results/native-metric-r1');args=p.parse_args()
    configure();study.run(args)
