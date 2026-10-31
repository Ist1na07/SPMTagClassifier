"""Paired ablation effects against the predeclared shared main pipeline."""
from coarse_search import read,write
from coarse_data import load_headers
from coarse_model import Path,np,FINE
from coarse_report import macro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse


def fit_accounting(parent):
    assert parent['variants']['sequence44']==parent['variants']['no_release']and parent['configs']['sequence44']==parent['configs']['no_release']
    folds=sum(len(p['inner'])for p in parent['plans'])
    identities=sum(len(c)for c in parent['configs'].values())*folds
    equivalent=len(parent['configs']['no_release'])*folds
    return {'storedVariantFoldConfigFitIdentities':identities,'repeatedEquivalentSpecFoldConfigs':equivalent,
      'distinctEffectiveTrainingProblems':identities-equivalent,'equivalentArms':['sequence44','no_release'],
      'meaning':'variant/fold/config identities, not independent statistical sample size; identical-spec arms do not provide independent replication; fresh runs may alias completed fits'}


def run(directory='research/rework/results/coarse-supported-all-r1',parent_directory='research/rework/results/coarse-shared-r3'):
    root=Path(directory);complete=read(root/'complete.json');data=read(root/'metrics.json');weak,manual,_=load_headers()
    rng=np.random.default_rng(20261101);groups={K:sorted({r['group']for r in weak if r['keys']==K})for K in[4,7]}
    draws={K:rng.multinomial(len(g),np.full(len(g),1/len(g)),size=2000)for K,g in groups.items()};md=rng.multinomial(6,np.full(6,1/6),size=2000)/6
    boots={};values={};effects={}
    for variant in complete['variants']:
        rows={r['uid']:r for r in read(root/variant/'outer-predictions.json')if r['variant']=='both'};boots[variant]={};values[variant]={}
        for K in[4,7]:
            records=[r for r in weak if r['keys']==K];y=np.eye(5)[[FINE.index(r['coarseTags'][0])for r in records]];p=np.zeros_like(y)
            for j,r in enumerate(records):
                predicted=rows[r['uid']]['predicted']
                if predicted is not None:p[j,FINE.index(predicted)]=1
            w=np.array([r['recordWeight']for r in records]);index={g:i for i,g in enumerate(groups[K])};gi=np.array([index[r['group']]for r in records]);block=[]
            for x in [y*p,(1-y)*p,y*(1-p)]:
                b=np.zeros((len(groups[K]),5));np.add.at(b,gi,x*w[:,None]);block.append(b)
            boots[variant][str(K)]=macro(*(draws[K]@b for b in block));values[variant][str(K)]=data[variant]['metrics']['both'][str(K)]['macroF1']
        a=np.array([r['accuracy']for r in data[variant]['metrics']['both']['segment']['charts']]);boots[variant]['segment']=md@a;values[variant]['segment']=float(a.mean())
    for variant in values:
        effects[variant]={k:{'delta':v-values['full53'][k],'delta95':np.quantile(boots[variant][k]-boots['full53'][k],[.025,.975]).tolist()}for k,v in values[variant].items()}
    write(root/'ablation-paired-effects.json',{'primary':'full53/both','scope':'matched actual outer outputs; source-control changes include supervision/selection, not pure feature effects','effects':effects})
    # This duplication is visible in the design: sequence44 and no_release are
    # the same effective spec, not two independent confirmations of a feature.
    write(root/'effective-fit-accounting.json',fit_accounting(read(Path(parent_directory)/'protocol.json')))
    names=list(values);fig,axes=plt.subplots(1,3,figsize=(15,9),sharey=True)
    for ax,k,title in zip(axes,['4','7','segment'],['4K macro-F1','7K coarse macro-F1','7K coarse segment accuracy']):
        for j,v in enumerate(names):
            e=effects[v][k];lo,hi=np.array(e['delta95'])*100;ax.hlines(j,lo,hi,color='#9eb7ba',lw=2);ax.plot(e['delta']*100,j,'o',color='#126878')
        ax.axvline(0,color='#555',ls='--',lw=1);ax.grid(axis='x',alpha=.15);ax.set_title(title,loc='left');ax.set_xlabel('Difference from full53/both (pp)');ax.spines[['top','right','left']].set_visible(False)
    axes[0].set_yticks(range(len(names)),names);axes[0].invert_yaxis();fig.suptitle('Complete coarse ablations: every arm retuned',x=.03,ha='left',weight='bold')
    fig.text(.03,.025,'Paired 95% bootstrap: cross-key song groups / six annotated charts. All arms use both sufficient templates; 6K has no human pattern truth.',fontsize=9)
    fig.subplots_adjust(left=.19,right=.98,top=.88,bottom=.12,wspace=.25);fig.savefig(root/'ablation-paired-effects.png',dpi=160,facecolor='white');plt.close(fig)
    print({'main':values['full53'],'pure4':values['weak4_only'],'pairedPure4VsMain':effects['weak4_only']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',nargs='?',default='research/rework/results/coarse-supported-all-r1')
    p.add_argument('--parent',default='research/rework/results/coarse-shared-r3');a=p.parse_args();run(a.directory,a.parent)
