"""Plot already computed paired comparisons; never reruns model fitting."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render(directory):
    root=Path(directory);data=json.loads((root/'metrics.json').read_text(encoding='utf8'))
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    matched=protocol['id'].startswith(('ln-anchored-release-','ln-extra-gate-'));reference='no_update'if matched else'original_fixed'
    names=[v for v in data['map']if v in data['segment']and v not in ['original_fixed',reference]]
    y=np.arange(len(names));fig,axes=plt.subplots(1,2,figsize=(13,max(5,.35*len(names)+2)),sharey=True)
    for ax,key,metric,title in [(axes[0],'map','macroF1','Whole-chart macro-F1'),
                                (axes[1],'segment','chartMeanAccuracy','Segment accuracy (equal-chart)')]:
        base=data[key][reference][metric]
        for i,v in enumerate(names):
            value=data[key][v];estimate=100*(value[metric]-base)
            field=('pairedGroupBootstrapDeltaVsNoUpdate95'if key=='map'else'pairedChartBootstrapDeltaVsNoUpdate95')if matched else('pairedGroupBootstrapDeltaVsOriginal95'if key=='map'else'pairedChartBootstrapDeltaVsOriginal95')
            interval=np.array(value[field])*100
            # Plot endpoints directly; percentile bootstrap intervals need not
            # contain the point estimate in a sparse-class small dataset.
            ax.hlines(i,interval[0],interval[1],color='#a1b5ba',lw=2)
            ax.plot(estimate,i,'o',ms=5,color='#126878')
        ax.axvline(0,color='#25363c',lw=1,ls='--');ax.set_title(title,loc='left',fontsize=12,pad=15)
        ax.set_xlabel('Difference from '+('matched frozen-prior control'if matched else'unchanged original')+' (percentage points)')
        ax.grid(axis='x',alpha=.17);ax.set_axisbelow(True)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0)
    display_names=[n.replace('pool_','').replace('_unrestricted',' / full').replace('_nested',' / nested')for n in names]
    axes[0].set_yticks(y,display_names,fontsize=10);axes[0].invert_yaxis()
    label='Native joint calibration'if protocol['id'].startswith('native-')else'Cold two-family expert joint fit'if protocol['id'].startswith('family-cold-')else'Fixed-boundary joint fit'
    if protocol['id'].startswith('native-nested-capacity-'):label='Nested policy update capacity'
    if protocol['id']=='native-nested-capacity-union-r1':label='Shared-pool nested policy capacity'
    if matched:label='Anchored LN Release update: matched zero-update control'
    warning=' | bounded r2 optimizer: diagnostic, superseded'if protocol['id']=='7k-mil-fixed-original-segmentation-r2'else''
    if data.get('optimizer',{}).get('accepted') is False:
        warning=' | PROVISIONAL: unaccepted optimizer fits'
    fig.suptitle(label+warning,x=.03,ha='left',fontsize=15,fontweight='bold',y=.97)
    fig.text(.03,.025,'Retrospective grouped CV. Paired 95% bootstrap: song groups / six annotated charts. Historical original-model exposure unknown.',fontsize=9,color='#47555a')
    fig.subplots_adjust(left=.21,right=.98,top=.86,bottom=.15,wspace=.12)
    fig.savefig(root/'paired-effects.png',dpi=170,facecolor='white');fig.savefig(root/'paired-effects.svg',facecolor='white');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');render(p.parse_args().directory)
