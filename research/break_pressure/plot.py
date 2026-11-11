"""A checkable distribution/time view of one real false-Break repair."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run():
    uid='manual/AA';before=next(r for r in map(json.loads,Path('research/core_retune/results/deployment-refit-r1/config-6-fixed-prior-product.jsonl').read_text(encoding='utf8').splitlines())if r['uid']==uid)
    after=next(r for r in map(json.loads,Path('.cache/break-pressure-product-r1.jsonl').read_text(encoding='utf8').splitlines())if r['uid']==uid)
    source=next(r for r in map(json.loads,Path('.cache/perception-final-resource.jsonl').read_text(encoding='utf8').splitlines())if r['uid']==uid)
    rating=source['ratings']['detail'];q=np.array(rating['query']);d=np.array(rating['demand']);dt=np.diff(q);valid=(dt>0)&(dt<2500);ref=after['consumers']['detail']['breakPolicy']['referenceDemand']
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(11,5.1),layout='constrained');gs=fig.add_gridspec(3,2,height_ratios=[4,1,1],width_ratios=[3,1]);ax=fig.add_subplot(gs[0,0]);hist=fig.add_subplot(gs[0,1]);axes=[fig.add_subplot(gs[i,0],sharex=ax)for i in [1,2]]
    ax.step(q/1000,d,where='post',color='#285f9e',linewidth=1.3);ax.axhline(ref,color='#bc7a00',linestyle='--',label='Active-time median D');ax.set_xlim(154,182);ax.set_ylim(bottom=0);ax.set_ylabel('Raw SPM demand D');ax.set_title('AA: sustained pressure was labelled Break by the low-confidence gate',loc='left',fontsize=12);ax.axvspan(164.351,172.226,color='#f7cfa5',alpha=.4);ax.legend(loc='upper right',fontsize=9)
    hist.hist(d[:-1][valid],weights=dt[valid]/1000,bins=30,orientation='horizontal',color='#b7d0e8');hist.axhline(ref,color='#bc7a00',linestyle='--');hist.set_xlabel('Supported time (s)');hist.set_ylabel('Raw demand D');hist.set_title('Whole-chart distribution',fontsize=10)
    for a,row,label in zip(axes,[before,after],['Before','After']):
        for s in row['consumers']['detail']['segments']:
            if s['end']<=154000 or s['start']>=182000:continue
            left=max(154,s['start']/1000);right=min(182,s['end']/1000);color='#cf605b'if s['tag']=='break'else'#50958b';a.broken_barh([(left,right-left)],(0,1),facecolors=color)
            if right-left>3:a.text((left+right)/2,.5,s['tag'],ha='center',va='center',color='white',fontsize=9)
        a.set_ylim(0,1);a.set_yticks([]);a.set_ylabel(label,rotation=0,labelpad=30);a.spines['left'].set_visible(False)
    axes[-1].set_xlabel('Native chart time (s)');ax.tick_params(labelbottom=False);axes[0].tick_params(labelbottom=False)
    fig.savefig('research/break_pressure/results/AA-demand-break.png',dpi=150);plt.close(fig)


if __name__=='__main__':run()
