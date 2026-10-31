"""Show the complete stored regularization comparison; no additional fitting."""
from pathlib import Path
import argparse,json,itertools
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def run(directory):
    root=Path(directory);protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    complete=json.loads((root/'complete.json').read_text(encoding='utf8'));search=json.loads((root/'search.json').read_text(encoding='utf8'))
    selected=json.loads((root/'selection.json').read_text(encoding='utf8'))['selections'];values=protocol['grid']['C']
    keys=[(r['variant'],r['outer'],json.dumps(r['config'],sort_keys=True))for r in search]
    expected=len(complete['variants'])*6*np.prod([len(v)for v in protocol['grid'].values()])
    if len(keys)!=expected or len(set(keys))!=expected:raise RuntimeError('Missing or repeated joint configuration score')
    profiles={};series={};allvalues=[]
    for name in complete['variants']:
        curves=[];rows=[]
        for outer in range(6):
            curve=[]
            for C in values:
                candidates=[r for r in search if r['variant']==name and r['outer']==outer and r['config']['C']==C]
                best=max(candidates,key=lambda r:(r['macroF1'],r['releaseF1'],r['averagePrecision'],-r['configId']))
                curve.append(best['macroF1']);rows.append({'outer':outer,'C':C,**best})
            curves.append(curve)
        profiles[name]=rows;series[name]=np.array(curves);allvalues.extend(series[name].ravel())
    boundary=Counter('lower'if r['config']['C']==min(values)else'upper'if r['config']['C']==max(values)else'interior'for r in selected)
    (root/'C-profiles.json').write_text(json.dumps({'scope':'stored inner-validation scores only; each C independently reselects class weight, effort pooling and exact threshold partitions',
       'uniqueConfigScores':len(search),'selectedBoundaries':dict(boundary),'profiles':profiles},ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,axes=plt.subplots(4,3,figsize=(14,10),sharex=True,sharey=True);x=np.log10(values)
    for ax,name in zip(axes.ravel(),complete['variants']):
        curves=series[name]
        for curve in curves:ax.plot(x,curve,color='#b7c8cd',lw=.8)
        ax.plot(x,curves.mean(0),color='#126878',lw=2)
        for r in selected:
            if r['variant']==name:ax.plot(np.log10(r['config']['C']),r['macroF1'],'o',ms=3,color='#263b43')
        ax.set_title(name,loc='left',fontsize=10);ax.grid(alpha=.12);ax.spines[['top','right']].set_visible(False)
        ax.set_xticks([-7,-4,-1,2,5]);ax.set_ylim(max(0,min(allvalues)-.03),min(1,max(allvalues)+.03))
    for ax in axes[-1]:ax.set_xlabel('log10(C)')
    for ax in axes[:,0]:ax.set_ylabel('Inner binary macro-F1')
    fig.suptitle('Release diagnostic: complete retuned regularization profiles',x=.035,ha='left',fontsize=14,fontweight='bold')
    fig.text(.035,.035,f"Thin lines: six outer-training scopes; dark line: their mean; dots: inner-selected C. Not outer performance or confidence intervals.\n{len(search):,} distinct joint configuration scores; boundary choices: {dict(boundary)}. Existing predictions only; no new fits.",fontsize=9,color='#47555a')
    fig.subplots_adjust(left=.07,right=.98,top=.91,bottom=.12,hspace=.35,wspace=.12);fig.savefig(root/'C-profiles.png',dpi=160,facecolor='white');plt.close(fig)
    print('complete stored profiles',len(search),dict(boundary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');run(p.parse_args().directory)
