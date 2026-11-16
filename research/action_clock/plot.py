"""Plot observed head rows, per-hand coactive events and musical boundaries."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter
r=json.loads(Path('research/action_clock/results/example-observations.json').read_text(encoding='utf-8'));rows=r['rows'];fig,axes=plt.subplots(3,1,figsize=(12,7),gridspec_kw={'height_ratios':[2,1.1,1.2]},constrained_layout=True)
blue='#2176ae';orange='#d46a1e';gray='#a3a8b0'
for row in rows:
 cols=[i for i in range(7)if row['mask']&(1<<i)];single=len(cols)==1
 axes[0].scatter([row['t']/1000]*len(cols),cols,s=13 if single else 23,color=gray if single else blue,marker='o'if single else's',zorder=3)
axes[0].set_yticks(range(7),['L ring','L middle','L index','Centre','R index','R middle','R ring']);axes[0].set_ylabel('Written head events')
for side,ys in enumerate(r['hands'][0]):
 coactive=[p for p in ys if p['coactive']];axes[1].scatter([p['t']/1000 for p in coactive],[side]*len(coactive),marker='|',s=170,color=blue if side==0 else orange)
axes[1].set_yticks([0,1],['Left hand','Right hand']);axes[1].set_ylim(-.6,1.6);axes[1].set_ylabel('Coactive articulations')
for ax in axes[:2]:
 for bar in r['bars']:ax.axvline(bar['t']/1000,color='#9ab59f',lw=1,ls='--',alpha=.75)
 ax.set_xlim(131.8,142.8);ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15)
axes[0].set_title('Inorganic compound — the 263 BPM alias comes from counting a different action',loc='left',weight='bold')
axes[0].scatter([],[],s=23,marker='s',color=blue,label='Coactive head row');axes[0].scatter([],[],s=13,color=gray,label='Interleaved single');axes[0].legend(loc='upper right',ncol=2,frameon=False)
axes[1].set_xlabel('Source time (s); dashed lines are timing-defined measures')
rows=[p for p in rows if 131980<=p['t']<142582];allgaps=Counter(round(b['t']-a['t'])for a,b in zip(rows,rows[1:]));left=[p for p in r['hands'][0][0]if 131980<=p['t']<142582 and p['coactive']];handgaps=Counter(round(b['t']-a['t'])for a,b in zip(left,left[1:]));xs=sorted(set(allgaps)|set(handgaps));axes[2].bar([x-2 for x in xs],[allgaps[x]for x in xs],width=4,color=gray,label='All head-row gaps');axes[2].bar([x+2 for x in xs],[handgaps[x]for x in xs],width=4,color=blue,label='Left coactive-event gaps')
axes[2].set_xlim(40,250);axes[2].set_xlabel('Inter-event interval (ms)');axes[2].set_ylabel('Count');axes[2].spines[['top','right']].set_visible(False);axes[2].legend(frameon=False,ncol=2,loc='upper right')
axes[2].annotate('~57 ms: inserted articulation\nnot the whole-hand Jack clock',xy=(57,allgaps[57]),xytext=(79,40),arrowprops={'arrowstyle':'->','color':gray},fontsize=10)
axes[2].annotate('~113 ms: supported Jack backbone\n15,000 / 113 ≈ 133 BPM',xy=(113,handgaps[113]),xytext=(145,22),arrowprops={'arrowstyle':'->','color':blue},fontsize=10,color=blue)
fig.savefig('research/action_clock/results/example-actions.png',dpi=160)
