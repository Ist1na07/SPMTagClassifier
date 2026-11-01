"""Unlabeled density/context follow-ups. OD stays8; no speed mod is used."""
import json
import math
from pathlib import Path
import struct
import wave
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def cases():
    lanes=[0,1,5,6]; out={}
    for rate in [2,3,4,6]:
        n=64*rate
        out[f'C{8*rate}']={'notes':[(round(1000+i*125/rate),lanes[i%4])for i in range(n)],
            'labelInterval':[1000,9000], 'nominalHeadRate':8*rate,
            'question':'Primary type/mixture and certainty at this density, without DT or an OD change'}
    low=[(1000+125*i,lanes[i%4])for i in range(192)]
    high=[]
    for i in range(192):
        t=1000+125*i
        cs=[lanes[i%4]]if 9000<=t<17000 else([0,2,4,6]if i%2==0 else[1,3,5])
        high.extend((t,c)for c in cs)
    for name,notes in [('CTX-L',low),('CTX-H',high)]:
        out[name]={'notes':notes,'labelInterval':[9000,17000],
            'question':'Label ONLY9–17seconds. Target heads identical; surrounding context differs.'}
    target=lambda rows:[(t,c)for t,c in rows if 9000<=t<17000]
    assert target(low)==target(high)
    return out


def run():
    out=Path('artifacts/frontier-annotation-r2');out.mkdir(parents=True,exist_ok=True);data=cases()
    fig,axes=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
    for ax,(name,r)in zip(axes.flat,data.items()):
        text=(f'osu file format v14\n\n[General]\nAudioFilename:metronome.wav\nMode:3\n\n[Editor]\nBookmarks:9000,17000\n\n'
            f'[Metadata]\nTitle:Frontier density contexts\nArtist:SPM research\nCreator:Unlabeled study\nVersion:{name}\nBeatmapID:-1\nBeatmapSetID:-1\n\n'
            '[Difficulty]\nHPDrainRate:5\nCircleSize:7\nOverallDifficulty:8\nApproachRate:5\nSliderMultiplier:1.4\nSliderTickRate:1\n\n[TimingPoints]\n0,500,4,2,1,100,1,0\n\n[HitObjects]\n')
        text+='\n'.join(f'{int((c+.5)*512/7)},192,{t},1,0,0:0:0:0:'for t,c in sorted(r['notes']))+'\n'
        (out/f'{name}.osu').write_text(text,encoding='utf8')
        lo,hi=(.9,2.1)if name.startswith('C')and not name.startswith('CTX')else(7.,11.)
        ax.set_facecolor('#f2f5f7');ax.axvspan(3,4,color='#e4e7ee')
        if name.startswith('CTX'):ax.axhspan(9,11,color='#f2d593',alpha=.45)
        for c in range(8):ax.axvline(c,color='#ccd5dc',lw=.6)
        for t,c in r['notes']:
            if lo<=t/1000<=hi:ax.add_patch(Rectangle((c+.12,t/1000-.004),.76,.008,
                facecolor='#b24f73'if c<3 else'#456fa0',zorder=3))
        ax.set(xlim=(0,7),ylim=(hi,lo),xticks=[i+.5 for i in range(7)],xticklabels=range(1,8),ylabel='Time (seconds)')
        ax.set_title(name+(f" | {r['nominalHeadRate']} notes/s"if'nominalHeadRate'in r else' | target begins at9s'),loc='left',fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Same roll, different density/context — fixed OD8 and no DT',fontsize=15,fontweight='bold')
    fig.savefig(out/'preview.png',dpi=160,facecolor='white');plt.close(fig)
    rate=22050
    with wave.open(str(out/'metronome.wav'),'wb')as w:
        w.setnchannels(1);w.setsampwidth(2);w.setframerate(rate);samples=bytearray()
        for i in range(rate*27):
            t=i/rate;phase=t%.5;a=.16*max(0.,1-phase/.025)if phase<.025 else 0.
            samples.extend(struct.pack('<h',round(32767*a*math.sin(2*math.pi*880*t))))
        w.writeframes(samples)
    manifest={'status':'unlabeled follow-up, no model predictions shown','keys':7,'OD':8,'speedRate':1,
        'timing':'source head times rounded to integer milliseconds; nominal phrase rate stated explicitly',
        'cases':{name:{k:v for k,v in r.items()if k!='notes'}for name,r in data.items()}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    with zipfile.ZipFile(out/'frontier-density-context-r2.osz','w',zipfile.ZIP_DEFLATED)as z:
        for name in [*(n+'.osu'for n in data),'metronome.wav']:z.write(out/name,name)
    print(out.resolve())


if __name__=='__main__':run()
