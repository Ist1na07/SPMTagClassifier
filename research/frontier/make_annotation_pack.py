"""Generate four unlabeled contrasts; model predictions are deliberately hidden."""
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
    lanes = [0, 1, 5, 6]
    a = [(1000+125*i, lanes[i % 4], None) for i in range(64)]
    b = [(1000+125*i, lanes[(i//4) % 4], None) for i in range(64)]
    fingers = [0, 1, 2, 4, 5, 6]
    c = [(1000+4000*i, lane, 4000+4000*i) for i in range(4) for lane in fingers]
    offsets = [-375, -250, -125, 125, 250, 375]
    d = [(1000+4000*i, lane, 4000+4000*i+offsets[j])
         for i in range(4) for j, lane in enumerate(fingers)]
    c.append((17000, 3, None)); d.append((17000, 3, None))
    return {'A01': a, 'A02': b, 'B01': c, 'B02': d}


def run():
    out = Path('artifacts/frontier-annotation-r1'); out.mkdir(parents=True, exist_ok=True)
    charts = cases()
    assert sorted(lane for _, lane, _ in charts['A01']) == sorted(lane for _, lane, _ in charts['A02'])
    assert [(t, lane) for t, lane, _ in charts['B01']] == [(t, lane) for t, lane, _ in charts['B02']]
    assert sum(end-t for t, _, end in charts['B01'] if end is not None) == sum(end-t for t, _, end in charts['B02'] if end is not None)
    manifest = {'status': 'unlabeled; no synthetic accuracy claim', 'keys': 7, 'bpm': 120,
        'pairA': 'same head times and column histogram; change only column sequence',
        'pairB': 'same heads, total hold area and chart endpoint; change only release times', 'cases': {}}
    fig, axes = plt.subplots(2, 2, figsize=(10, 11), constrained_layout=True)
    for ax, (name, rows) in zip(axes.flat, charts.items()):
        objects = []
        for start, lane, end in sorted(rows):
            tail = f'{end}:' if end is not None else ''
            objects.append(f'{int((lane+.5)*512/7)},192,{start},{128 if end is not None else 1},0,{tail}0:0:0:0:')
        text = (f'osu file format v14\n\n[General]\nAudioFilename:metronome.wav\nMode:3\n\n[Metadata]\nTitle:Frontier contrasts\nArtist:SPM research\nCreator:Unlabeled study\nVersion:{name}\nBeatmapID:-1\nBeatmapSetID:-1\n\n'
            '[Difficulty]\nHPDrainRate:5\nCircleSize:7\nOverallDifficulty:8\nApproachRate:5\nSliderMultiplier:1.4\nSliderTickRate:1\n\n'
            '[TimingPoints]\n0,500,4,2,1,100,1,0\n\n[HitObjects]\n'+'\n'.join(objects)+'\n')
        (out/f'{name}.osu').write_text(text, encoding='utf8')
        limit = 9.2 if name.startswith('A') else 4.9
        ax.set_facecolor('#f2f5f7')
        ax.axvspan(3, 4, color='#e4e7ee', zorder=0)
        for lane in range(8): ax.axvline(lane, color='#ccd5dc', lw=.7)
        for t in range(int(limit*2)+1): ax.axhline(t/2, color='#dce3e8', lw=.5)
        for start, lane, end in rows:
            if start/1000 > limit: continue
            color = '#b24f73' if lane < 3 else '#456fa0'
            if end is not None:
                ax.add_patch(Rectangle((lane+.16, start/1000), .68, (end-start)/1000,
                    facecolor=color, alpha=.25, edgecolor=color, lw=.8))
                ax.plot([lane+.16, lane+.84], [end/1000]*2, color=color, lw=2)
            ax.add_patch(Rectangle((lane+.1, start/1000-.025), .8, .05, facecolor=color))
        ax.set(xlim=(0, 7), ylim=(limit, 0), xticks=[i+.5 for i in range(7)], xticklabels=range(1, 8), ylabel='Time (seconds)')
        ax.set_title(name+(' | full 8 s phrase' if name.startswith('A') else ' | first cycle; full file has four'), loc='left', fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        manifest['cases'][name] = {'file': name+'.osu', 'objects': len(rows), 'labels': None}
    fig.suptitle('7K contrasts — please label from your own playing experience', fontsize=15, fontweight='bold')
    fig.savefig(out/'preview.png', dpi=180, facecolor='white'); plt.close(fig)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf8')
    form = {name: {'primaryAlternatives': [], 'coexistingAttributes': [], 'wholeTags': [],
        'certainty': None, 'notes': ''} for name in charts}
    if not (out/'label-form.json').exists():
        (out/'label-form.json').write_text(json.dumps(form, indent=2)+'\n', encoding='utf8')
    # Original synthesized click track makes the pack importable/playable.
    rate = 22050
    with wave.open(str(out/'metronome.wav'), 'wb') as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        samples = bytearray()
        for i in range(rate*19):
            t = i/rate; phase = t % .5
            amplitude = .16 * max(0., 1-phase/.025) if phase < .025 else 0.
            samples.extend(struct.pack('<h', round(32767*amplitude*math.sin(2*math.pi*880*t))))
        w.writeframes(samples)
    with zipfile.ZipFile(out/'frontier-contrasts-r1.osz', 'w', zipfile.ZIP_DEFLATED) as z:
        for name in [*(name+'.osu' for name in charts), 'metronome.wav']:
            z.write(out/name, name)
    print(out.resolve())


if __name__ == '__main__':
    run()
