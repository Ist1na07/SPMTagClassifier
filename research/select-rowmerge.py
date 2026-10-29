"""Compare validation choices only; preserve the original acquisition convention on ties."""
import json
from pathlib import Path
report={}
for key in [4,7]:
 main=json.loads(Path(f'research/results/{key}k-selection.json').read_text(encoding='utf-8'))
 winner=main['winner'];score=main['variants'][winner]['validationMacroF1'];row={'rowMergeMs':10,'variant':winner,'validationMacroF1':score}
 compared=[row.copy()]
 for merge in [0,5]:
  alt=json.loads(Path(f'research/results/rowmerge{merge}/{key}k-selection.json').read_text(encoding='utf-8'))
  s=alt['variants']['full']['validationMacroF1'];compared.append({'rowMergeMs':merge,'variant':'full','validationMacroF1':s})
  if s>score:
   profile=Path(f'.cache/models-merge{merge}/{key}k.json').read_text(encoding='utf-8')
   Path(f'src/models/{key}k.json').write_text(profile,encoding='utf-8');score=s;row=compared[-1].copy()
 report[str(key)]={'selected':row,'compared':compared,'selectionData':'validation only, original 10 ms wins exact ties'}
Path('research/results/rowmerge-selection.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
