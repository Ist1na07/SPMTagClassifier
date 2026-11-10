"""Lossless bounded checkpoint storage. Numeric state and JSON values unchanged."""
import gzip,json,os
from pathlib import Path


def exists(path):
    path=Path(path);return path.exists()or Path(str(path)+'.gz').exists()


def read(path):
    path=Path(path)
    if path.exists():return json.loads(path.read_text(encoding='utf8'))
    with gzip.open(str(path)+'.gz','rt',encoding='utf8')as f:return json.load(f)


def write(path,value):
    path=Path(str(path)+'.gz');path.parent.mkdir(parents=True,exist_ok=True);temp=Path(str(path)+'.pending')
    with gzip.open(temp,'wt',encoding='utf8',compresslevel=6)as f:json.dump(value,f,ensure_ascii=False,separators=(',',':'))
    # Atomic publication prevents a short write looking like an accepted fit.
    os.replace(temp,path)
