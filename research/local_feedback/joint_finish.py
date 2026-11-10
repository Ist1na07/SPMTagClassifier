"""Finish the finite running study; do not restart any fitter on failure."""
from pathlib import Path
import sys,time,subprocess,json

root=Path('research/local_feedback/results/joint-r7-physical');log=Path('.cache/local-feedback-joint-r7-physical-study.log');begin=time.monotonic()
while not(root/'inner-complete.json').exists():
    if log.exists()and 'Traceback (most recent call last)'in log.read_text(encoding='utf8',errors='replace'):raise RuntimeError('Joint study failed; inspect it before changing or retrying anything')
    if time.monotonic()-begin>6*3600:raise TimeoutError('No complete joint INNER receipt after six hours; no automatic re-run')
    time.sleep(15)
if not(root/'complete.json').exists():
    print('Joint INNER complete; freeze common choices, then consume OUTER',flush=True);subprocess.run([sys.executable,'-X','utf8','research/local_feedback/joint_report.py'],check=True)
print('Joint study/report complete; native/overlay acceptance still required before adoption',flush=True)
