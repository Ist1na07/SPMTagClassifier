"""Bounded continuation: consume the running INNER study, never restart it."""
from pathlib import Path
import sys,time,subprocess

root=Path('research/local_feedback/results/activity-r6');log=Path('.cache/local-feedback-activity-r6-study.log');begin=time.monotonic()
while not(root/'inner-complete.json').exists():
    if log.exists()and 'Traceback (most recent call last)'in log.read_text(encoding='utf8',errors='replace'):
        raise RuntimeError('INNER study failed; inspect original log instead of repeating the fit')
    if time.monotonic()-begin>6*3600:raise TimeoutError('No completed INNER receipt after six hours; no automatic re-run')
    time.sleep(15)
if not(root/'complete.json').exists():
    print('INNER complete; freezing common choices and starting OUTER/ablation report',flush=True)
    subprocess.run([sys.executable,'-X','utf8','research/local_feedback/activity_report.py'],check=True)
print('R6 study and report complete. Product adoption still requires acceptance.',flush=True)
