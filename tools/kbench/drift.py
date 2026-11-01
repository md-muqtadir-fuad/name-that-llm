"""Monthly drift panel: run the FROZEN tools/kbench/ntl_gen_task_drift.py (prompts g0003-g0062, range format) as batch
drift_YYYYMM through daily.py's runner, on a copy of the task with only BATCH set. Month 0 = drift_202611 (b001 discarded).
Usage (repo root):  .venv\\Scripts\\python.exe tools\\kbench\\drift.py drift_202611 <model slug> [<model slug> ...] [--dry]"""
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import daily  # noqa: E402

batch, models = sys.argv[1], [m for m in sys.argv[2:] if m != '--dry']
assert re.fullmatch(r'drift_\d{6}', batch), batch
frozen = daily.ROOT / 'tools/kbench/ntl_gen_task_drift.py'
copy = daily.ROOT / f'research/ext_data/kbench/{batch}_task.py'
copy.parent.mkdir(parents=True, exist_ok=True)
copy.write_text(frozen.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
daily.TASK = copy   # set_task only rewrites BATCH (START, N re-set to the frozen 3, 60)
print(daily.run_batch(3, 60, batch, models, '--dry' in sys.argv))
