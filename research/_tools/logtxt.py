import json,sys,re
sys.stdout.reconfigure(encoding='utf-8',errors='replace')
raw=open(sys.argv[1],encoding='utf-8',errors='replace').read()
try: ev=json.loads(raw)
except Exception: ev=json.loads(re.sub(r',\s*\]$',']',raw.strip()))
for e in ev:
    if e.get('stream_name')=='stdout': print(f"[{e['time']:.0f}s] "+e['data'].rstrip('\n'))
print('LAST TIME', ev[-1]['time'])
