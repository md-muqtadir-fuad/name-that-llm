import json,sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
nb=json.load(open(sys.argv[1],encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    src=''.join(c['source']) if isinstance(c['source'],list) else c['source']
    print(f"\n# ===== [{i}] {c['cell_type']} =====")
    print(src)
