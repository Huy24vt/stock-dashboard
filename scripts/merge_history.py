import subprocess
import pandas as pd
from pathlib import Path
import io

commit = 'fb7cf66'
files = ['data/daily/acb_daily.csv', 'data/daily/fpt_daily.csv', 'data/daily/hpg_daily.csv']

for path in files:
    p = Path(path)
    print('Processing', path)
    try:
        old_bytes = subprocess.check_output(['git', 'show', f'{commit}:{path}'])
        old = pd.read_csv(io.BytesIO(old_bytes))
    except subprocess.CalledProcessError:
        print('No historical file found in commit for', path)
        old = pd.DataFrame()
    except Exception as e:
        print('Error reading historical file for', path, e)
        old = pd.DataFrame()

    try:
        cur = pd.read_csv(p)
    except Exception:
        cur = pd.DataFrame()

    if not old.empty:
        old.columns = [c.strip().lower() for c in old.columns]
        old['date'] = pd.to_datetime(old['date'], errors='coerce')
    if not cur.empty:
        cur.columns = [c.strip().lower() for c in cur.columns]
        cur['date'] = pd.to_datetime(cur['date'], errors='coerce')

    if old.empty and cur.empty:
        print('No data available for', path)
        continue

    merged = pd.concat([old, cur], ignore_index=True)
    if 'date' in merged.columns:
        merged = merged.drop_duplicates(subset=['date']).sort_values('date')
    else:
        merged = merged.drop_duplicates().reset_index(drop=True)

    merged.to_csv(p, index=False, encoding='utf-8-sig')
    print('Wrote', path, 'rows=', len(merged))
