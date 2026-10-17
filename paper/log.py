"""Append one row to paper/experiments.csv.
Usage: python paper/log.py --id E032 --cat blend --desc "..." --base 0.4915 --value 0.5238 --folds 5/5
       [--metric oof_bacc] [--ci "0.01,0.05"] [--config "w=0.5"] [--files "experiments/x.py;out/x.log"] [--notes "..."]
Also importable: from paper.log import log_row"""
import argparse
import csv
import datetime as dt
import pathlib

LOG = pathlib.Path(__file__).resolve().parent / 'experiments.csv'
FIELDS = ['date', 'id', 'category', 'description', 'config', 'metric', 'baseline', 'value', 'delta', 'folds_pos',
          'ci95', 'files', 'notes']


def log_row(**kw):
    new = not LOG.exists()
    row = {k: kw.get(k, '') for k in FIELDS}
    row['date'] = row['date'] or dt.date.today().isoformat()
    if row['baseline'] not in ('', None) and row['value'] not in ('', None):
        try:
            row['delta'] = f"{float(row['value']) - float(row['baseline']):+.4f}"
        except ValueError:
            pass
    with open(LOG, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    return row


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('id', 'cat', 'desc', 'config', 'metric', 'base', 'value', 'folds', 'ci', 'files', 'notes', 'date'):
        ap.add_argument(f'--{k}', default='')
    a = ap.parse_args()
    print(log_row(id=a.id, category=a.cat, description=a.desc, config=a.config, metric=a.metric or 'oof_bacc',
                  baseline=a.base, value=a.value, folds_pos=a.folds, ci95=a.ci, files=a.files, notes=a.notes,
                  date=a.date))
