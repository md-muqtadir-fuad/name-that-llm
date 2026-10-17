"""Summarise screen CSVs: per (source, algo, q) means over seeds; best q per (source, algo) by downstream gain.
usage: python summarize.py <tag>[,<tag>...] [top_n]"""
import sys

import numpy as np
import pandas as pd

from common import OUT

tags = sys.argv[1].split(',')
top = int(sys.argv[2]) if len(sys.argv) > 2 else 40
d = pd.concat([pd.read_csv(OUT / f'screen_{t}.csv') for t in tags], ignore_index=True)
d['gA'] = d['rec_A'] - d['plain_A']
d['gB'] = d['rec_B'] - d['plain_B']
d['g'] = (d['gA'] + d['gB']) / 2
d['true_g'] = (d['true_A'] - d['plain_A'] + d['true_B'] - d['plain_B']) / 2
d['q'] = d['q'].fillna(-1)
key = ['regime', 'source', 'algo', 'q']
agg = d.groupby(key).agg(n=('seed', 'nunique'), g=('g', 'mean'), g_sd=('g', 'std'), gA=('gA', 'mean'), gB=('gB', 'mean'),
                         true_g=('true_g', 'mean'), thr=('thr', 'mean'), P_t=('P_t', 'mean'), R_t=('R_t', 'mean'),
                         F1_t=('F1_t', 'mean'), P_x=('P_x', 'mean'), R_x=('R_x', 'mean'), F1_x=('F1_x', 'mean'),
                         grp_t=('grp_t', 'mean'), pure_t=('pure_t', 'mean')).reset_index()
agg['g_se'] = agg['g_sd'] / np.sqrt(agg['n'])
best = agg.sort_values('g', ascending=False).groupby(['regime', 'source', 'algo']).head(1)
pd.set_option('display.width', 250)
pd.set_option('display.max_rows', 500)
cols = ['regime', 'source', 'algo', 'q', 'thr', 'n', 'g', 'g_se', 'gA', 'gB', 'true_g', 'P_t', 'R_t', 'F1_t', 'P_x', 'R_x',
        'F1_x', 'grp_t', 'pure_t']
print('== best threshold per (source, algo), by downstream gain')
print(best.sort_values('g', ascending=False)[cols].head(top).round(4).to_string(index=False))
bf = agg.sort_values('F1_x', ascending=False).groupby(['regime', 'source', 'algo']).head(1)
print('\n== best F1 on extra-api rows per (source, algo)')
print(bf.sort_values('F1_x', ascending=False)[cols].head(top).round(4).to_string(index=False))
agg.to_csv(OUT / f'summary_{"_".join(tags)}.csv', index=False)
