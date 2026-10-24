SRC = r'''"""E051: parts of speech with meaning. Per passage: UPOS rates; pronoun person and gender; gendered person nouns
(male / female / neutral); sentiment of adjectives, adverbs, verbs and nouns (VADER lexicon valence: positive / negative /
neutral share, mean valence, intensity); whole-passage VADER scores; exclamation, question, interjection, intensifier and
hedge rates. Plain POS n-grams are already in the style member (experiments/style/writeprints.py); this adds the meaning.
Usage: python experiments/pos_affect/pos_affect.py extract   (Kaggle CPU kernel: spaCy en_core_web_sm + vaderSentiment)
       python experiments/pos_affect/pos_affect.py eval      (local: per-model profiles, within-family contrasts, LR OOF,
                                                              added to the strong OOF blend; logs E051)"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MODE = sys.argv[1]
OUT = Path('experiments/pos_affect/out')
OUT.mkdir(parents=True, exist_ok=True)
UPOS = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT', 'SCONJ',
        'SYM', 'VERB', 'X']
PRON = dict(p1s='i me my mine myself', p1p='we us our ours ourselves', p2='you your yours yourself yourselves',
            p3m='he him his himself', p3f='she her hers herself', p3n='it its itself', p3p='they them their theirs themselves')
NOUNS = dict(male='man men boy father dad son brother husband king gentleman uncle nephew grandfather sir mr prince lord '
                  'monk boyfriend groom fisherman',
             female='woman women girl mother mom mum daughter sister wife queen lady aunt niece grandmother madam mrs ms '
                    'princess nun girlfriend bride',
             neutral='person people individual child kid parent sibling spouse partner friend human someone everyone '
                     'anyone citizen student worker reader user customer neighbor neighbour stranger teacher')
INTENS = set('very really truly incredibly extremely deeply absolutely remarkably profoundly utterly quite so totally '
             'genuinely entirely'.split())
HEDGE = set('may might perhaps possibly arguably likely seem seems seemed somewhat often generally typically could '
            'probably suggest suggests'.split())


def extract():
    import spacy
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    vader = SentimentIntensityAnalyzer()
    lex = vader.lexicon
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'lemmatizer'])
    pron = {w: k for k, v in PRON.items() for w in v.split()}
    noun = {w: k for k, v in NOUNS.items() for w in v.split()}
    train = pd.read_csv('data/train.csv', dtype=str)
    rows = []
    for doc, text in zip(nlp.pipe(train.text, batch_size=64), train.text):
        toks = [t for t in doc if not t.is_space]
        words = [t for t in toks if t.pos_ != 'PUNCT']
        n = max(len(words), 1)
        low = [t.text.lower() for t in words]
        f = {f'upos_{p}': sum(t.pos_ == p for t in toks) / max(len(toks), 1) for p in UPOS}
        prons = [pron[w] for w in low if w in pron]
        for k in PRON:
            f[f'pron_{k}'] = prons.count(k) / n
            f[f'pronshare_{k}'] = prons.count(k) / max(len(prons), 1)
        nn_ = [t.text.lower() for t in words if t.pos_ == 'NOUN']
        hits = [noun[w] for w in nn_ if w in noun]
        for k in NOUNS:
            f[f'noun_{k}'] = hits.count(k) / max(len(nn_), 1)
        f['gender_m_minus_f'] = (prons.count('p3m') + hits.count('male') - prons.count('p3f') - hits.count('female')) / n
        for p in ('ADJ', 'ADV', 'VERB', 'NOUN'):
            vals = [lex.get(t.text.lower(), 0.0) for t in words if t.pos_ == p]
            m = max(len(vals), 1)
            f[f'{p}_pos'] = sum(v > 0 for v in vals) / m
            f[f'{p}_neg'] = sum(v < 0 for v in vals) / m
            f[f'{p}_neu'] = sum(v == 0 for v in vals) / m
            f[f'{p}_valence'] = sum(vals) / m
            f[f'{p}_intensity'] = sum(abs(v) for v in vals) / m
        s = vader.polarity_scores(text)
        f.update({f'vader_{k}': v for k, v in s.items()})
        n_sent = max(len(list(doc.sents)), 1)
        f['excl_per_sent'] = text.count('!') / n_sent
        f['quest_per_sent'] = text.count('?') / n_sent
        f['intens'] = sum(w in INTENS for w in low) / n
        f['hedge'] = sum(w in HEDGE for w in low) / n
        f['feeling_words'] = sum(lex.get(w, 0.0) != 0 for w in low) / n
        rows.append(f)
    feats = pd.DataFrame(rows)
    feats.insert(0, 'id', train.id)
    feats.to_csv(OUT / 'features.csv', index=False)
    print('features', feats.shape)


def evaluate():
    import csv
    from scipy.stats import kruskal
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    train = pd.read_csv('data/train.csv', dtype=str)
    feats = pd.read_csv(OUT / 'features.csv').set_index('id').loc[train.id]
    X = feats.to_numpy(float)
    cols = feats.columns.tolist()
    labels = np.array(sorted(train.model.unique()))
    y = np.searchsorted(labels, train.model.to_numpy())
    fam = np.array([m.split('-')[0] for m in labels])
    fam = np.where(np.char.startswith(fam, 'qwen'), 'qwen', fam)
    fold = np.full(len(y), -1)
    for k, (_, va) in enumerate(GroupKFold(5).split(X, y, train.prompt_id)):
        fold[va] = k
    bacc = lambda yt, p: np.mean([(p[yt == j] == j).mean() for j in np.unique(yt)])

    # 1. per-model profile (mean) and z-scores across models
    prof = feats.groupby(train.model.to_numpy()).mean().loc[labels]
    prof.to_csv(OUT / 'profile_mean.csv')
    z = (prof - prof.mean()) / prof.std().replace(0, 1)
    z.to_csv(OUT / 'profile_z.csv')
    # 2. separability: Kruskal-Wallis across all 18 and within each family (epsilon^2 effect size)
    rows = []
    for scope in ['all'] + [f for f in np.unique(fam) if (fam == f).sum() > 1]:
        labs = range(18) if scope == 'all' else np.flatnonzero(fam == scope)
        for j, c in enumerate(cols):
            groups = [X[y == lab, j] for lab in labs]
            if np.ptp(np.concatenate(groups)) == 0:
                continue
            h, p = kruskal(*groups)
            nn_ = sum(len(g) for g in groups)
            rows.append(dict(scope=scope, feature=c, H=h, p=p, eps2=(h - len(groups) + 1) / (nn_ - len(groups))))
    sep = pd.DataFrame(rows).sort_values(['scope', 'eps2'], ascending=[True, False])
    sep.to_csv(OUT / 'separability.csv', index=False)

    # 3. classifier OOF (plain; and prompt-centred, an upper bound since test prompts are only inferred)
    oof = {}
    for name, Xv in (('plain', X), ('centred', X - feats.groupby(train.prompt_id.to_numpy()).transform('mean').to_numpy())):
        o = np.zeros((len(y), 18))
        for k in range(5):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000, class_weight='balanced'))
            clf.fit(Xv[fold != k], y[fold != k])
            o[fold == k] = clf.predict_log_proba(Xv[fold == k])
        oof[name] = o
        print(f'{name}: standalone OOF bacc {bacc(y, o.argmax(1)):.4f}')
    np.savez(OUT / 'oof_pos_affect.npz', ids=train.id.to_numpy(dtype=str), **oof)

    # 4. added to the strong OOF blend; weight chosen on the other 4 folds
    zb = np.load('research/ext_oof/q35heads/oof_predictions.npz', allow_pickle=True)   # our own OOF bundle
    pos = pd.Series(np.arange(len(y)), index=train.id).loc[zb['ids'].astype(str)].to_numpy()
    ref = np.empty_like(zb['reference'])
    ref[:, pos] = zb['reference']
    GRID = [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    res = []
    for s in range(ref.shape[0]):
        score = {w: (ref[s] + w * oof['plain']).argmax(1) for w in GRID}
        pred = np.empty(len(y), dtype=int)
        for k in range(5):
            o = fold != k
            wk = max(GRID, key=lambda w: bacc(y[o], score[w][o]))
            pred[fold == k] = score[wk][fold == k]
            res.append(dict(scenario=s, fold=k, w=wk, base=bacc(y[fold == k], score[0][fold == k]),
                            value=bacc(y[fold == k], pred[fold == k])))
        res.append(dict(scenario=s, fold='all', w=None, base=bacc(y, score[0]), value=bacc(y, pred)))
    r = pd.DataFrame(res)
    r.to_csv(OUT / 'blend.csv', index=False)
    al = r[r.fold == 'all']
    pf = r[r.fold != 'all'].groupby('fold')[['base', 'value']].mean()
    fpos = int((pf.value - pf.base).gt(0).sum())
    print(f'blend {al.base.mean():.4f} -> {al.value.mean():.4f} ({al.value.mean() - al.base.mean():+.4f}, {fpos}/5 folds)')

    # 5. figure: z-scored profile of the most separating features
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    top = sep[sep.scope == 'all'].feature.head(30).tolist()
    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(z[top].T.to_numpy(), cmap='RdBu_r', vmin=-2.5, vmax=2.5, aspect='auto')
    ax.set_xticks(range(18), labels, rotation=60, ha='right', fontsize=8)
    ax.set_yticks(range(len(top)), top, fontsize=8)
    ax.set_title('POS-with-meaning profile per model (z-score across models), 30 most separating features')
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.tight_layout()
    Path('research/pos_affect').mkdir(parents=True, exist_ok=True)
    fig.savefig('research/pos_affect/profile_z.png', dpi=130)

    md = ['# E051 POS with meaning', '',
          f"Standalone OOF bacc: plain {bacc(y, oof['plain'].argmax(1)):.4f}, prompt-centred {bacc(y, oof['centred'].argmax(1)):.4f} (chance 0.056)",
          f'Added to the strong OOF blend: {al.base.mean():.4f} -> {al.value.mean():.4f} '
          f'({al.value.mean() - al.base.mean():+.4f}, {fpos}/5 folds)', '']
    for scope, g in sep.groupby('scope'):
        md.append(f'- **{scope}**: ' + ', '.join(f'{a} ({b:.3f})' for a, b in zip(g.feature.head(6), g.eps2.head(6))))
    (Path('research/pos_affect') / 'SUMMARY.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('\n'.join(md))

    sys.path.insert(0, 'paper')
    from log import LOG, log_row
    keep = [x for x in csv.DictReader(open(LOG, encoding='utf-8')) if not x['id'].startswith('E051')]
    with open(LOG, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(keep[0].keys()))
        w.writeheader()
        w.writerows(keep)
    log_row(id='E051', category='features', metric='oof_bacc_strong_blend_lofo_w',
            description='POS with meaning (pronoun person/gender, gendered nouns, sentiment per POS, VADER, exclamation, '
                        'intensifiers, hedges) -> LR, added to strong OOF blend',
            config='spaCy en_core_web_sm + vaderSentiment; LR C=0.3 balanced', baseline=f'{al.base.mean():.4f}',
            value=f'{al.value.mean():.4f}', folds_pos=f'{fpos}/5', files='experiments/pos_affect/out/*; research/pos_affect/*',
            notes=f"standalone plain {bacc(y, oof['plain'].argmax(1)):.4f}, centred {bacc(y, oof['centred'].argmax(1)):.4f}")


if __name__ == '__main__':
    extract() if MODE == 'extract' else evaluate()
'''
import glob, os, subprocess, sys
from pathlib import Path
os.chdir('/kaggle/working')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'vaderSentiment'], check=True)
try:
    import spacy; spacy.load('en_core_web_sm')
except Exception:
    subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], check=True)
Path('data').mkdir(exist_ok=True)
hit = glob.glob('/kaggle/input/**/name-that-llm/train.csv', recursive=True)
assert hit
if not Path('data/train.csv').exists():
    os.symlink(hit[0], 'data/train.csv')
Path('pos_affect.py').write_text(SRC)
subprocess.run([sys.executable, '-u', 'pos_affect.py', 'extract'], check=True)
os.remove('pos_affect.py')
