"""
RTO scorecard (Weight of Evidence + logistic regression) on real Indian e-commerce orders.

Data: "Amazon Sale Report.csv" (Amazon.in seller report, Mar-Jun 2022, ~129k orders),
public on Kaggle ("Unlock Profits with E-commerce Sales Data") and mirrored on GitHub.

Label: only merchant-shipped (Easy Ship) orders carry a final courier outcome.
  RTO = 1  -> "Shipped - Returned to Seller", "Shipped - Returning to Seller", "Shipped - Rejected by Buyer"
  RTO = 0  -> "Shipped - Delivered to Buyer"
Validation: train on Mar-May, test on June (out-of-time), the way a scorecard is validated in production.
"""
import json, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

import sys
SRC = sys.argv[1] if len(sys.argv) > 1 else 'Amazon Sale Report.csv'   # download from Kaggle: 'Unlock Profits with E-commerce Sales Data'
d = pd.read_csv(SRC, low_memory=False)
d['dt'] = pd.to_datetime(d.Date, format='mixed')
m = d[d.Fulfilment == 'Merchant']
RTO = ['Shipped - Returned to Seller', 'Shipped - Returning to Seller', 'Shipped - Rejected by Buyer']
f = m[m.Status.isin(RTO + ['Shipped - Delivered to Buyer'])].copy()
f['y'] = f.Status.isin(RTO).astype(int)

# ---------- raw features ----------
f['state'] = f['ship-state'].astype(str).str.upper().str.strip().replace({'PONDICHERRY': 'PUDUCHERRY', 'RJ': 'RAJASTHAN', 'NEW DELHI': 'DELHI', 'ORISSA': 'ODISHA'})
f['pin'] = f['ship-postal-code'].fillna(0).astype(int).astype(str)
f['pin3'] = f.pin.str[:3]
METRO = {'MUMBAI', 'BENGALURU', 'BANGALORE', 'NEW DELHI', 'DELHI', 'CHENNAI', 'HYDERABAD', 'KOLKATA', 'PUNE', 'AHMEDABAD'}
f['city_u'] = f['ship-city'].astype(str).str.upper().str.strip()
f['metro'] = np.where(f.city_u.isin(METRO), 'Metro city', 'Non-metro')
f['dom'] = f.dt.dt.day
f['month_part'] = pd.cut(f.dom, [0, 7, 15, 23, 31], labels=['Day 1-7', 'Day 8-15', 'Day 16-23', 'Day 24-31']).astype(str)
f['weekday'] = np.where(f.dt.dt.dayofweek >= 5, 'Weekend', 'Weekday')
f['size_g'] = f.Size.replace({'4XL': '4XL+', '5XL': '4XL+', '6XL': '4XL+'})
f['cat_g'] = f.Category.replace({'Shoes': 'Accessories', 'Socks': 'Accessories', 'Wallet': 'Accessories', 'Perfume': 'Accessories', 'Watch': 'Accessories'})

train = f[f.dt < '2022-06-01'].copy()
test = f[f.dt >= '2022-06-01'].copy()
base_tr = train.y.mean()

# Historical area RTO rate, computed from TRAIN ONLY and shrunk toward the parent area
def smoothed_rate(tr, key, parent_rate, k):
    g = tr.groupby(key).y.agg(['sum', 'count'])
    return g, lambda keys, parents: [(g.loc[x, 'sum'] + k * p) / (g.loc[x, 'count'] + k) if x in g.index else p for x, p in zip(keys, parents)]

st = train.groupby('state').y.agg(['sum', 'count'])
K = 50
def state_rate(s): return (st.loc[s, 'sum'] + K * base_tr) / (st.loc[s, 'count'] + K) if s in st.index else base_tr
p3 = train.groupby('pin3').y.agg(['sum', 'count'])
def pin3_rate(p, s): pr = state_rate(s); return (p3.loc[p, 'sum'] + K * pr) / (p3.loc[p, 'count'] + K) if p in p3.index else pr

# Out-of-fold area rates for train rows (so a row never sees its own label), plain lookup for test
rng = np.random.default_rng(0)
train['fold'] = rng.integers(0, 5, len(train))
train['area_rate'] = np.nan
for k in range(5):
    tr_k = train[train.fold != k]; b_k = tr_k.y.mean()
    st_k = tr_k.groupby('state').y.agg(['sum', 'count']); p3_k = tr_k.groupby('pin3').y.agg(['sum', 'count'])
    def sr(s): return (st_k.loc[s, 'sum'] + K * b_k) / (st_k.loc[s, 'count'] + K) if s in st_k.index else b_k
    idx = train.fold == k
    train.loc[idx, 'area_rate'] = [((p3_k.loc[p, 'sum'] + K * sr(s)) / (p3_k.loc[p, 'count'] + K)) if p in p3_k.index else sr(s) for p, s in zip(train.loc[idx, 'pin3'], train.loc[idx, 'state'])]
test['area_rate'] = [pin3_rate(p, s) for p, s in zip(test.pin3, test.state)]

# ---------- binning ----------
AREA_BINS = [0, 0.05, 0.065, 0.08, 0.10, 1]
AREA_LBL = ['< 5%', '5-6.5%', '6.5-8%', '8-10%', '> 10%']
AMT_BINS = [-1, 399, 549, 699, 824, 1e9]
AMT_LBL = ['≤ ₹399', '₹400-549', '₹550-699', '₹700-824', '> ₹824']
for df in (train, test):
    df['area_bin'] = pd.cut(df.area_rate, AREA_BINS, labels=AREA_LBL).astype(str)
    df['amt_bin'] = pd.cut(df.Amount.fillna(0), AMT_BINS, labels=AMT_LBL).astype(str)

FEATS = {
    'area_bin': 'Area (pin-3) historical RTO rate',
    'metro': 'Metro vs non-metro city',
    'month_part': 'Day of month (payday cycle)',
    'size_g': 'Size',
    'cat_g': 'Category',
    'amt_bin': 'Order value',
    'weekday': 'Weekday vs weekend',
}
ORDER = {'area_bin': AREA_LBL, 'amt_bin': AMT_LBL, 'month_part': ['Day 1-7', 'Day 8-15', 'Day 16-23', 'Day 24-31'],
         'size_g': ['XS', 'S', 'M', 'L', 'XL', 'XXL', '3XL', '4XL+', 'Free']}

# ---------- WoE + IV ----------
G, B = (train.y == 0).sum(), (train.y == 1).sum()
woe_tables, iv = {}, {}
for c in FEATS:
    t = train.groupby(c).y.agg(bad='sum', n='count'); t['good'] = t.n - t.bad
    pg = (t.good + 0.5) / (G + 0.5 * len(t)); pb = (t.bad + 0.5) / (B + 0.5 * len(t))
    t['woe'] = np.log(pg / pb)                    # WoE = ln(%non-RTO / %RTO); negative = riskier
    t['iv'] = (pg - pb) * t.woe
    t['rto'] = t.bad / t.n
    if c in ORDER: t = t.reindex([x for x in ORDER[c] if x in t.index])
    woe_tables[c] = t; iv[c] = float(t.iv.sum())

def woe_x(df, cols):
    return np.column_stack([df[c].map(woe_tables[c].woe).fillna(0).values for c in cols])

# keep features with IV >= 0.02 (standard "weak" threshold)
KEEP = [c for c in FEATS if iv[c] >= 0.02]
Xtr, Xte = woe_x(train, KEEP), woe_x(test, KEEP)
lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xtr, train.y)
p_tr, p_te = lr.predict_proba(Xtr)[:, 1], lr.predict_proba(Xte)[:, 1]

def evaluate(y, p):
    y = np.asarray(y); o = np.argsort(-p, kind='stable'); ys = y[o]; n = len(y)
    auc = roc_auc_score(y, p)
    cum_b = np.cumsum(ys) / ys.sum(); cum_g = np.cumsum(1 - ys) / (1 - ys).sum(); ks = float(np.max(cum_b - cum_g))
    dec = []
    for i in range(10):
        a, b = i * n // 10, (i + 1) * n // 10
        dec.append({'d': i + 1, 'n': int(b - a), 'rto': float(ys[a:b].mean()), 'pred': float(p[o][a:b].mean())})
    gains = [[k / 100, float(cum_b[max(0, k * n // 100 - 1)])] for k in range(0, 101)]; gains[0] = [0, 0]
    return {'auc': auc, 'ks': ks, 'base': float(y.mean()), 'dec': dec, 'gains': gains, 'top10_capture': float(cum_b[n // 10 - 1]), 'n': int(n), 'rtos': int(y.sum())}

ev_te, ev_tr = evaluate(test.y, p_te), evaluate(train.y, p_tr)

# bootstrap CI for test AUC
bs = []
yt = test.y.values
r2 = np.random.default_rng(1)
for _ in range(300):
    i = r2.integers(0, len(yt), len(yt))
    if yt[i].sum() > 0: bs.append(roc_auc_score(yt[i], p_te[i]))
ev_te['auc_ci'] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]

# ---------- scorecard points (PDO scaling) ----------
# Score 600 = odds 50:1 (delivered:RTO); every 20 points doubles the odds of delivery.
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50
factor = PDO / np.log(2); offset = BASE_SCORE - factor * np.log(BASE_ODDS)
b0, betas = lr.intercept_[0], lr.coef_[0]
nf = len(KEEP)
points = {}
for c, bta in zip(KEEP, betas):
    t = woe_tables[c]
    # log-odds of RTO = b0 + sum(beta*woe); score is on log-odds of delivery, so flip sign
    points[c] = {str(k): int(round(-(bta * w + b0 / nf) * factor + offset / nf)) for k, w in t.woe.items()}

# ---------- worked example ----------
ex = {'area_bin': '> 10%', 'metro': 'Non-metro', 'month_part': 'Day 24-31', 'size_g': 'XS', 'cat_g': 'T-shirt', 'amt_bin': '₹550-699', 'weekday': 'Weekday'}
x = np.array([[woe_tables[c].woe.get(ex[c], 0) for c in KEEP]]); p_ex = float(lr.predict_proba(x)[0, 1])
ex_pts = {c: points[c].get(ex[c]) for c in KEEP}

out = {
    'source': 'Amazon Sale Report.csv (Amazon.in seller report, Mar-Jun 2022)',
    'rows_total': int(len(d)), 'rows_labelled': int(len(f)), 'rto_rate': float(f.y.mean()),
    'train': {'n': int(len(train)), 'rtos': int(train.y.sum()), 'period': 'Mar 31 - May 31 2022'},
    'test': {'n': int(len(test)), 'rtos': int(test.y.sum()), 'period': 'Jun 1 - Jun 29 2022'},
    'features': [{'k': c, 'label': FEATS[c], 'iv': iv[c], 'kept': c in KEEP,
                  'bins': [{'bin': str(k), 'n': int(r.n), 'rto': float(r.rto), 'woe': float(r.woe), 'pts': points.get(c, {}).get(str(k))} for k, r in woe_tables[c].iterrows()]} for c in FEATS],
    'coef': {c: float(b) for c, b in zip(KEEP, betas)}, 'intercept': float(b0),
    'test_eval': ev_te, 'train_eval': {k: ev_tr[k] for k in ('auc', 'ks', 'top10_capture')},
    'example': {'profile': ex, 'p': p_ex, 'points': ex_pts, 'score': int(sum(ex_pts.values()))},
    'scaling': {'pdo': PDO, 'base_score': BASE_SCORE, 'base_odds': BASE_ODDS},
}
json.dump(out, open('data/amazon_scorecard_results.json', 'w'), indent=1, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ('rows_labelled', 'rto_rate', 'train', 'test', 'coef', 'intercept', 'train_eval', 'example')}, indent=1, ensure_ascii=False))
print('IV', {c: round(v, 4) for c, v in iv.items()})
print('TEST auc %.3f ci %s ks %.3f top10 capture %.3f' % (ev_te['auc'], ev_te['auc_ci'], ev_te['ks'], ev_te['top10_capture']))
print([(x['d'], round(x['rto'] * 100, 1), round(x['pred'] * 100, 1)) for x in ev_te['dec']])
for c in KEEP: print(c, points[c])
