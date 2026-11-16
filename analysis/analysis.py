"""Recompute all Results statistics with ROI-0.6+TT (ROI-m06-Transplant) as reference.
Input: results/predictions (Run 6 export), data/ (lists), results/run6_tables. Output: results/*.csv. Run from the repository root: python analysis/analysis.py"""
import json, numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
E = ROOT / 'results/predictions'; T = ROOT / 'results/run6_tables'; D = ROOT / 'data'
O = ROOT / 'results'; O.mkdir(parents=True, exist_ok=True)
K = 3; C = {'glioma': 0, 'meningioma': 1, 'pituitary': 2}
REF = 'ROI-m06-Transplant'
NAMES = {  # Run 6 name -> paper name (order = Table 2)
    'CNX-Plain': 'Baseline', 'ResNet-50': 'ResNet-50', 'EfficientNet-B0': 'EfficientNet-B0',
    'ConvNeXt-Tiny': 'ConvNeXt-T (standard head)', 'DeiT-Small': 'DeiT-S',
    'MTL-Seg': 'MTL-Seg', 'RRR': 'RRR', 'CAM-Guide': 'CAM-Guide', 'CTR-Net': 'CTR-Net',
    'ROI-Crop': 'ROI-0.2', 'ROI-Crop-m06': 'ROI-0.6', 'ROI-Crop-m15': 'ROI-1.5', 'ROI-Mask': 'ROI-Mask',
    'ROI-Loc': 'ROI-Loc', 'CNX-Transplant': 'TT', 'ROI-m06-Transplant': 'ROI-0.6+TT', 'CNX-Strip': 'Baseline-Strip'}
MODELS = list(NAMES)
SEEDS = {n: sorted({int(p.name.split('_s')[1].split('_')[0]) for p in (E / 'runs').glob(f'{n}_s*_f1.json')}) for n in MODELS}

def sm(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def hard(P):
    p = P.argmax(1); p[(P.max(1) - P.min(1)) < 1e-6] = -1; return p
def macro_f1(y, p):
    f = []
    for k in range(K):
        tp = ((p == k) & (y == k)).sum(); fp = ((p == k) & (y != k)).sum(); fn = ((p != k) & (y == k)).sum()
        if (y == k).sum() == 0: continue
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))
def bal_acc(y, p):
    return float(np.mean([((p[y == k] == k) + (p[y == k] == -1) / K).mean() for k in range(K) if (y == k).any()]))
KEEP = {}
def _ld(n, sd, f, s):
    z = np.load(E / 'preds' / f'{n}_s{sd}_f{f}_{s}.npy')
    if s in KEEP and len(z) == len(KEEP[s]): z = z[KEEP[s]]     # older cache holds all images; keep the deduplicated ones
    return z
def ens(n, s, seeds=(0,), T_=None):
    return np.mean([sm(_ld(n, sd, f, s) / (T_[(n, sd, f)] if T_ else 1)) for sd in seeds for f in range(1, 6)], 0)
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m)
    adj[o] = np.minimum(1, np.maximum.accumulate([(m - i) * p[j] for i, j in enumerate(o)])); return adj
def boot_p(v):
    v = np.asarray(v); return float(min(1, 2 * min((v <= 0).mean(), (v >= 0).mean())))

# ---------- item lists ----------
sp = pd.read_csv(D / 'split_3class.csv')
te = sp[sp.split == 'test'].reset_index(drop=True); yt = te.label.values; pid_t = te.pid.astype(str).values
br = pd.read_csv(D / 'manifest_brats.csv'); br = br[br.tc_pixels >= 100].reset_index(drop=True)
yb = br.label.map(C).values; pid_b = br.patient_id.astype(str).values; ds_b = br.dataset.values
bf = pd.read_csv(D / 'manifest_free.csv')
ov = pd.read_csv(D / 'overlap_audit.csv')
def clean(target):
    rows = ov[(ov.target == target) & (ov.source == 'Figshare')].reset_index(drop=True)
    bad = set(ov[(ov.target == target) & ov.duplicate].tgt_path)
    keep = ~rows.tgt_path.isin(bad).values
    return keep, rows.tgt_class.map(C).values[keep]
EXT = {s: clean(t) for s, t in [('bdneuro', 'BDNeuro'), ('mendeley', 'Mendeley'), ('brisc', 'BRISC')]}
print({s: int(v[0].sum()) for s, v in EXT.items()}); KEEP.update({s: v[0] for s, v in EXT.items()})

# ---------- temperatures ----------
grid = np.exp(np.linspace(np.log(0.25), np.log(8), 300)); TEMP = {}
for n in MODELS:
    for s in SEEDS[n]:
        for f in range(1, 6):
            z = np.load(E / 'oof' / f'{n}_s{s}_f{f}_logits.npy'); y = np.array(json.load(open(E / 'runs' / f'{n}_s{s}_f{f}.json'))['val_labels'])
            TEMP[(n, s, f)] = grid[np.argmin([-np.log(sm(z / t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid])]

rng_master = 42
# ---------- 1. Cross validation ----------
n_val = sp[sp.split.str.startswith('fold')].groupby('split').size(); ratio = float((n_val / (n_val.sum() - n_val)).mean())
def _rec(n, s, f):   # validation metrics recomputed from out-of-fold logits; abstention = miss (as Run 6 table_cv)
    r = json.load(open(E / 'runs' / f'{n}_s{s}_f{f}.json')); y = np.array(r['val_labels']); P = sm(np.load(E / 'oof' / f'{n}_s{s}_f{f}_logits.npy'))
    return {'val_f1': macro_f1(y, hard(P)), 'val_ba': bal_acc(y, hard(P))}
REC = {n: {(s, f): _rec(n, s, f) for s in SEEDS[n] for f in range(1, 6)} for n in MODELS}
cv_rows = []
for n in MODELS:
    v = np.array([r['val_f1'] for r in REC[n].values()])
    row = {'model': NAMES[n], 'run6': n, 'seeds': len(SEEDS[n]), 'cv_f1_mean': 100 * v.mean(), 'cv_f1_sd': 100 * v.std(ddof=1)}
    cv_rows.append(row)
cv = pd.DataFrame(cv_rows)

# ---------- 2. Internal test ----------
UP = np.unique(pid_t); PIDX = {u: np.where(pid_t == u)[0] for u in UP}
rng = np.random.default_rng(rng_master); BOOT_T = [np.concatenate([PIDX[u] for u in rng.choice(UP, len(UP))]) for _ in range(2000)]
def patient_vote(P):
    py, pp = [], []
    for u in UP:
        i = PIDX[u]; h = hard(P[i]); mp = P[i].mean(0)
        if (h >= 0).any():
            v = np.bincount(h[h >= 0], minlength=K); tied = np.flatnonzero(v == v.max()); vote = int(tied[np.argmax(mp[tied])])
        else: vote = -1
        py.append(yt[i][0]); pp.append(vote)
    return np.array(py), np.array(pp)
TP = {n: ens(n, 'test') for n in MODELS}; TH = {n: hard(TP[n]) for n in MODELS}
test_rows = []
for n in MODELS:
    h = TH[n]; f1b = [macro_f1(yt[i], h[i]) for i in BOOT_T]; py, pp = patient_vote(TP[n])
    row = {'model': NAMES[n], 'test_f1': 100 * macro_f1(yt, h), 'test_f1_lo': 100 * np.percentile(f1b, 2.5), 'test_f1_hi': 100 * np.percentile(f1b, 97.5),
           'test_ba': 100 * bal_acc(yt, h), 'test_abstain': 100 * (h < 0).mean(), 'patients_correct': int((pp == py).sum()), 'patients': len(py),
           'test_f1_all_seeds': 100 * macro_f1(yt, hard(ens(n, 'test', SEEDS[n])))}
    test_rows.append(row)
tst = pd.DataFrame(test_rows)

# ---------- 3. BraTS ----------
gm = np.isin(yb, [0, 1]); UB = np.unique(pid_b[gm]); bidx = {u: np.where(pid_b[gm] == u)[0] for u in UB}
rng = np.random.default_rng(rng_master); BOOT_B = [np.concatenate([bidx[u] for u in rng.choice(UB, len(UB))]) for _ in range(1000)]
def bscore(n):
    P = ens(n, 'brats'); return P, P[gm, 0] / (P[gm, 0] + P[gm, 1] + 1e-12)
ybg = (yb[gm] == 0); BS = {n: bscore(n) for n in MODELS}
br_rows = []
for n in MODELS:
    P, s = BS[n]; h = hard(P); auc = roc_auc_score(ybg, s)
    ab = [roc_auc_score(ybg[i], s[i]) for i in BOOT_B]
    row = {'model': NAMES[n], 'brats_auc': auc, 'brats_auc_lo': np.percentile(ab, 2.5), 'brats_auc_hi': np.percentile(ab, 97.5),
           'gli_recall': 100 * (h[ds_b == 'gli'] == 0).mean(), 'men_recall': 100 * (h[ds_b == 'men'] == 1).mean(),
           'men_to_pit': 100 * (h[ds_b == 'men'] == 2).mean(), 'brats_abstain': 100 * (h < 0).mean()}
    br_rows.append(row)
brt = pd.DataFrame(br_rows)

# ---------- 4. BDNeuro, Mendeley (image bootstrap) ----------
ext_rows = []; EH = {}
for s in ['bdneuro', 'mendeley']:
    keep, y = EXT[s]; rng = np.random.default_rng(rng_master); BI = [rng.integers(0, len(y), len(y)) for _ in range(1000)]
    H = {n: hard(ens(n, s)) for n in MODELS}; EH[s] = (y, H, BI)
    for n in MODELS:
        h = H[n]; row = {'model': NAMES[n], 'set': s, 'n': len(y), 'ba': 100 * bal_acc(y, h), 'f1': 100 * macro_f1(y, h)}
        ext_rows.append(row)
ext = pd.DataFrame(ext_rows)

# ---------- 5. Tumor-free safety ----------
saf_rows = []; FREE = {}
for n in MODELS:
    Pf = ens(n, 'bfree', T_=TEMP); FREE[n] = Pf; Pt = ens(n, 'brats', T_=TEMP); hf = hard(Pf)
    conf_f = np.where(hf < 0, 0, Pf.max(1)); conf_t = np.where(hard(Pt) < 0, 0, Pt.max(1))
    lab = np.r_[np.ones(len(conf_t)), np.zeros(len(conf_f))]
    saf_rows.append({'model': NAMES[n], 'free_confident': 100 * (conf_f >= 0.85).mean(), 'free_abstain': 100 * (hf < 0).mean(),
                     'free_glioma': 100 * (hf == 0).mean(), 'free_meningioma': 100 * (hf == 1).mean(), 'free_pituitary': 100 * (hf == 2).mean(),
                     'auroc_tumor_vs_free': roc_auc_score(lab, np.r_[conf_t, conf_f])})
saf = pd.DataFrame(saf_rows)

# ---------- 6. Tumor reliance with patient bootstrap CI (seed 0, swap operator) ----------
dev = sp.reset_index(drop=True); ydev = dev.label.values; pid_all = dev.pid.astype(str).values
def rel_probs(n):
    out = {}
    tidx = np.where(dev.split == 'test')[0]
    for f in range(1, 6):
        z = np.load(E / 'reliance' / f'{n}_s0_f{f}.npz'); idx = np.concatenate([np.where(dev.split == f'fold{f}')[0], tidx]); nd = len(idx) - len(tidx)
        for c in z.files:
            out.setdefault(c, np.zeros((len(dev), K)))
            out[c][idx[:nd]] = z[c][:nd]; out[c][tidx] += z[c][nd:] / 5
    return out
UA = np.unique(pid_all); AIDX = {u: np.where(pid_all == u)[0] for u in UA}
rng = np.random.default_rng(rng_master); BOOT_A = [np.concatenate([AIDX[u] for u in rng.choice(UA, len(UA))]) for _ in range(1000)]
def rho(y, ho, hh):
    bo, bh = bal_acc(y, ho), bal_acc(y, hh); return float(np.clip((bo - bh) / (bo - 1 / K), 0, 1))
rel_rows = []; RELH = {}
for n in MODELS:
    R = rel_probs(n); RELH[n] = R; ho, hh, hs = hard(R['original']), hard(R['hidden (swap)']), hard(R['skull / position only'])
    rb = [rho(ydev[i], ho[i], hh[i]) for i in BOOT_A]
    rel_rows.append({'model': NAMES[n], 'ba_orig': 100 * bal_acc(ydev, ho), 'ba_hidden': 100 * bal_acc(ydev, hh), 'ba_head_only': 100 * bal_acc(ydev, hs),
                     'ba_random_spot': 100 * bal_acc(ydev, hard(R['random spot hidden'])), 'rho': rho(ydev, ho, hh),
                     'rho_lo': np.percentile(rb, 2.5), 'rho_hi': np.percentile(rb, 97.5),
                     'rho_inpaint': rho(ydev, ho, hard(R['hidden (inpaint)'])),
                     **{f'still_correct_{c}': 100 * (hh[ydev == k] == k).mean() for c, k in C.items()}})
rel = pd.DataFrame(rel_rows)


# ---------- 7. Comparisons: every model vs Baseline; combined model vs its two components ----------
BASE, COMB = 'CNX-Plain', 'ROI-m06-Transplant'
FAMILIES = {'vs_baseline': [(n, BASE) for n in MODELS if n != BASE],
            'combined_vs_components': [(COMB, 'ROI-Crop-m06'), (COMB, 'CNX-Transplant')]}
pid_f = bf.patient_id.astype(str).values; UF = np.unique(pid_f); FIDX = {u: np.where(pid_f == u)[0] for u in UF}
rng = np.random.default_rng(rng_master); BOOT_F = [np.concatenate([FIDX[u] for u in rng.choice(UF, len(UF))]) for _ in range(1000)]
def ci(v): return np.percentile(v, 2.5), np.percentile(v, 97.5)
def conf_free(n):
    P = FREE[n]; return np.where(hard(P) < 0, 0, P.max(1)) >= 0.85
rows = []
for fam, pairs in FAMILIES.items():
    for a, b in pairs:
        # CV: corrected resampled t test on matched (seed, fold) runs
        ks = sorted(set(REC[a]) & set(REC[b])); d = np.array([REC[a][k]['val_f1'] - REC[b][k]['val_f1'] for k in ks]); J = len(d)
        t = d.mean() / np.sqrt((1 / J + ratio) * d.var(ddof=1)); rows.append(dict(family=fam, metric='cv_f1', a=NAMES[a], b=NAMES[b], delta=100 * d.mean(), lo=np.nan, hi=np.nan, p=2 * stats.t.sf(abs(t), J - 1), J=J))
        # internal test macro F1 (patient bootstrap)
        v = [macro_f1(yt[i], TH[a][i]) - macro_f1(yt[i], TH[b][i]) for i in BOOT_T]; lo, hi = ci(v)
        rows.append(dict(family=fam, metric='test_f1', a=NAMES[a], b=NAMES[b], delta=100 * (macro_f1(yt, TH[a]) - macro_f1(yt, TH[b])), lo=100 * lo, hi=100 * hi, p=boot_p(v)))
        # BraTS AUC (patient bootstrap)
        sa, sb = BS[a][1], BS[b][1]; v = [roc_auc_score(ybg[i], sa[i]) - roc_auc_score(ybg[i], sb[i]) for i in BOOT_B]; lo, hi = ci(v)
        rows.append(dict(family=fam, metric='brats_auc', a=NAMES[a], b=NAMES[b], delta=roc_auc_score(ybg, sa) - roc_auc_score(ybg, sb), lo=lo, hi=hi, p=boot_p(v)))
        # BDNeuro / Mendeley balanced accuracy (image bootstrap)
        for s_, (y_, H_, BI_) in EH.items():
            v = [bal_acc(y_[i], H_[a][i]) - bal_acc(y_[i], H_[b][i]) for i in BI_]; lo, hi = ci(v)
            rows.append(dict(family=fam, metric=f'{s_}_ba', a=NAMES[a], b=NAMES[b], delta=100 * (bal_acc(y_, H_[a]) - bal_acc(y_, H_[b])), lo=100 * lo, hi=100 * hi, p=boot_p(v)))
        # tumor-free confident share (patient bootstrap)
        ca, cb = conf_free(a), conf_free(b); v = [ca[i].mean() - cb[i].mean() for i in BOOT_F]; lo, hi = ci(v)
        rows.append(dict(family=fam, metric='free_confident', a=NAMES[a], b=NAMES[b], delta=100 * (ca.mean() - cb.mean()), lo=100 * lo, hi=100 * hi, p=boot_p(v)))
        # tumor reliance (patient bootstrap, all 3,064 slices)
        Ra, Rb = RELH[a], RELH[b]; ha, hha, hb, hhb = hard(Ra['original']), hard(Ra['hidden (swap)']), hard(Rb['original']), hard(Rb['hidden (swap)'])
        v = [rho(ydev[i], ha[i], hha[i]) - rho(ydev[i], hb[i], hhb[i]) for i in BOOT_A]; lo, hi = ci(v)
        rows.append(dict(family=fam, metric='rho', a=NAMES[a], b=NAMES[b], delta=rho(ydev, ha, hha) - rho(ydev, hb, hhb), lo=lo, hi=hi, p=boot_p(v)))
cmp_ = pd.DataFrame(rows)
for (fam, met), g in cmp_.groupby(['family', 'metric']):
    cmp_.loc[g.index, 'p_holm'] = holm(g.p.values)
cmp_.round(5).to_csv(O / 'comparisons.csv', index=False)

for name, d in [('cv', cv), ('test', tst), ('brats', brt), ('external', ext), ('safety', saf), ('reliance', rel)]:
    d.round(4).to_csv(O / f'{name}.csv', index=False)
main = cv.merge(tst, on='model').merge(brt, on='model').merge(saf, on='model').merge(rel, on='model')
main.round(4).to_csv(O / 'main.csv', index=False)
print(main[['model', 'cv_f1_mean', 'test_f1', 'rho', 'brats_auc', 'free_confident']].round(3).to_string())
print(cmp_.round(3).to_string())
