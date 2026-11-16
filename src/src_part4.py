# %% [markdown]
# ## 4. Cross-validation results (development patients)

# %%
# ─── 4.1 CV table: mean ± SD over all (seed, fold) runs ───
_REC = {}
def rec_of(name, s, f):
    """Training record with the validation metrics RECOMPUTED from the saved out-of-fold logits, so that
    'no tumour found' (ROI models) counts as abstained, not as glioma (fix of Run 6)."""
    key = (name, s, f)
    if key not in _REC:
        _, rec_path, oof_path = run_paths(name, s, f)
        r = dict(load_json(rec_path)); z = np.load(oof_path); y = np.array(r["val_labels"]); P = softmax_np(z)
        pr = hard(P)
        r.update(val_f1=macro_f1(y, pr), val_acc=float((pr == y).mean()), val_auc=macro_auc(y, P),
                 val_abstain=float((pr < 0).mean()))
        _REC[key] = r
    return _REC[key]

def run_records(name):
    return [rec_of(name, s, f) for s in READY.get(name, []) for f in range(1, N_FOLDS + 1)]

cv_rows = []
for name in MODEL_ORDER:
    recs = run_records(name)
    if not recs:
        continue
    f1s, accs, aucs = [r["val_f1"] for r in recs], [r["val_acc"] for r in recs], [r["val_auc"] for r in recs]
    cv_rows.append({"Model": name, "Seeds": len(READY[name]), "Runs": len(recs),
                    "Macro-F1 (%)": f"{100*np.mean(f1s):.2f} ± {100*np.std(f1s, ddof=1):.2f}",
                    "Accuracy (%)": f"{100*np.mean(accs):.2f} ± {100*np.std(accs, ddof=1):.2f}",
                    "Macro-AUC": f"{np.mean(aucs):.4f} ± {np.std(aucs, ddof=1):.4f}",
                    "Best epoch (median)": int(np.median([r["best_epoch"] for r in recs])),
                    "Abstained (%)": round(100 * np.mean([r["val_abstain"] for r in recs]), 1),
                    "_f1": np.mean(f1s)})
df_cv = pd.DataFrame(cv_rows)
if len(df_cv):
    print(df_cv.drop(columns="_f1").to_string(index=False))
    df_cv.drop(columns="_f1").to_csv(OUT_DIR / "tables" / "table_cv.csv", index=False)

# %%
# ─── 4.2 Corrected resampled t-test (Nadeau & Bengio) vs the proposed model on matched (seed, fold) runs ───
def corrected_ttest(a, b, ratio):
    d = np.asarray(a) - np.asarray(b); J = len(d)
    if J < 3 or d.std(ddof=1) == 0:
        return np.nan, np.nan
    t = d.mean() / math.sqrt((1 / J + ratio) * d.var(ddof=1))
    return float(t), float(2 * sps.t.sf(abs(t), J - 1))

ratio = np.mean([len(FOLDS[k]["val"]) / len(FOLDS[k]["train"]) for k in FOLDS])
cv_stat_rows = []
if PROPOSED in READY:
    for name in READY:
        if name == PROPOSED:
            continue
        pairs = [(s, f) for s in set(READY[name]) & set(READY[PROPOSED]) for f in range(1, N_FOLDS + 1)]
        a = [rec_of(PROPOSED, s, f)["val_f1"] for s, f in pairs]
        b = [rec_of(name, s, f)["val_f1"] for s, f in pairs]
        t, p = corrected_ttest(a, b, ratio)
        cv_stat_rows.append({"Comparison": f"{PROPOSED} vs {name}", "Runs": len(pairs),
                             "ΔF1 (pp)": round(100 * (np.mean(a) - np.mean(b)), 2), "t": t, "p": p})
df_cv_stats = pd.DataFrame(cv_stat_rows)
if len(df_cv_stats):
    ok_ = df_cv_stats.p.notna()
    holm = np.full(len(df_cv_stats), np.nan); pv = df_cv_stats.p[ok_].values; o = np.argsort(pv); m_ = len(pv)
    adj = np.maximum.accumulate([min(1, (m_ - i) * pv[j]) for i, j in enumerate(o)]) if m_ else []
    tmp = np.empty(m_); tmp[o] = adj; holm[ok_.values] = tmp
    df_cv_stats["p (Holm)"] = np.round(holm, 4); df_cv_stats["t"] = df_cv_stats.t.round(2); df_cv_stats["p"] = df_cv_stats.p.round(4)
    print(df_cv_stats.to_string(index=False))
    df_cv_stats.to_csv(OUT_DIR / "tables" / "table_cv_ttest.csv", index=False)

# %%
# ─── 4.3 Learning curves (proposed vs its closest baselines, seed 0) ───
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
for name, col in [(PROPOSED, "#d1495b"), ("CNX-Plain" if PROPOSED == "CTR-Net" else "CNN-Backbone", "#30638e"),
                  ("CTR-noCF" if PROPOSED == "CTR-Net" else "MTK-noSeq", "#888888")]:
    if name not in READY:
        continue
    for f in range(1, N_FOLDS + 1):
        h = load_json(run_paths(name, READY[name][0], f)[1])["history"]
        ax[0].plot(h["train_loss"], color=col, alpha=.5, lw=1, label=name if f == 1 else None)
        ax[1].plot(h["val_f1"], color=col, alpha=.5, lw=1, label=name if f == 1 else None)
ax[0].set(title="Training loss", xlabel="Epoch"); ax[1].set(title="Validation macro-F1", xlabel="Epoch")
ax[1].legend(frameon=False); fig.tight_layout(); fig.savefig(FIG_DIR / "learning_curves.png"); plt.show()

# %% [markdown]
# ## 5. Inference on all evaluation sets (cached)
# Every (model, seed, fold) checkpoint is applied to: internal test, BRISC, Mendeley, BraTS.
# Final predictions = mean softmax over the 5 fold models (and over seeds). No best-fold selection.

# %%
# ─── 5.1 Predictions (only the images that are actually used; cached) ───
EVAL_SETS = {"test": TEST_ITEMS, "brisc": BRISC_CLEAN, "mendeley": MEND_CLEAN,
             "brats": BRATS_ITEMS, "bdneuro": BD_CLEAN, "bfree": BFREE_ITEMS}      # unique images only
EVAL_SETS = {k: v for k, v in EVAL_SETS.items() if len(v)}

def pred_path(name, seed, fold, set_name):
    return OUT_DIR / "preds" / f"{name}_s{seed}_f{fold}_{set_name}.npy"

_ARCH = {"name": None, "model": None}
_HASH_OK = set()

def free_mem():
    gc.collect()
    if DEVICE.type == "cuda": torch.cuda.empty_cache()
    try:
        import ctypes; ctypes.CDLL("libc.so.6").malloc_trim(0)      # give freed RAM back to the system
    except Exception:
        pass

def load_checkpoint(name, seed, fold):
    """The architecture is built once per model name and re-used for every fold / seed."""
    ckpt, rec_path, _ = run_paths(name, seed, fold)
    if ckpt not in _HASH_OK:
        rec = load_json(rec_path)
        if rec.get("sha256") and sha256_file(ckpt) != rec["sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {ckpt}")
        _HASH_OK.add(ckpt)
    if _ARCH["name"] != name:
        _ARCH["model"] = None; free_mem()
        _ARCH["name"], _ARCH["model"] = name, build_model(name, pretrained=False).to(DEVICE)
    m = _ARCH["model"]
    m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True)
    m.zero_grad(set_to_none=True)
    return m.eval()

def cached_ok(path, n):
    try:
        return path.exists() and np.load(path, mmap_mode="r").shape[0] == n
    except Exception:                      # half-written file from a stopped run -> redo it
        return False

if RUN_EVALUATION:
    banner("5.1 Predictions on all evaluation sets",
           "  ".join(f"{k}: {len(v)}" for k, v in EVAL_SETS.items()))
    ARRS = {s: [load_u8(p) for p, _ in items] for s, items in EVAL_SETS.items()}   # read images once
    jobs = [(n, s, f) for n in READY for s in READY[n] for f in range(1, N_FOLDS + 1)]
    todo_jobs = [j for j in jobs if not all(cached_ok(pred_path(*j, s), len(EVAL_SETS[s])) for s in EVAL_SETS)]
    print(f"{len(jobs)} fold-models in total | {len(jobs) - len(todo_jobs)} already cached | {len(todo_jobs)} to predict now\n")
    t_all = time.time()
    for k, (name, seed, fold) in enumerate(todo_jobs, 1):
        t0 = time.time()
        sets = [s for s in EVAL_SETS if not cached_ok(pred_path(name, seed, fold, s), len(EVAL_SETS[s]))]
        model = load_checkpoint(name, seed, fold); mean, std = norm_for(name, fold)
        for s in sets:
            _CUR_SET["name"] = s
            np.save(pred_path(name, seed, fold, s), predict_any(name, fold, model, ARRS[s], mean, std))
        _CUR_SET["name"] = None
        left = (time.time() - t_all) / k * (len(todo_jobs) - k)
        print(f"  [{k:3d}/{len(todo_jobs)}] {name:14s} seed {seed} fold {fold} | {time.time() - t0:4.0f}s "
              f"| about {left / 60:.0f} min left", flush=True)
    free_mem()
    print(f"\n✔ All predictions ready ({(time.time() - t_all) / 60:.1f} min)")

# %%
# ─── 5.2 Temperature scaling (fit per fold model on its own validation fold) ───
def fit_temperature(logits, y):
    grid = np.exp(np.linspace(np.log(0.25), np.log(8), 300))
    nll = [-np.log(softmax_np(logits / t)[np.arange(len(y)), y] + 1e-12).mean() for t in grid]
    return float(grid[int(np.argmin(nll))])

TEMPS = {}
for name in READY:
    for s in READY[name]:
        for f in range(1, N_FOLDS + 1):
            _, rec_path, oof_path = run_paths(name, s, f)
            TEMPS[(name, s, f)] = fit_temperature(np.load(oof_path), np.array(load_json(rec_path)["val_labels"]))

def ensemble_probs(name, set_name, calibrated=False, seeds=None):
    """Mean softmax of the 5 fold models. FAIR COMPARISON: by default only the FIRST seed is used for
    every model (models with 3 seeds are not given a bigger ensemble). seeds="all" uses every seed."""
    seeds = READY[name][:1] if seeds is None else (READY[name] if seeds == "all" else seeds)
    ps = []
    for s in seeds:
        for f in range(1, N_FOLDS + 1):
            z = np.load(pred_path(name, s, f, set_name))
            ps.append(softmax_np(z / (TEMPS[(name, s, f)] if calibrated else 1.0)))
    return np.mean(ps, axis=0)

def labels_of(set_name):
    return np.array([y for _, y in EVAL_SETS[set_name]])

def paths_of(set_name):
    return [p for p, _ in EVAL_SETS[set_name]]

# %% [markdown]
# ## 5.3 Tumour-reliance test: does the model still answer when the tumour is hidden?
# Every Figshare slice is predicted by the fold model that never trained on it (test slices: mean of the 5 fold models),
# in 6 versions. **Main measure = "hidden (swap)"**, a hiding method that CTR-Net never saw during training.
# **Tumour-reliance score** = (accuracy drop when the tumour is hidden) / (accuracy above guessing):
# 0 = the tumour is not needed at all, 1 = without the tumour the model is at guessing level (what a doctor would be).
#
# **Run 6 fix - "unsure" answers:** an ROI model that finds no tumour outputs exactly equal probabilities. These cases are
# now counted as **abstained** (shown in their own column). For balanced accuracy an abstained answer counts as 1/3
# correct (= a random guess); for accuracy / F1 / recall it counts as a miss. (Before Run 6, numpy's argmax silently
# turned every "unsure" into "glioma".)

# %%
# ─── 5.3 Tumour-reliance test (cached per model) ───
REL_CONDS = {"original": None, "hidden (swap)": "swap", "hidden (inpaint)": "inpaint",
             "random spot hidden": "random", "skull / position only": "skull", "tumour only": "tonly"}
REL_DF = df3[df3.filepath.isin(HIDE)].reset_index(drop=True) if HIDE else pd.DataFrame()
REL_Y = REL_DF.label.values if len(REL_DF) else np.array([])
REL_CTRL = np.array([HIDE[p]["has_control"] for p in REL_DF.filepath]) if len(REL_DF) else np.array([])

def bal_acc(y, pred):
    """Balanced accuracy. An ABSTAINED answer (-1, ROI model found no tumour) counts as 1/3 correct = the expected
    value of a random guess, so 'unsure' is neither rewarded (old argmax bug: always glioma) nor punished below chance."""
    pred = np.asarray(pred)
    return float(np.mean([((pred[y == k] == k) + (pred[y == k] == -1) / N_CLASSES).mean()
                          for k in range(N_CLASSES) if (y == k).any()]))

def reliance_probs(name, seed=0):
    """(condition -> (N,3) probabilities) for all REL_DF slices, out-of-fold for development slices."""
    out = {c: np.zeros((len(REL_DF), N_CLASSES), np.float32) for c in REL_CONDS}
    test_idx = np.where(REL_DF.role == "test")[0]
    for f in range(1, N_FOLDS + 1):
        idx = np.concatenate([np.where((REL_DF.role == "dev") & (REL_DF.val_fold == f))[0], test_idx])
        cache = OUT_DIR / "reliance" / f"{name}_s{seed}_f{f}.npz"
        z = np.load(cache) if cache.exists() else None
        if z is None or z["original"].shape[0] != len(idx):
            m = load_checkpoint(name, seed, f); mean, std = norm_for(name, f)
            res = {}
            for c, key in REL_CONDS.items():
                arrs = [load_u8(p) if key is None else HIDE[p][key] for p in REL_DF.filepath.values[idx]]
                res[c] = softmax_np(predict_any(name, f, m, arrs, mean, std))
            cache.parent.mkdir(exist_ok=True); np.savez_compressed(cache, **res); z = res
            del m; gc.collect()
        n_dev = len(idx) - len(test_idx)
        for c in REL_CONDS:
            out[c][idx[:n_dev]] = z[c][:n_dev]
            out[c][test_idx] += z[c][n_dev:] / N_FOLDS
    return out

REL, rel_rows, rel_cls_rows = {}, [], []
if RUN_EVALUATION and RUN_RELIANCE and len(REL_DF):
    banner("5.3 Tumour-reliance test", f"{len(READY)} models x 6 versions of {len(REL_DF)} slices "
           "(first time ~1 min per model, then cached)")
    for name in READY:
        REL[name] = reliance_probs(name, READY[name][0])
        ba = {c: bal_acc(REL_Y, hard(REL[name][c])) for c in REL_CONDS}
        o, h, hi_ = (bal_acc(REL_Y[REL_CTRL], hard(REL[name][c][REL_CTRL]))
                     for c in ["original", "hidden (swap)", "random spot hidden"])
        score = lambda hid: (float(np.clip((ba["original"] - hid) / (ba["original"] - 1 / 3), 0, 1))
                             if ba["original"] > 0.45 else np.nan)       # undefined for a model near guessing
        ent = -(REL[name]["hidden (swap)"] * np.log(REL[name]["hidden (swap)"] + 1e-12)).sum(1).mean() / np.log(3)
        rel_rows.append({"Model": name, **{c: round(100 * v, 1) for c, v in ba.items()},
                         "Control drop (pts)": round(100 * (o - hi_), 1),
                         "Reliance (swap)": round(score(ba["hidden (swap)"]), 2),
                         "Reliance (inpaint)": round(score(ba["hidden (inpaint)"]), 2),
                         "Uncertainty when hidden (0-1)": round(float(ent), 2)})
        pr = hard(REL[name]["hidden (swap)"])
        rel_cls_rows.append({"Model": name, **{f"{c} still correct (%)": round(100 * (pr[REL_Y == k] == k).mean(), 1)
                                               for k, c in enumerate(CLASSES)},
                             "abstained when hidden (%)": round(100 * (pr < 0).mean(), 1)})
    df_rel = pd.DataFrame(rel_rows).sort_values("Reliance (swap)", ascending=False)
    RELIANCE = dict(zip(df_rel.Model, df_rel["Reliance (swap)"]))
    print("Balanced accuracy (%) per version  |  guessing = 33.3%")
    print(df_rel[["Model", "original", "hidden (swap)", "hidden (inpaint)", "random spot hidden",
                  "skull / position only", "Reliance (swap)", "Uncertainty when hidden (0-1)"]].to_string(index=False))
    print("\nHow to read: a model that really uses the tumour falls towards 33% when the tumour is hidden "
          "(reliance close to 1) and becomes uncertain (uncertainty close to 1).")
    print("Control: hiding a random spot of the same size changes accuracy by at most "
          f"{df_rel['Control drop (pts)'].abs().max():.1f} points -> the test is fair.")
    df_rel.to_csv(OUT_DIR / "tables" / "table_reliance.csv", index=False)
    df_relc = pd.DataFrame(rel_cls_rows); df_relc.to_csv(OUT_DIR / "tables" / "table_reliance_by_class.csv", index=False)
    print("\nWith the tumour hidden (swap), share of slices still classified correctly, per class:")
    print(df_relc.set_index("Model").loc[df_rel.Model].to_string())

    order = list(df_rel.Model)
    fig, ax = plt.subplots(figsize=(max(8, 0.75 * len(order)), 3.8))
    xs = np.arange(len(order)); w = 0.22
    for k, (c, col) in enumerate([("original", "#30638e"), ("hidden (swap)", "#d1495b"),
                                  ("random spot hidden", "#bbbbbb"), ("skull / position only", "#edae49")]):
        ax.bar(xs + (k - 1.5) * w, df_rel.set_index("Model").loc[order, c], w, color=col, label=c)
    ax.axhline(100 / 3, color="k", ls="--", lw=1); ax.text(len(order) - 0.5, 35, "guessing", fontsize=7, ha="right")
    ax.set_xticks(xs); ax.set_xticklabels([f"{n}\nR={RELIANCE[n]:.2f}" for n in order], fontsize=7)
    ax.set(ylabel="Balanced accuracy (%)", ylim=(0, 100),
           title="Tumour-hiding test (R = tumour-reliance score; higher = decides from the tumour)")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_reliance_test.png", bbox_inches="tight"); plt.show()
else:
    RELIANCE = {}

# %% [markdown]
# ## 5.4 Is the tumour-reliance score robust? (metric check, experiment 5)
# The main test hides the expert tumour outline + a 9-pixel ring with tissue from the same slice ("swap").
# Here the same test is repeated with **bigger hidden areas** (25 and 41 pixel rings: these also hide the swelling /
# oedema around a tumour) and with a **second hiding method**: the area is filled with tissue from the SAME place in a
# slice of ANOTHER patient (same viewing-plane cluster, brightness matched). If the ranking of the models stays the
# same, the score measures a property of the model, not of our hiding trick.

# %%
# ─── 5.4 Metric check: bigger hidden areas + a second hiding method (cached per model) ───
from scipy.stats import spearmanr
REL2_KEYS = [f"swap {k}" for k in METRIC_SIZES] + ["other patient 9"]
REL2, rel2_rows = {}, []
if RUN_EVALUATION and RUN_RELIANCE and len(REL_DF):
    banner("5.4 Metric check", "building the extra hidden versions once (~3-5 min), then ~1 min per model")
    PID_OF = dict(zip(df3.filepath, df3.pid))
    CLUSTER_OF = dict(zip(df_plane.filepath, df_plane.cluster)) if len(df_plane) else {}
    CL_PATHS = defaultdict(list)
    for p_, c_ in CLUSTER_OF.items():
        if p_ in MASKS: CL_PATHS[c_].append(p_)
    rng5 = np.random.default_rng(5)

    def swap_hide(img, reg, bm):
        sh = shifted_region(reg, bm, rng5)
        if sh is None:
            return None
        dy, dx, _ = sh; ys, xs = np.where(reg); out = img.copy(); out[ys, xs] = img[ys + dy, xs + dx]
        return out

    def other_patient_hide(p, img, reg, bm):
        cands = CL_PATHS.get(CLUSTER_OF.get(p), [])
        for _ in range(40):
            if not cands:
                return None
            q = cands[rng5.integers(len(cands))]
            if PID_OF[q] == PID_OF[p]:
                continue
            if (cv2.dilate(MASKS[q].astype(np.uint8), HIDE_KER, iterations=2).astype(bool) & reg).any():
                continue
            aq = load_u8(q); bq = brain_bm(aq, q)
            if bq[reg].mean() < 0.95:
                continue
            k_ = (np.median(img[bm]) + 1e-3) / (np.median(aq[bq]) + 1e-3)
            out = img.copy(); out[reg] = np.clip(aq[reg].astype(np.float32) * k_, 0, 255).astype(np.uint8)
            return out
        return None

    VAR = {k: [None] * len(REL_DF) for k in REL2_KEYS}
    for i, p in enumerate(tqdm(REL_DF.filepath.values, desc="  building", mininterval=10)):
        img = load_u8(p); bm = brain_bm(img, p)
        for k in METRIC_SIZES:
            reg = cv2.dilate(MASKS[p].astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
            VAR[f"swap {k}"][i] = HIDE[p]["swap"] if (k == 9 and HIDE[p]["has_control"]) else (swap_hide(img, reg, bm) if k != 9 else None)
            if k == 9:
                VAR["other patient 9"][i] = other_patient_hide(p, img, reg, bm)
    AVAIL = np.array([all(VAR[k][i] is not None for k in REL2_KEYS) for i in range(len(REL_DF))])
    print(f"  slices where every hiding version could be made: {AVAIL.sum()} of {len(REL_DF)} "
          f"({100 * AVAIL.mean():.1f}%; big tumours cannot be moved/filled) -> all numbers below use these slices")

    def reliance2_probs(name, seed=0):
        out = {k: np.zeros((len(REL_DF), N_CLASSES), np.float32) for k in REL2_KEYS}
        test_idx = np.where((REL_DF.role == "test").values & AVAIL)[0]
        for f in range(1, N_FOLDS + 1):
            idx = np.concatenate([np.where((REL_DF.role == "dev").values & (REL_DF.val_fold == f).values & AVAIL)[0], test_idx])
            cache = OUT_DIR / "reliance2" / f"{name}_s{seed}_f{f}.npz"
            z = np.load(cache) if cache.exists() else None
            if z is None or z[REL2_KEYS[0]].shape[0] != len(idx):
                m = load_checkpoint(name, seed, f); mean, std = norm_for(name, f); res = {}
                for k in REL2_KEYS:
                    res[k] = softmax_np(predict_any(name, f, m, [VAR[k][i] for i in idx], mean, std))
                cache.parent.mkdir(exist_ok=True); np.savez_compressed(cache, **res); z = res
            n_dev = len(idx) - len(test_idx)
            for k in REL2_KEYS:
                out[k][idx[:n_dev]] = z[k][:n_dev]; out[k][test_idx] += z[k][n_dev:] / N_FOLDS
        return out

    yA = REL_Y[AVAIL]
    for name in READY:
        REL2[name] = reliance2_probs(name, READY[name][0])
        o = bal_acc(yA, hard(REL[name]["original"][AVAIL]))
        row = {"Model": name, "original": round(100 * o, 1)}
        for k in REL2_KEYS:
            h = bal_acc(yA, hard(REL2[name][k][AVAIL]))
            row[f"BA {k}"] = round(100 * h, 1)
            row[f"R {k}"] = round(float(np.clip((o - h) / (o - 1 / 3), 0, 1)), 2) if o > 0.45 else np.nan
        pr = hard(REL2[name]["swap 41"][AVAIL])
        row["glioma still correct, swap 41 (%)"] = round(100 * (pr[yA == 0] == 0).mean(), 1)
        row["abstained, swap 41 (%)"] = round(100 * (pr < 0).mean(), 1)
        rel2_rows.append(row)
    df_rel2 = pd.DataFrame(rel2_rows).sort_values("R swap 9", ascending=False)
    print("Tumour-reliance score R with different hidden areas and a second hiding method (higher = uses the tumour):")
    print(df_rel2[["Model", "original"] + [f"R {k}" for k in REL2_KEYS] + ["glioma still correct, swap 41 (%)"]].to_string(index=False))
    df_rel2.to_csv(OUT_DIR / "tables" / "table_metric_check.csv", index=False)
    ok_ = df_rel2[["R swap 9", "R other patient 9"]].dropna()
    RHO_METRIC = spearmanr(ok_["R swap 9"], ok_["R other patient 9"])[0] if len(ok_) >= 4 else np.nan
    print(f"\nRanking stability: Spearman(swap, other patient) = {RHO_METRIC:.2f}  (criterion S5: >= 0.80)")

    key_models = [n for n in [PROPOSED, "ROI-m06-Transplant", "ROI-Crop-m06", "ROI-Crop", "CNX-Transplant", "CNX-Strip", "CNX-Plain", "CTR-Net", "MTL-Seg",
                              "CAM-Guide", "MTK", "EfficientNet-B0"] if n in REL2]
    key_models = list(dict.fromkeys(key_models))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    d2 = df_rel2.set_index("Model")
    for n in key_models:
        ax[0].plot(METRIC_SIZES, [d2.loc[n, f"R swap {k}"] for k in METRIC_SIZES], "-o", ms=4,
                   lw=2 if n == PROPOSED else 1.2, color=FAMILY_COLOR[family_of(n)], label=n)
    ax[0].set(xlabel="Size of the hidden ring around the tumour (pixels)", ylabel="Tumour-reliance score R",
              ylim=(0, 1.02), xticks=METRIC_SIZES, title="Hiding more of the surroundings (incl. oedema)")
    ax[0].legend(frameon=False, fontsize=6, ncol=2)
    for n in d2.index:
        ax[1].scatter(d2.loc[n, "R swap 9"], d2.loc[n, "R other patient 9"], color=FAMILY_COLOR[family_of(n)], s=28, zorder=3)
        ax[1].annotate(n, (d2.loc[n, "R swap 9"], d2.loc[n, "R other patient 9"]), fontsize=5.5, xytext=(2, 2), textcoords="offset points")
    ax[1].plot([0, 1], [0, 1], color="#999999", ls="--", lw=1)
    ax[1].set(xlabel="R, hidden with tissue from the same slice (main test)", ylabel="R, hidden with another patient's tissue",
              xlim=(0, 1), ylim=(0, 1), title=f"Two hiding methods agree? Spearman ρ = {RHO_METRIC:.2f}")
    fig.suptitle("Metric check: is the tumour-reliance score a property of the MODEL (robust) or of the hiding trick?", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_metric_check.png", bbox_inches="tight"); plt.show()
else:
    df_rel2, RHO_METRIC = pd.DataFrame(), np.nan

# %%
# ─── 5.5 Why do ROI models still name GLIOMAS when the tumour is hidden? (look at what the tumour map finds) ───
ROI_REF = next((n for n in [PROPOSED, "ROI-Crop"] if n in REL and is_roi(n)), None)
if RUN_EVALUATION and ROI_REF and len(REL_DF):
    gi = np.where((REL_Y == 0) & (REL_DF.role == "dev").values & REL_CTRL)[0]
    found_any, inside, outside, rows_g = 0, [], [], []
    rng_g = np.random.default_rng(11); show_g = set(rng_g.choice(gi, min(8, len(gi)), replace=False).tolist())
    maps_show = {}
    for f in range(1, N_FOLDS + 1):
        ii = [i for i in gi if REL_DF.val_fold.values[i] == f]
        if not ii:
            continue
        maps = roi_seg_maps([HIDE[REL_DF.filepath.values[i]]["swap"] for i in ii], f)
        for i, mp in zip(ii, maps):
            b = box_from_mask(mp >= 0.5); reg = HIDE[REL_DF.filepath.values[i]]["region"]
            if b is not None:
                found_any += 1; comp = b[4]
                inside.append((comp & reg).sum() / comp.sum())
                ring = cv2.dilate(reg.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))).astype(bool) & ~reg
                outside.append((comp & ring).sum() / comp.sum())
            if i in show_g:
                maps_show[i] = (mp, b)
    pr_h = hard(REL[ROI_REF]["hidden (swap)"])
    print(f"Glioma slices with the tumour hidden (swap), development folds: {len(gi)}")
    print(f"  the tumour map still finds 'a tumour' in {100 * found_any / max(len(gi), 1):.1f}% of them")
    if inside:
        print(f"  of the found region: {100 * np.mean(inside):.1f}% lies inside the hidden area, "
              f"{100 * np.mean(outside):.1f}% in the 20-pixel ring AROUND it (oedema / mass effect), rest elsewhere")
    print(f"  {ROI_REF} still says 'glioma' for {100 * (pr_h[gi] == 0).mean():.1f}% of them")
    if maps_show:
        ks = sorted(maps_show); fig, ax = plt.subplots(2, len(ks), figsize=(1.9 * len(ks), 4.2), squeeze=False)
        for c, i in enumerate(ks):
            p = REL_DF.filepath.values[i]; mp, b = maps_show[i]
            ax[0, c].imshow(load_u8(p), cmap="gray"); ax[0, c].contour(MASKS[p], levels=[0.5], colors="lime", linewidths=0.7)
            ax[1, c].imshow(HIDE[p]["swap"], cmap="gray")
            ax[1, c].contour(HIDE[p]["region"], levels=[0.5], colors="lime", linewidths=0.6, linestyles="--")
            if b is not None:
                ax[1, c].contour(mp >= 0.5, levels=[0.5], colors="red", linewidths=0.7)
                x, y, w, h, _ = b; s_ = max(w, h) * 1.4
                ax[1, c].add_patch(plt.Rectangle((x + w / 2 - s_ / 2, y + h / 2 - s_ / 2), s_, s_, fill=False, ec="cyan", lw=0.9))
            ax[1, c].set_xlim(0, IMG_SIZE); ax[1, c].set_ylim(IMG_SIZE, 0)
            ax[1, c].set_title(f"→ {IDX2CLASS[int(pr_h[i])][:6]} ({REL[ROI_REF]['hidden (swap)'][i].max():.2f})", fontsize=6,
                               color="green" if pr_h[i] == 0 else "red")
            for a in ax[:, c]: a.axis("off")
        ax[0, 0].set_title("original (green = tumour)", fontsize=6, loc="left")
        fig.suptitle(f"Glioma with the tumour hidden: dashed green = hidden area, red = what the tumour map still finds, "
                     f"cyan = {ROI_REF}'s crop", fontsize=7.5)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_glioma_hidden_roi.png", bbox_inches="tight"); plt.show()


# %% [markdown]
# ## 6. Internal held-out test (35 patients, never used for training or selection)

# %%
# ─── 6.1 Image- and patient-level metrics + patient-clustered bootstrap CIs ───
TEST_Y = labels_of("test") if "test" in EVAL_SETS else np.array([])
TEST_PID = np.array([figshare_pid(p) for p in paths_of("test")]) if len(TEST_Y) else np.array([])
UPID = np.unique(TEST_PID)
PID_IDX = {u: np.where(TEST_PID == u)[0] for u in UPID}

def patient_vote(probs, pid=TEST_PID, y=TEST_Y):
    pids = np.unique(pid); py, pp, pm = [], [], []
    for u in pids:
        idx = np.where(pid == u)[0]
        h_ = hard(probs[idx]); mp = probs[idx].mean(0)
        if (h_ >= 0).any():
            votes = np.bincount(h_[h_ >= 0], minlength=N_CLASSES)
            tied = np.flatnonzero(votes == votes.max()); vote = int(tied[np.argmax(mp[tied])])
        else:
            vote = -1                                   # every slice of this patient abstained
        py.append(y[idx][0]); pp.append(vote); pm.append(mp)
    return np.array(py), np.array(pp), np.array(pm)

def cluster_boot(fn, n=N_BOOT, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        idx = np.concatenate([PID_IDX[u] for u in rng.choice(UPID, len(UPID), replace=True)])
        vals.append(fn(idx))
    return np.nanpercentile(vals, [2.5, 97.5])

TEST_PROBS = {}
test_rows = []
if RUN_EVALUATION and len(TEST_Y):
    for name in READY:
        P = ensemble_probs(name, "test"); TEST_PROBS[name] = P
        pred = hard(P)
        py, pp, pm = patient_vote(P)
        acc_ci = cluster_boot(lambda i: (pred[i] == TEST_Y[i]).mean())
        f1_ci = cluster_boot(lambda i: macro_f1(TEST_Y[i], pred[i]))
        per_seed = [macro_f1(TEST_Y, hard(ensemble_probs(name, "test", seeds=[s]))) for s in READY[name]]
        f1_all = macro_f1(TEST_Y, hard(ensemble_probs(name, "test", seeds="all")))
        test_rows.append({
            "Model": name, "Seeds": len(READY[name]),
            "Image acc (%)": f"{100*(pred==TEST_Y).mean():.2f} [{100*acc_ci[0]:.1f}, {100*acc_ci[1]:.1f}]",
            "Image macro-F1 (%)": f"{100*macro_f1(TEST_Y, pred):.2f} [{100*f1_ci[0]:.1f}, {100*f1_ci[1]:.1f}]",
            "Image macro-AUC": round(macro_auc(TEST_Y, P), 4),
            "Macro-F1 per seed (%)": "; ".join(f"{100*v:.2f}" for v in per_seed),
            "Macro-F1 all-seed ensemble (%)": round(100 * f1_all, 2) if len(per_seed) > 1 else "",
            "Patients correct": f"{int((pp==py).sum())}/{len(py)}",
            "Patient macro-F1 (%)": round(100 * macro_f1(py, pp), 2),
            "Patient macro-AUC": round(macro_auc(py, pm), 4)})
    df_test = pd.DataFrame(test_rows)
    print(df_test.to_string(index=False))
    df_test.to_csv(OUT_DIR / "tables" / "table_test.csv", index=False)
    print("\n[ ] = 95% CI from 2,000 patient-level (cluster) bootstrap resamples. All main columns use seed 0 only "
          "(same ensemble size for every model); extra seeds are shown separately.")

# %%
# ─── 6.2 Paired comparisons vs MTK (patient-clustered bootstrap + sign-flip permutation, Holm) ───
def paired_compare(a, b, metric):
    pa, pb = hard(TEST_PROBS[a]), hard(TEST_PROBS[b])
    f = (lambda y, p: (p == y).mean()) if metric == "acc" else macro_f1
    obs = f(TEST_Y, pa) - f(TEST_Y, pb)
    ci = cluster_boot(lambda i: f(TEST_Y[i], pa[i]) - f(TEST_Y[i], pb[i]))
    d = np.array([(pa[PID_IDX[u]] == TEST_Y[PID_IDX[u]]).mean() - (pb[PID_IDX[u]] == TEST_Y[PID_IDX[u]]).mean()
                  for u in UPID])
    rng = np.random.default_rng(0)
    perm = np.array([(d * rng.choice([-1, 1], len(d))).mean() for _ in range(10000)])
    return obs, ci, float((np.abs(perm) >= abs(d.mean()) - 1e-12).mean())

stat_rows = []
if RUN_EVALUATION and PROPOSED in TEST_PROBS:
    for other in TEST_PROBS:
        if other == PROPOSED:
            continue
        for metric in ["acc", "f1"]:
            obs, ci, p = paired_compare(PROPOSED, other, metric)
            stat_rows.append({"Comparison": f"{PROPOSED} − {other}", "Metric": metric,
                              "Δ (pp)": round(100 * obs, 2), "95% CI": f"[{100*ci[0]:.1f}, {100*ci[1]:.1f}]", "p": p})
    df_stats = pd.DataFrame(stat_rows)
    if len(df_stats):
        for metric in ["acc", "f1"]:
            m = df_stats.Metric == metric
            p = df_stats.loc[m, "p"].values; order = np.argsort(p); adj = np.empty_like(p)
            adj[order] = np.minimum(1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]))
            df_stats.loc[m, "p (Holm)"] = adj
        print(df_stats.to_string(index=False))
        df_stats.to_csv(OUT_DIR / "tables" / "table_test_paired.csv", index=False)

# %%
# ─── 6.3 Class-wise table + figure for the proposed model (image and patient level) ───
if RUN_EVALUATION and PROPOSED in TEST_PROBS:
    P = TEST_PROBS[PROPOSED]; pred = hard(P); py, pp, pm = patient_vote(P)
    rows = []
    for level, yy, pr, pb in [("Image", TEST_Y, pred, P), ("Patient", py, pp, pm)]:
        for k, c in enumerate(CLASSES):
            tp = ((pr == k) & (yy == k)).sum()
            rows.append({"Level": level, "Class": c, "n": int((yy == k).sum()),
                         "Precision (%)": round(100 * tp / max((pr == k).sum(), 1), 1),
                         "Recall (%)": round(100 * tp / max((yy == k).sum(), 1), 1),
                         "AUC": round(roc_auc_score(yy == k, pb[:, k]), 4)})
    df_cls = pd.DataFrame(rows); print(df_cls.to_string(index=False))
    df_cls.to_csv(OUT_DIR / "tables" / "table_test_classwise.csv", index=False)

    fig, ax = plt.subplots(2, 2, figsize=(7, 6.5))
    for c, (lvl, yy, pr, pb) in enumerate([("Image", TEST_Y, pred, P), ("Patient", py, pp, pm)]):
        cm = confusion_matrix(yy, pr, labels=range(N_CLASSES))
        ax[0, c].imshow(cm / cm.sum(1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax[0, c].text(j, i, cm[i, j], ha="center", va="center", fontsize=8)
        ax[0, c].set(xticks=range(3), yticks=range(3), xticklabels=CLASSES, yticklabels=CLASSES,
                     title=f"{lvl} level (n={len(yy)})", xlabel="Predicted", ylabel="True")
        for k, cl in enumerate(CLASSES):
            fpr, tpr, _ = roc_curve(yy == k, pb[:, k])
            ax[1, c].plot(fpr, tpr, color=CLASS_COLORS[cl], label=f"{cl} ({roc_auc_score(yy == k, pb[:, k]):.3f})")
        ax[1, c].plot([0, 1], [0, 1], ":", color="grey"); ax[1, c].legend(frameon=False, fontsize=7)
        ax[1, c].set(xlabel="False positive rate", ylabel="True positive rate")
    fig.suptitle(f"{PROPOSED} on the internal test set", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_test_proposed.png"); plt.show()

# %%
# ─── 6.3b Confusion matrices + ROC curves of the main models (internal test, image and patient level) ───
CMP_MODELS = list(dict.fromkeys(n for n in [PROPOSED, "CNX-Plain", "CTR-Net", "MTL-Seg", "MTK"] if n in TEST_PROBS))[:5]
CMP_MODELS = list(dict.fromkeys(CMP_MODELS))
if RUN_EVALUATION and CMP_MODELS:
    fig, ax = plt.subplots(2, len(CMP_MODELS), figsize=(2.6 * len(CMP_MODELS), 5.2), squeeze=False)
    for c, n in enumerate(CMP_MODELS):
        P = TEST_PROBS[n]; py, pp, _ = patient_vote(P)
        for r, (yy, pr, lvl) in enumerate([(TEST_Y, hard(P), "image"), (py, pp, "patient")]):
            cm = confusion_matrix(yy, pr, labels=range(N_CLASSES))
            ax[r, c].imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
            for i in range(N_CLASSES):
                for j in range(N_CLASSES):
                    ax[r, c].text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                                  color="white" if cm[i, j] > cm.sum(1)[i] * 0.6 else "black")
            ax[r, c].set(xticks=range(3), yticks=range(3), xticklabels=[k[:4] for k in CLASSES],
                         yticklabels=[k[:4] for k in CLASSES] if c == 0 else [],
                         title=f"{n}\n{lvl} level (F1 {100 * macro_f1(yy, pr):.1f}%)")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_confusion_models.png"); plt.show()

    fig, ax = plt.subplots(1, N_CLASSES, figsize=(3.2 * N_CLASSES, 3), sharey=True)
    for k, cl in enumerate(CLASSES):
        for n in CMP_MODELS:
            P = TEST_PROBS[n]; fpr, tpr, _ = roc_curve(TEST_Y == k, P[:, k])
            ax[k].plot(fpr, tpr, lw=1.6 if n == PROPOSED else 1, label=f"{n} ({roc_auc_score(TEST_Y == k, P[:, k]):.3f})")
        ax[k].plot([0, 1], [0, 1], ":", color="grey")
        ax[k].set(title=f"{cl} vs rest (internal test)", xlabel="False positive rate"); ax[k].legend(frameon=False, fontsize=6)
    ax[0].set_ylabel("True positive rate"); fig.tight_layout(); fig.savefig(FIG_DIR / "fig_roc_test.png"); plt.show()

# %%
# ─── 6.4 Ablation evidence: gate weights and prediction agreement ───
if RUN_EVALUATION and len(TEST_Y):
    agree_rows = []
    if PROPOSED in TEST_PROBS:
        _, pp_ref, _ = patient_vote(TEST_PROBS[PROPOSED])
        for other in (["ROI-Mask", "CTR-Net", "CNX-Plain", "MTL-Seg", "RRR", "CAM-Guide"] if PROPOSED != "MTK" else ABLATIONS):
            if other in TEST_PROBS:
                _, pp_o, _ = patient_vote(TEST_PROBS[other])
                agree_rows.append({"Variant": other,
                                   "Slices with different label": int((hard(TEST_PROBS[PROPOSED]) != hard(TEST_PROBS[other])).sum()),
                                   "Patients with different label": int((pp_ref != pp_o).sum())})
        if agree_rows:
            print(pd.DataFrame(agree_rows).to_string(index=False))
            pd.DataFrame(agree_rows).to_csv(OUT_DIR / "tables" / "table_ablation_agreement.csv", index=False)

    gate_rows = []
    for name in [n for n in READY if MODEL_SPECS[n]["kind"] in ("mtk", "tamtk")]:
        probe = build_model(name, pretrained=False)
        if len(probe.paths) < 2:
            continue
        gates = []
        print(f"  gate weights: {name} ({len(READY[name][:1]) * N_FOLDS} fold models)", flush=True)
        for s in READY[name][:1]:
            for f in range(1, N_FOLDS + 1):
                m = load_checkpoint(name, s, f); mean, std = norm_for(name, f); arrs = [load_u8(p) for p in paths_of("test")]
                with torch.no_grad():
                    gates.append(np.concatenate([m(to_batch(arrs[i:i + 128], mean, std), return_gate=True)[1].float().cpu().numpy()
                                                 for i in range(0, len(arrs), 128)]))
                del m
        g = np.mean(gates, 0)
        for k, c in enumerate(CLASSES):
            gate_rows.append({"Model": name, "Class": c, **{f"w_{p}": round(float(g[TEST_Y == k, i].mean()), 3)
                                                             for i, p in enumerate(probe.paths)}})
    if gate_rows:
        print(pd.DataFrame(gate_rows).to_string(index=False))
        pd.DataFrame(gate_rows).to_csv(OUT_DIR / "tables" / "table_gate_weights.csv", index=False)

# %%
# ─── 6.5 Ablation summary: table + figure (CV folds, internal test, external) ───
ABL_SHOW = [n for n in ["ROI-Loc", "ROI-m06-Transplant", "ROI-Crop", "ROI-Crop-m06", "ROI-Crop-m15", "ROI-Mask", "CNX-Transplant", "CNX-Strip", "CTR-Net", "CTR-noCF", "CTR-noSeg", "CTR-MambaKAN", "MTL-Seg", "RRR", "CAM-Guide",
                         "CNX-Plain", "ConvNeXt-Tiny", "MTK", "MTK-noSeq", "MTK-noKAN", "MTK-noCNN", "CNN-Backbone"]
            if n in READY]
if RUN_EVALUATION and len(ABL_SHOW) >= 2:
    params = dict(zip(df_complexity.Model, df_complexity["Params (M)"]))
    rows = []
    for n in ABL_SHOW:
        f1s = [r["val_f1"] for r in run_records(n)]
        row = {"Variant": n, "Params (M)": params.get(n), "CV macro-F1 (%)": round(100 * np.mean(f1s), 2),
               "CV SD": round(100 * np.std(f1s, ddof=1), 2),
               "Test macro-F1 (%)": round(100 * macro_f1(TEST_Y, hard(TEST_PROBS[n])), 2)}
        if "mendeley" in EVAL_SETS:
            keep = np.isin(paths_of("mendeley"), [p for p, _ in MEND_CLEAN])
            row["Mendeley dedup macro-F1 (%)"] = round(100 * macro_f1(labels_of("mendeley")[keep],
                                                                      hard(ensemble_probs(n, "mendeley")[keep])), 2)
        if "brats" in EVAL_SETS:
            yb_ = labels_of("brats"); Pb = ensemble_probs(n, "brats")
            row["BraTS glioma-vs-meningioma AUC"] = round(roc_auc_score(yb_ == 0, Pb[:, 0] / (Pb[:, 0] + Pb[:, 1] + 1e-12)), 4)
        if "bdneuro" in EVAL_SETS and BD_CLEAN:
            keep = np.isin(paths_of("bdneuro"), [p for p, _ in BD_CLEAN])
            row["BDNeuro bal. acc (%)"] = round(100 * bal_acc(labels_of("bdneuro")[keep],
                                                              hard(ensemble_probs(n, "bdneuro")[keep])), 2)
        row["Tumour reliance"] = RELIANCE.get(n, np.nan)
        rows.append(row)
    df_abl = pd.DataFrame(rows)
    print("What each part contributes (seed 0 everywhere; higher is better in every column):")
    print(df_abl.to_string(index=False))
    df_abl.to_csv(OUT_DIR / "tables" / "table_ablation.csv", index=False)

    metrics = [("Test macro-F1 (%)", "Internal test macro-F1 (%)", 1),
               ("BraTS glioma-vs-meningioma AUC", "BraTS AUC (new hospital)", 1),
               ("BDNeuro bal. acc (%)", "BDNeuro balanced acc. (%)", 1),
               ("Tumour reliance", "Tumour reliance (0-1)", 1)]
    metrics = [m for m in metrics if m[0] in df_abl and df_abl[m[0]].notna().any()]
    show = df_abl.set_index("Variant")
    fig, ax = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 0.32 * len(show) + 1.2), sharey=True, squeeze=False)
    for a, (col, title, _) in zip(ax[0], metrics):
        vals = show[col].astype(float)
        a.barh(range(len(show)), vals, color=[FAMILY_COLOR[family_of(n)] for n in show.index])
        lo = np.nanmin(vals) - 0.15 * (np.nanmax(vals) - np.nanmin(vals) + 1e-6)
        a.set(xlim=(max(0, lo), np.nanmax(vals) * 1.03), title=title)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                a.text(v, i, f" {v:.2f}" if v < 2 else f" {v:.1f}", va="center", fontsize=6)
    ax[0][0].set_yticks(range(len(show))); ax[0][0].set_yticklabels(show.index, fontsize=7); ax[0][0].invert_yaxis()
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in FAMILY_COLOR.values()]
    fig.legend(handles, ["proposed", "CTR ablations", "existing guidance methods", "pretrained baselines", "scratch models"],
               frameon=False, fontsize=7, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Ablation and comparison (seed 0)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_ablation.png", bbox_inches="tight"); plt.show()

# %%
# ─── 6.6 Error analysis: which test patients are misclassified, and by which models ───
if RUN_EVALUATION and TEST_PROBS:
    err = {}
    for n in TEST_PROBS:
        py, pp, pm = patient_vote(TEST_PROBS[n])
        err[n] = dict(zip(np.unique(TEST_PID), pp))
    true_p = dict(zip(np.unique(TEST_PID), patient_vote(next(iter(TEST_PROBS.values())))[0]))
    rows = []
    for u in np.unique(TEST_PID):
        wrong = [n for n in err if err[n][u] != true_p[u]]
        if wrong:
            idx = PID_IDX[u]
            rows.append({"Patient": u, "True": IDX2CLASS[true_p[u]], "Slices": len(idx),
                         "Wrong in models": f"{len(wrong)}/{len(err)}",
                         f"{PROPOSED} predicts": IDX2CLASS[err[PROPOSED][u]] if PROPOSED in err else "-",
                         "Models wrong": ", ".join(wrong)})
    df_err = pd.DataFrame(rows).sort_values("Wrong in models", ascending=False) if rows else pd.DataFrame()
    print("Misclassified test patients (patient-level vote):" if rows else "No misclassified test patients.")
    if rows:
        print(df_err.to_string(index=False))
        df_err.to_csv(OUT_DIR / "tables" / "table_error_patients.csv", index=False)
    if PROPOSED in TEST_PROBS:
        cm = confusion_matrix(TEST_Y, hard(TEST_PROBS[PROPOSED]), labels=range(N_CLASSES))
        pairs = [(IDX2CLASS[i], IDX2CLASS[j], int(cm[i, j])) for i in range(3) for j in range(3) if i != j and cm[i, j]]
        print(f"\n{PROPOSED} image-level confusions (true -> predicted):",
              sorted(pairs, key=lambda t: -t[2]))


# %%
# ─── 6.7 Does the tumour-mask head find the tumour? (Dice, internal test + BraTS) ───
def seg_probs(name, items, seed=None):
    """Mean predicted tumour probability map (224x224) over the 5 fold models."""
    seed = READY[name][0] if seed is None else seed
    out = np.zeros((len(items), IMG_SIZE, IMG_SIZE), np.float32)
    for f in range(1, N_FOLDS + 1):
        m = load_checkpoint(name, seed, f); mean, std = norm_for(name, f)
        arrs = [load_u8(p) for p, _ in items]
        with torch.no_grad():
            for i in range(0, len(arrs), 64):
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    _, sl = m(to_batch(arrs[i:i + 64], mean, std), return_seg=True)
                pr = torch.sigmoid(F.interpolate(sl.float(), (IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False))
                out[i:i + 64] += pr[:, 0].cpu().numpy() / N_FOLDS
        del m; gc.collect()
    return out

def dice(a, b):
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-8))

SEG_MAPS = {}
seg_rows = []
SEG_MODEL = (ROI_SEG_SOURCE if is_roi(PROPOSED) and ROI_SEG_SOURCE in READY
             else PROPOSED if MODEL_SPECS[PROPOSED].get("seg") else None)
if RUN_EVALUATION and SEG_MODEL and MASKS:
    banner(f"6.7 Does {SEG_MODEL} find the tumour by itself? (Dice of its predicted tumour map)",
           f"this tumour map is what {PROPOSED} uses to decide where to look" if is_roi(PROPOSED) else "~2-4 min")
    sets = {"Internal test": [(p, y) for p, y in TEST_ITEMS if p in MASKS]}
    if len(BRATS_META):
        sets["BraTS (tumour core)"] = list(zip(BRATS_META.path, BRATS_META.y))
    for set_name, items in sets.items():
        maps = seg_probs(SEG_MODEL, items); SEG_MAPS[set_name] = (items, maps)
        for (p, y), mp in zip(items, maps):
            gt = MASKS[p] if set_name == "Internal test" else load_mask(str(BRATS_DIR / Path(p).parent.parent.name
                                                                          / "masks_tc" / Path(p).name))
            if gt.sum() == 0:
                continue
            yx = np.unravel_index(mp.argmax(), mp.shape)
            seg_rows.append({"Set": set_name, "Class": IDX2CLASS[y], "Dice": dice(mp >= 0.5, gt),
                             "Peak inside tumour": float(gt[yx]), "Tumour area (%)": 100 * float(gt.mean())})
    df_seg = pd.DataFrame(seg_rows)
    tab = df_seg.groupby(["Set", "Class"]).agg(n=("Dice", "size"), Dice=("Dice", "mean"),
                                               Dice_median=("Dice", "median"),
                                               Peak_inside=("Peak inside tumour", "mean")).round(3)
    print(tab.to_string()); tab.to_csv(OUT_DIR / "tables" / "table_segmentation_head.csv")
    print("Note: the mask head is an auxiliary output (28x28), trained only on Figshare; BraTS is a different "
          "scanner/protocol and is skull-stripped.")

# %%
# ─── 6.8 Accuracy vs model size ───
if RUN_EVALUATION and TEST_PROBS:
    params = dict(zip(df_complexity.Model, df_complexity["Params (M)"]))
    fig, ax = plt.subplots(1, 2 if "brats" in EVAL_SETS else 1, figsize=(9, 3.4), squeeze=False)
    for n in TEST_PROBS:
        col = "#d1495b" if n == PROPOSED else ("#30638e" if is_pretrained(n) else "#888888")
        f1 = 100 * macro_f1(TEST_Y, hard(TEST_PROBS[n]))
        ax[0, 0].scatter(params[n], f1, color=col, s=40); ax[0, 0].annotate(n, (params[n], f1), fontsize=6, xytext=(3, 2), textcoords="offset points")
        if "brats" in EVAL_SETS:
            yb_ = labels_of("brats"); Pb = ensemble_probs(n, "brats")
            auc = roc_auc_score(yb_ == 0, Pb[:, 0] / (Pb[:, 0] + Pb[:, 1] + 1e-12))
            ax[0, 1].scatter(params[n], auc, color=col, s=40); ax[0, 1].annotate(n, (params[n], auc), fontsize=6, xytext=(3, 2), textcoords="offset points")
    ax[0, 0].set(xscale="log", xlabel="Parameters (M, log)", ylabel="Internal test macro-F1 (%)", title="Accuracy vs size")
    if "brats" in EVAL_SETS:
        ax[0, 1].set(xscale="log", xlabel="Parameters (M, log)", ylabel="AUC", title="External BraTS glioma-vs-meningioma")
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_accuracy_vs_size.png"); plt.show()

# %%
# ─── 6.9 Is the label given away by the slice PLANE or by the tumour LOCATION? ───
# location-only = only the expert tumour outline's position/shape (no image). Tumour location is a REAL clinical
# cue (pituitary tumours sit at the sella, meningiomas touch the brain surface), so a high score here is legitimate.
# plane-only    = only the viewing plane of the slice (axial/coronal/sagittal). This says nothing about the tumour;
# a high score means the dataset was assembled in a way that leaks the label.
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict

def location_features(p):
    mk = MASKS[p]; img = load_u8(p); bm = brain_mask(img)
    ys, xs = np.where(mk); bys, bxs = np.where(bm) if bm.any() else (ys, xs)
    h = max(bys.max() - bys.min(), 1); w = max(bxs.max() - bxs.min(), 1)
    cy, cx = (ys.mean() - bys.min()) / h, (xs.mean() - bxs.min()) / w
    edge = cv2.distanceTransform(bm.astype(np.uint8), cv2.DIST_L2, 3)
    return [cy, cx, abs(cx - 0.5), mk.mean() / max(bm.mean(), 1e-3), float(edge[mk].min()) / max(h, w),
            (ys.max() - ys.min() + 1) / h, (xs.max() - xs.min() + 1) / w]

if RUN_EVALUATION and MASKS:
    d = df3[df3.filepath.isin(MASKS)].reset_index(drop=True)
    Xl = np.array([location_features(p) for p in d.filepath]); yl = d.label.values
    prl = cross_val_predict(make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)), Xl, yl,
                            groups=d.pid.values, cv=GroupKFold(5))
    rows = [{"Information used (no deep model)": "Tumour location + size only (from expert outline)",
             "Balanced acc (%)": round(100 * bal_acc(yl, prl), 1), "Legitimate clinical cue?": "yes"}]
    if PLANE_OF:
        pl = d.filepath.map(PLANE_OF).values; okp = pl != "mixed"
        tab = pd.crosstab(pd.Series(pl[okp], name="plane"), pd.Series(np.array(CLASSES)[yl[okp]], name="class"))
        print("How many slices of each tumour type come from each viewing plane:")
        print(tab.to_string()); tab.to_csv(OUT_DIR / "tables" / "table_plane_by_class.csv")
        Xp = pd.get_dummies(pd.Series(pl)).values.astype(float)
        prp = cross_val_predict(LogisticRegression(max_iter=5000), Xp[okp], yl[okp], groups=d.pid.values[okp], cv=GroupKFold(5))
        rows.append({"Information used (no deep model)": "Viewing plane only (axial / coronal / sagittal)",
                     "Balanced acc (%)": round(100 * bal_acc(yl[okp], prp), 1), "Legitimate clinical cue?": "no"})
        if RELIANCE:
            REL_PLANE = np.array([PLANE_OF.get(p, "mixed") for p in REL_DF.filepath])
            pr_rows = []
            for n in [m for m in [PROPOSED, "ROI-m06-Transplant", "ROI-Crop", "CNX-Transplant", "CNX-Plain", "CTR-Net", "MTL-Seg", "MTK"] if m in REL]:
                for c in ["original", "hidden (swap)"]:
                    pred = hard(REL[n][c])
                    pr_rows.append({"Model": n, "Version": c, **{pl_: round(100 * bal_acc(REL_Y[REL_PLANE == pl_], pred[REL_PLANE == pl_]), 1)
                                                                 for pl_ in ["axial", "coronal", "sagittal"] if (REL_PLANE == pl_).any()}})
            df_pp = pd.DataFrame(pr_rows)
            print("\nBalanced accuracy (%) per viewing plane, with the tumour visible and hidden:")
            print(df_pp.to_string(index=False)); df_pp.to_csv(OUT_DIR / "tables" / "table_accuracy_by_plane.csv", index=False)
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        tabp = (tab.T / tab.sum(1)).T * 100
        left = np.zeros(len(tabp))
        for c in CLASSES:
            if c in tabp:
                ax[0].barh(tabp.index, tabp[c], left=left, color=CLASS_COLORS[c], label=c); left += tabp[c].values
        ax[0].set(xlabel="% of slices of this plane", title="Tumour type per viewing plane (Figshare)")
        ax[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22))
        if RELIANCE and len(df_pp):
            dh = df_pp[df_pp.Version == "hidden (swap)"].set_index("Model")[[c for c in ["axial", "coronal", "sagittal"] if c in df_pp]]
            dh.T.plot.bar(ax=ax[1], rot=0, width=0.8, color=[FAMILY_COLOR[family_of(n)] for n in dh.index])
            ax[1].axhline(100 / 3, color="k", ls="--", lw=1)
            ax[1].set(ylabel="Balanced accuracy with tumour HIDDEN (%)", ylim=(0, 100), xlabel="",
                      title="Without the tumour, which planes still give the answer?")
            ax[1].legend(frameon=False, fontsize=6)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_plane_analysis.png", bbox_inches="tight"); plt.show()
    rows.append({"Information used (no deep model)": "Brain outline / position only (Figshare, quick check)",
                 "Balanced acc (%)": 67.0, "Legitimate clinical cue?": "no"})
    df_loc = pd.DataFrame(rows); print("\nHow much do simple cues alone reveal the tumour type? (guessing = 33.3%)")
    print(df_loc.to_string(index=False)); df_loc.to_csv(OUT_DIR / "tables" / "table_location_plane_baselines.csv", index=False)
    if not PLANE_OF:
        print("(plane rows appear after PLANE_MAP is filled in cell 0.2)")
