# %% [markdown]
# ## 7. External evaluation (no fine-tuning)
# BRISC and Mendeley: 3 tumour classes, on **all** images and on **deduplicated** images (no near-duplicate of Figshare;
# for Mendeley also none of BRISC). No patient IDs → image level only.
# BraTS-GLI / BraTS-MEN: one tumour type each → class recall (image and patient level) and glioma-vs-meningioma AUC.

# %%
# ─── 7.1 BRISC + Mendeley ───
ext_rows = []
if RUN_EVALUATION:
    for set_name, clean in [("brisc", BRISC_CLEAN), ("mendeley", MEND_CLEAN)]:
        if set_name not in EVAL_SETS:
            continue
        y_all = labels_of(set_name)
        keep = np.isin(paths_of(set_name), [p for p, _ in clean])
        for name in READY:
            P = ensemble_probs(name, set_name)
            for subset, mask in [("all", np.ones(len(y_all), bool)), ("deduplicated", keep)]:
                y, pr = y_all[mask], P[mask]
                rec = per_class_recall(y, hard(pr))
                present = sorted(np.unique(y))           # deduplicated BRISC has NO glioma left
                ext_rows.append({"Dataset": set_name, "Subset": subset, "n": int(mask.sum()), "Model": name,
                                 "n per class": "/".join(str(int((y == k).sum())) for k in range(N_CLASSES)),
                                 "Acc (%)": round(100 * (hard(pr) == y).mean(), 2),
                                 "Macro-F1 (%)": round(100 * macro_f1(y, hard(pr), labels=present), 2),
                                 "Macro-AUC": round(macro_auc(y, pr), 4),
                                 **{f"Recall {c} (%)": round(100 * v, 1) for c, v in rec.items()}})
    df_ext = pd.DataFrame(ext_rows)
    if len(df_ext):
        print(df_ext.to_string(index=False))
        df_ext.to_csv(OUT_DIR / "tables" / "table_external.csv", index=False)
        print("Macro-F1 / AUC use only the classes present in each subset (BRISC dedup: meningioma + pituitary only, "
              "because every BRISC glioma is a Figshare duplicate).")

# %%
# ─── 7.2 BraTS-GLI / BraTS-MEN ───
def brats_cluster_ci(fn, pids, n=1000, seed=SPLIT_SEED):
    """95% CI by resampling BraTS PATIENTS (all slices of a patient together)."""
    rng = np.random.default_rng(seed); up = np.unique(pids); groups = {u: np.where(pids == u)[0] for u in up}
    vals = []
    for _ in range(n):
        idx = np.concatenate([groups[u] for u in rng.choice(up, len(up), replace=True)])
        try:
            vals.append(fn(idx))
        except ValueError:
            pass
    return np.percentile(vals, [2.5, 97.5])

def brats_auc_scores(name):
    yb = labels_of("brats"); gm = np.isin(yb, [0, 1]); P = ensemble_probs(name, "brats")
    return yb[gm] == 0, P[gm, 0] / (P[gm, 0] + P[gm, 1] + 1e-12), BRATS_META["patient_id"].values[gm]

brats_rows = []
if RUN_EVALUATION and "brats" in EVAL_SETS:
    yb = labels_of("brats"); pid_b = BRATS_META["patient_id"].values; ds = BRATS_META["dataset"].values
    for name in READY:
        P = ensemble_probs(name, "brats"); pred = hard(P)
        for d, cls in [("gli", "glioma"), ("men", "meningioma")]:
            m = ds == d
            if not m.any():
                continue
            k = CLASS2IDX[cls]
            pids = np.unique(pid_b[m])
            def _vote(v):
                v = v[v >= 0]
                return int(np.bincount(v, minlength=N_CLASSES).argmax()) if len(v) else -1
            pat_pred = [_vote(pred[m & (pid_b == u)]) for u in pids]
            dist = np.bincount(pred[m][pred[m] >= 0], minlength=N_CLASSES) / m.sum()
            brats_rows.append({"Model": name, "Subset": f"BraTS-{d.upper()}", "Slices": int(m.sum()), "Patients": len(pids),
                               f"Image recall (%)": round(100 * (pred[m] == k).mean(), 1),
                               f"Patient recall (%)": round(100 * np.mean(np.array(pat_pred) == k), 1),
                               **{f"→{c} (%)": round(100 * dist[j], 1) for j, c in enumerate(CLASSES)},
                               "→abstained (%)": round(100 * (pred[m] < 0).mean(), 1)})
        gm = np.isin(yb, [0, 1])
        score = P[gm, 0] / (P[gm, 0] + P[gm, 1] + 1e-12)
        auc_ci = brats_cluster_ci(lambda i: roc_auc_score(yb[gm][i] == 0, score[i]), pid_b[gm])
        brats_rows.append({"Model": name, "Subset": "GLI + MEN", "Slices": int(gm.sum()),
                           "Glioma-vs-meningioma AUC": round(roc_auc_score(yb[gm] == 0, score), 4),
                           "AUC 95% CI": f"[{auc_ci[0]:.3f}, {auc_ci[1]:.3f}]"})
    df_brats = pd.DataFrame(brats_rows)
    print("BraTS (new hospital, never used for training). Recall = share of slices given the correct tumour type;")
    print("'→ class' = where the slices went; AUC = glioma vs meningioma ranking (CI by patient resampling).")
    print(df_brats.to_string(index=False))
    df_brats.to_csv(OUT_DIR / "tables" / "table_brats.csv", index=False)
    print("Note: BraTS images are skull-stripped; Figshare images are not (acquisition/preprocessing shift).")

# %%
# ─── 7.2b Is the proposed model better on BraTS? (paired, patient-level bootstrap, Holm-corrected) ───
def brats_cluster_boot(fn, pids, n=1000, seed=SPLIT_SEED):
    rng = np.random.default_rng(seed); up = np.unique(pids); groups = {u: np.where(pids == u)[0] for u in up}
    vals = []
    for _ in range(n):
        try:
            vals.append(fn(np.concatenate([groups[u] for u in rng.choice(up, len(up), replace=True)])))
        except ValueError:
            pass
    return np.array(vals)

def holm(pvals):
    pv = np.asarray(pvals, float); o = np.argsort(pv); m_ = len(pv)
    adj = np.maximum.accumulate([min(1.0, (m_ - i) * pv[j]) for i, j in enumerate(o)]) if m_ else np.array([])
    out = np.empty(m_); out[o] = adj; return out

def brats_paired(ref, others):
    yt, s_ref, pids = brats_auc_scores(ref); rows = []
    for other in others:
        _, s_o, _ = brats_auc_scores(other)
        d = roc_auc_score(yt, s_ref) - roc_auc_score(yt, s_o)
        v = brats_cluster_boot(lambda i: roc_auc_score(yt[i], s_ref[i]) - roc_auc_score(yt[i], s_o[i]), pids)
        rows.append({"Comparison": f"{ref} − {other}", "ΔAUC": round(d, 4),
                     "95% CI": f"[{np.percentile(v, 2.5):+.3f}, {np.percentile(v, 97.5):+.3f}]",
                     "p (bootstrap)": round(float(min(1, 2 * min((v <= 0).mean(), (v >= 0).mean()))), 4)})
    df_ = pd.DataFrame(rows)
    if len(df_):
        df_["p (Holm)"] = np.round(holm(df_["p (bootstrap)"].values), 4)
        df_["Significant (Holm)"] = np.where(df_["p (Holm)"] < 0.05, "yes", "no")
    return df_.sort_values("ΔAUC") if len(df_) else df_

if RUN_EVALUATION and "brats" in EVAL_SETS and PROPOSED in READY:
    df_bd_cmp = brats_paired(PROPOSED, [n for n in READY if n != PROPOSED])
    print(f"BraTS glioma-vs-meningioma AUC: {PROPOSED} minus each other model (positive = {PROPOSED} better).")
    print("CI and p from 1,000 resamples of BraTS PATIENTS; Holm-corrected over all comparisons.")
    print(df_bd_cmp.to_string(index=False))
    df_bd_cmp.to_csv(OUT_DIR / "tables" / "table_brats_paired.csv", index=False)

# %%
# ─── 7.2c BDNeuro-MRI (independent Bangladeshi hospital, deduplicated, image level) ───
bd_rows = []
if RUN_EVALUATION and "bdneuro" in EVAL_SETS and BD_CLEAN:
    keep = np.isin(paths_of("bdneuro"), [p for p, _ in BD_CLEAN]); yb_ = labels_of("bdneuro")[keep]
    rng_b = np.random.default_rng(SPLIT_SEED)
    boots = [rng_b.integers(0, len(yb_), len(yb_)) for _ in range(1000)]
    P_prop = ensemble_probs(PROPOSED, "bdneuro")[keep] if PROPOSED in READY else None
    for name in READY:
        P = ensemble_probs(name, "bdneuro")[keep]; pr = hard(P)
        ba = bal_acc(yb_, pr); ci = np.percentile([bal_acc(yb_[b], pr[b]) for b in boots], [2.5, 97.5])
        row = {"Model": name, "n": int(keep.sum()), "Balanced acc (%)": round(100 * ba, 2),
               "95% CI": f"[{100 * ci[0]:.1f}, {100 * ci[1]:.1f}]",
               "Macro-F1 (%)": round(100 * macro_f1(yb_, pr), 2), "Macro-AUC": round(macro_auc(yb_, P), 4),
               **{f"Recall {c} (%)": round(100 * v, 1) for c, v in per_class_recall(yb_, pr).items()}}
        if P_prop is not None and name != PROPOSED:
            pp = hard(P_prop)
            dci = np.percentile([bal_acc(yb_[b], pp[b]) - bal_acc(yb_[b], pr[b]) for b in boots], [2.5, 97.5])
            row[f"Δ vs {PROPOSED} (pts)"] = f"{100 * (bal_acc(yb_, pp) - ba):+.1f} [{100 * dci[0]:+.1f}, {100 * dci[1]:+.1f}]"
        bd_rows.append(row)
    df_bdn = pd.DataFrame(bd_rows).sort_values("Balanced acc (%)", ascending=False)
    print(f"BDNeuro-MRI, {int(keep.sum())} unique tumour images (no patient IDs -> image-level bootstrap CIs).")
    print(f"Δ = {PROPOSED} minus the model (positive = {PROPOSED} better), with 95% CI.")
    print(df_bdn.to_string(index=False))
    df_bdn.to_csv(OUT_DIR / "tables" / "table_bdneuro.csv", index=False)

# %%
# ─── 7.2d KEY FIGURE: models that rely on the tumour generalise better ───
from scipy.stats import spearmanr
if RUN_EVALUATION and RELIANCE:
    gen = {}
    if "brats" in EVAL_SETS:
        gen["BraTS AUC (glioma vs meningioma)"] = {n: roc_auc_score(*brats_auc_scores(n)[:2]) for n in RELIANCE}
    if "bdneuro" in EVAL_SETS and BD_CLEAN:
        keep = np.isin(paths_of("bdneuro"), [p for p, _ in BD_CLEAN])
        gen["BDNeuro balanced accuracy (%)"] = {n: 100 * bal_acc(labels_of("bdneuro")[keep],
                                                                hard(ensemble_probs(n, "bdneuro")[keep])) for n in RELIANCE}
    if gen:
        fig, ax = plt.subplots(1, len(gen), figsize=(5 * len(gen), 3.8), squeeze=False)
        corr_rows = []
        for a, (lab, g) in zip(ax[0], gen.items()):
            names = [n for n in RELIANCE if np.isfinite(RELIANCE[n])]
            x = np.array([RELIANCE[n] for n in names]); y = np.array([g[n] for n in names])
            rho, pval = spearmanr(x, y)
            corr_rows.append({"External set": lab, "Spearman rho": round(rho, 3), "p": round(pval, 4), "models": len(names)})
            for n, xi, yi in zip(names, x, y):
                a.scatter(xi, yi, s=55 if n == PROPOSED else 30, color=FAMILY_COLOR[family_of(n)],
                          edgecolor="k" if n == PROPOSED else "none", zorder=3)
                a.annotate(n, (xi, yi), fontsize=6, xytext=(3, 3), textcoords="offset points")
            a.set(xlabel="Tumour-reliance score (tumour-hiding test)", ylabel=lab,
                  title=f"Spearman ρ = {rho:.2f} (p = {pval:.3f}, {len(names)} models)")
        fig.suptitle("Is tumour reliance related to generalisation? (each dot = one model; see the honest checks below)", fontsize=9)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_reliance_vs_generalisation.png", bbox_inches="tight"); plt.show()
        df_corr = pd.DataFrame(corr_rows)
        # HONEST CHECK: is the correlation only "pretrained vs scratch"? -> within-family + partial correlation
        fam = {n: ("pretrained" if is_pretrained(n) else "scratch") for n in RELIANCE}
        extra = []
        for lab, g in gen.items():
            names = [n for n in RELIANCE if np.isfinite(RELIANCE[n])]
            if len(names) < 4:
                continue
            for f_ in ["pretrained", "scratch"]:
                nn_ = [n for n in names if fam[n] == f_]
                if len(nn_) >= 4:
                    r_, p_ = spearmanr([RELIANCE[n] for n in nn_], [g[n] for n in nn_])
                    extra.append({"External set": lab, "Models": f"only {f_} ({len(nn_)})", "Spearman rho": round(r_, 3), "p": round(p_, 4)})
            rx = pd.Series([RELIANCE[n] for n in names]).rank().values; ry = pd.Series([g[n] for n in names]).rank().values
            z = np.array([fam[n] == "pretrained" for n in names], float)
            Z = np.column_stack([np.ones_like(z), z])
            res_x = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]; res_y = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
            r_, p_ = sps.pearsonr(res_x, res_y)
            extra.append({"External set": lab, "Models": f"all, controlling for pretrained vs scratch ({len(names)})",
                          "Spearman rho": round(r_, 3), "p": round(p_, 4)})
            # the strongest check: leave out the ROI / Run-5 models (designed to be tumour-reliant) and stay in ONE family
            for lab2, sub in [("pretrained WITHOUT ROI / Run-5 models", [n for n in names if fam[n] == "pretrained"
                                and not n.startswith("ROI") and n not in ("CNX-Transplant", "CNX-Strip")]),
                              ("ConvNeXt backbone only, WITHOUT ROI / Run-5 models", [n for n in names if MODEL_SPECS[n]["kind"] in ("guide", "tamtk")
                                and n not in ("CNX-Transplant", "CNX-Strip")] + (["ConvNeXt-Tiny"] if "ConvNeXt-Tiny" in names else []))]:
                sub = list(dict.fromkeys(sub))
                if len(sub) >= 4:
                    r_, p_ = spearmanr([RELIANCE[n] for n in sub], [g[n] for n in sub])
                    extra.append({"External set": lab, "Models": f"{lab2} ({len(sub)})", "Spearman rho": round(r_, 3), "p": round(p_, 4)})
        df_corr = pd.concat([df_corr.assign(Models="all"), pd.DataFrame(extra)], ignore_index=True)
        print(df_corr[["External set", "Models", "Spearman rho", "p"]].to_string(index=False))
        df_corr.to_csv(OUT_DIR / "tables" / "table_reliance_vs_generalisation.csv", index=False)
        print("\nHow to read: if the correlation disappears inside one backbone family without the ROI models, the overall"
              "\ncorrelation reflects model families, NOT a general law. Report it as descriptive only (Run 4: it disappeared).")

# %%
# ─── 7.3 Figure: per-class recall in every dataset (proposed vs plain baseline vs original MTK) ───
if RUN_EVALUATION and TEST_PROBS:
    show = [n for n in [PROPOSED, "CNX-Plain", "MTK"] if n in READY]
    show = list(dict.fromkeys(show))
    sets = [("Internal test", lambda n: (TEST_Y, TEST_PROBS[n]))]
    if "bdneuro" in EVAL_SETS and BD_CLEAN:
        kb = np.isin(paths_of("bdneuro"), [p for p, _ in BD_CLEAN])
        sets.append(("BDNeuro (new hospital)", lambda n: (labels_of("bdneuro")[kb], ensemble_probs(n, "bdneuro")[kb])))
    if "brats" in EVAL_SETS:
        sets.append(("BraTS (new hospital)", lambda n: (labels_of("brats"), ensemble_probs(n, "brats"))))
    if "mendeley" in EVAL_SETS and MEND_CLEAN:
        km = np.isin(paths_of("mendeley"), [p for p, _ in MEND_CLEAN])
        sets.append(("Mendeley (unique)", lambda n: (labels_of("mendeley")[km], ensemble_probs(n, "mendeley")[km])))
    fig, ax = plt.subplots(1, len(sets), figsize=(3.4 * len(sets), 3.2), sharey=True, squeeze=False)
    w = 0.8 / len(show)
    for a, (title, get) in zip(ax[0], sets):
        for j, n in enumerate(show):
            y_, P_ = get(n); rec = per_class_recall(y_, hard(P_))
            vals = [100 * rec[c] if np.isfinite(rec[c]) else 0 for c in CLASSES]
            a.bar(np.arange(N_CLASSES) + (j - (len(show) - 1) / 2) * w, vals, w, label=n,
                  color=FAMILY_COLOR[family_of(n)] if n != "MTK" else "#888888")
        a.set(xticks=range(N_CLASSES), xticklabels=CLASSES, ylim=(0, 105), title=title)
    ax[0][0].set_ylabel("Recall (%)"); ax[0][0].legend(frameon=False, fontsize=7)
    fig.suptitle("Recall per tumour type in each dataset (BraTS has no pituitary class)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_external_recall.png", bbox_inches="tight"); plt.show()

# %%
# ─── 7.4 Feature space (t-SNE): do images group by TUMOUR TYPE or by SOURCE DATASET? ───
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
TSNE_MODELS = [n for n in ["CNX-Plain", "CTR-Net", "MTK"] if n in READY]       # ROI models see crops, not slices
TSNE_MODELS = list(dict.fromkeys(TSNE_MODELS))
tsne_rows = []
if RUN_EVALUATION and TSNE_MODELS:
    rng_t = np.random.default_rng(0)
    def sub(items, n):
        idx = rng_t.choice(len(items), min(n, len(items)), replace=False); return [items[i] for i in idx]
    src_items = [("Figshare test", TEST_ITEMS)]
    if MEND_CLEAN: src_items.append(("Mendeley (dedup)", sub(MEND_CLEAN, 400)))
    if BRISC_CLEAN: src_items.append(("BRISC (dedup)", BRISC_CLEAN))
    if BRATS_ITEMS: src_items.append(("BraTS", sub(BRATS_ITEMS, 400)))
    if BD_CLEAN: src_items.append(("BDNeuro", sub(BD_CLEAN, 400)))
    items = [(p, y, s) for s, its in src_items for p, y in its]
    fig, ax = plt.subplots(len(TSNE_MODELS), 2, figsize=(8, 3.6 * len(TSNE_MODELS)), squeeze=False)
    for r, n in enumerate(TSNE_MODELS):
        m = load_checkpoint(n, READY[n][0], 1); mean, std = norm_for(n, 1)
        arrs = [load_u8(p) for p, _, _ in items]; feats = []
        with torch.no_grad():
            for i in range(0, len(arrs), 64):
                with torch.autocast(device_type=DEVICE.type, enabled=DEVICE.type == "cuda"):
                    feats.append(embed(m, to_batch(arrs[i:i + 64], mean, std)).float().cpu().numpy())
        del m; feats = np.concatenate(feats)
        src = np.array([s for _, _, s in items]); yy = np.array([y for _, y, _ in items])
        Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(feats)
        for s in dict.fromkeys(src):
            ax[r, 0].scatter(*Z[src == s].T, s=4, alpha=.6, label=s)
        for k, c in enumerate(CLASSES):
            ax[r, 1].scatter(*Z[yy == k].T, s=4, alpha=.6, color=CLASS_COLORS[c], label=c)
        ax[r, 0].set_title(f"{n}: colour = source dataset", fontsize=8); ax[r, 1].set_title(f"{n}: colour = tumour type", fontsize=8)
        for a in ax[r]: a.set(xticks=[], yticks=[]); a.legend(frameon=False, fontsize=6, markerscale=3)
        # how easily can the SOURCE be read from the features? (within glioma + meningioma, present in all sets)
        gm = np.isin(yy, [0, 1])
        acc_src = cross_val_score(LogisticRegression(max_iter=2000), feats[gm], src[gm], cv=5).mean()
        acc_cls = cross_val_score(LogisticRegression(max_iter=2000), feats[gm], yy[gm], cv=5).mean()
        tsne_rows.append({"Model": n, "Source predictable from features (%)": round(100 * acc_src, 1),
                          "Tumour type predictable (%)": round(100 * acc_cls, 1),
                          "Chance (source, %)": round(100 * pd.Series(src[gm]).value_counts(normalize=True).max(), 1)})
    fig.suptitle("t-SNE of penultimate features (fold-1 model, seed 0)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_tsne_source_vs_class.png"); plt.show()
    df_tsne = pd.DataFrame(tsne_rows); print(df_tsne.to_string(index=False))
    df_tsne.to_csv(OUT_DIR / "tables" / "table_feature_source.csv", index=False)

# %% [markdown]
# ## 7.5 The solutions side by side (experiments 1, 1b, 2, 3)
# One table and four figures that answer: *does the solution keep the accuracy, use the tumour, and transfer?*

# %%
# ─── 7.5a Solution table + solution map ───
SOL_MODELS = [n for n in ["ROI-Loc", "ROI-m06-Transplant", "ROI-Crop", "ROI-Crop-m06", "ROI-Crop-m15", "ROI-Mask", "CNX-Transplant", "CNX-Strip",
                          "CNX-Plain", "CTR-Net", "MTL-Seg", "RRR", "CAM-Guide", "ConvNeXt-Tiny", "MTK"] if n in READY]
def cv_f1(n):
    return 100 * np.mean([rec_of(n, s, f)["val_f1"] for s in READY[n] for f in range(1, N_FOLDS + 1)])   # abstain-corrected
SOLV = {}
if RUN_EVALUATION and SOL_MODELS:
    bd_keep = np.isin(paths_of("bdneuro"), [p for p, _ in BD_CLEAN]) if "bdneuro" in EVAL_SETS and BD_CLEAN else None
    d_rel = df_rel.set_index("Model") if RELIANCE else None
    d_r2 = df_rel2.set_index("Model") if len(df_rel2) else None
    d_rc = df_relc.set_index("Model") if RELIANCE else None
    rows = []
    for n in SOL_MODELS:
        r = {"Model": n, "CV macro-F1 (%)": round(cv_f1(n), 2),
             "Test macro-F1 (%)": round(100 * macro_f1(TEST_Y, hard(ensemble_probs(n, "test"))), 2)}
        if d_rel is not None and n in d_rel.index:
            r["Reliance (swap)"] = d_rel.loc[n, "Reliance (swap)"]
            r["Skull/position-only BA (%)"] = d_rel.loc[n, "skull / position only"]
            r["Pituitary correct when hidden (%)"] = d_rc.loc[n, "pituitary still correct (%)"]
        if d_r2 is not None and n in d_r2.index:
            r["Reliance (other patient)"] = d_r2.loc[n, "R other patient 9"]; r["Reliance (swap 41)"] = d_r2.loc[n, "R swap 41"]
        if "brats" in EVAL_SETS:
            r["BraTS AUC"] = round(roc_auc_score(*brats_auc_scores(n)[:2]), 4)
        if bd_keep is not None:
            r["BDNeuro BA (%)"] = round(100 * bal_acc(labels_of("bdneuro")[bd_keep], hard(ensemble_probs(n, "bdneuro")[bd_keep])), 1)
        if "mendeley" in EVAL_SETS:
            r["Mendeley macro-F1 (%)"] = round(100 * macro_f1(labels_of("mendeley"), hard(ensemble_probs(n, "mendeley"))), 1)
        if "bfree" in EVAL_SETS:
            r["Confident on tumour-free (%)"] = round(100 * (ensemble_probs(n, "bfree", calibrated=True).max(1) >= 0.85).mean(), 1)
        rows.append(r)
    df_sol = pd.DataFrame(rows); SOLV = df_sol.set_index("Model").to_dict("index")
    print("ALL SOLUTIONS AND REFERENCES (seed 0 ensembles; CV over all trained seeds). Goal: high accuracy AND high reliance")
    print("AND high BraTS AUC AND low confidence on tumour-free slices.")
    print(df_sol.to_string(index=False)); df_sol.to_csv(OUT_DIR / "tables" / "table_solutions.csv", index=False)

    if "Reliance (swap)" in df_sol:
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.add_patch(plt.Rectangle((0.6, 96), 0.42, 3.5, color="#2a9d8f", alpha=0.10, lw=0))
        ax.text(0.61, 99.2, "goal: uses the tumour AND accurate", fontsize=7, color="#2a9d8f", va="top")
        for _, r in df_sol.iterrows():
            if pd.isna(r.get("Reliance (swap)")):
                continue
            sz = 40 + 1200 * max(r.get("BraTS AUC", 0.8) - 0.8, 0)
            ax.scatter(r["Reliance (swap)"], r["CV macro-F1 (%)"], s=sz, color=FAMILY_COLOR[family_of(r.Model)],
                       edgecolor="k" if r.Model == PROPOSED else "white", lw=1.2, zorder=3, alpha=0.9)
            ax.annotate(r.Model, (r["Reliance (swap)"], r["CV macro-F1 (%)"]), fontsize=6.5, xytext=(5, -3),
                        textcoords="offset points", annotation_clip=True)
        f1s = df_sol["CV macro-F1 (%)"].dropna()
        ax.set(xlabel="Tumour-reliance score  (0 = ignores the tumour, 1 = needs it)", ylabel="Cross-validation macro-F1 (%)",
               xlim=(0, 1.02), ylim=(min(88, f1s.min() - 1), 100),
               title="Solution map: accuracy vs. use of the tumour (dot size = BraTS AUC)")
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_solution_map.png", bbox_inches="tight"); plt.show()

# %%
# ─── 7.5b Context sweep: how much surrounding tissue turns into a shortcut? (experiment 1b) ───
SWEEP = [(m, lab) for m, lab in [("ROI-Crop", "tight box\n(margin 0.2)"), ("ROI-Crop-m06", "margin 0.6"),
                                 ("ROI-Crop-m15", "margin 1.5"), ("CNX-Plain", "whole slice")] if m in SOLV]
if RUN_EVALUATION and len(SWEEP) >= 2 and "Reliance (swap)" in df_sol:
    xs = np.arange(len(SWEEP)); g = lambda col: [SOLV[m].get(col, np.nan) for m, _ in SWEEP]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(xs, g("CV macro-F1 (%)"), "-o", color="#30638e", label="CV macro-F1 (%)")
    ax[0].plot(xs, 100 * np.array(g("BraTS AUC")), "-s", color="#2a9d8f", label="BraTS AUC (x100)")
    lo_ = np.nanmin(np.r_[g("CV macro-F1 (%)"), 100 * np.array(g("BraTS AUC"))])
    ax[0].set(xticks=xs, xticklabels=[l for _, l in SWEEP], ylabel="%", title="Accuracy and transfer",
              ylim=(min(85, lo_ - 2), 100))
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(xs, g("Reliance (swap)"), "-o", color="#d1495b", label="tumour reliance")
    ax[1].plot(xs, np.array(g("Skull/position-only BA (%)")) / 100, "-^", color="#edae49", label="skull/position-only BA (/100)")
    ax[1].axhline(1 / 3, color="k", ls="--", lw=0.8); ax[1].text(xs[-1], 0.35, "guessing", fontsize=6, ha="right")
    if "ROI-m06-Transplant" in SOLV and "ROI-Crop-m06" in dict(SWEEP):
        xm = [m for m, _ in SWEEP].index("ROI-Crop-m06"); c6 = SOLV["ROI-m06-Transplant"]
        ax[0].scatter([xm], [c6["CV macro-F1 (%)"]], marker="*", s=160, color="#264653", zorder=4, label="margin 0.6 + transplant (CV)")
        ax[0].scatter([xm], [100 * c6.get("BraTS AUC", np.nan)], marker="*", s=160, color="#2a9d8f", zorder=4)
        ax[1].scatter([xm], [c6.get("Reliance (swap)", np.nan)], marker="*", s=160, color="#264653", zorder=4,
                      label="margin 0.6 + transplant")
        ax[0].legend(frameon=False, fontsize=7)
    ax[1].set(xticks=xs, xticklabels=[l for _, l in SWEEP], ylim=(0, 1), title="Use of the tumour vs. shortcut")
    ax[1].legend(frameon=False, fontsize=7)
    fig.suptitle("Context sweep: the classifier sees more and more of the slice around the found tumour", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_context_sweep.png", bbox_inches="tight"); plt.show()
elif RUN_EVALUATION:
    print("Context sweep: needs at least ROI-Crop + one wider-margin model (not all trained yet -> next run).")

# %%
# ─── 7.5c ROI-Loc: what does each input contribute? (appearance vs location; location knock-out) ───
if RUN_EVALUATION and "ROI-Loc" in READY and MASKS:
    dloc = df3[df3.filepath.isin(MASKS)].reset_index(drop=True)
    Xl = np.array([loc_feats(load_u8(p), MASKS[p]) for p in dloc.filepath]); yl = dloc.label.values
    dev_m = (dloc.role == "dev").values
    lr_loc = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    pr_cv = cross_val_predict(lr_loc, Xl[dev_m], yl[dev_m], groups=dloc.pid.values[dev_m], cv=GroupKFold(5))
    lr_loc.fit(Xl[dev_m], yl[dev_m]); pr_te = lr_loc.predict(Xl[~dev_m])
    # knock-out: replace every location by the average training location -> how much accuracy is lost?
    s0 = READY["ROI-Loc"][0]; mean_loc = torch.tensor(Xl[dev_m].mean(0), dtype=torch.float32)
    ko = []
    for f in range(1, N_FOLDS + 1):
        m = load_checkpoint("ROI-Loc", s0, f); mean, std = norm_for("ROI-Loc", f)
        m.loc_override = mean_loc; ko.append(softmax_np(predict_any("ROI-Loc", f, m, ARRS["test"], mean, std))); m.loc_override = None
    P_ko = np.mean(ko, 0)
    parts = [("location only\n(6 numbers, no image)", 100 * macro_f1(yl[dev_m], pr_cv), 100 * macro_f1(yl[~dev_m], pr_te)),
             ("appearance only\n(ROI-Crop)", SOLV.get("ROI-Crop", {}).get("CV macro-F1 (%)", np.nan),
              SOLV.get("ROI-Crop", {}).get("Test macro-F1 (%)", np.nan)),
             ("ROI-Loc with location\nknocked out", np.nan, 100 * macro_f1(TEST_Y, hard(P_ko))),
             ("both\n(ROI-Loc)", SOLV["ROI-Loc"]["CV macro-F1 (%)"], SOLV["ROI-Loc"]["Test macro-F1 (%)"])]
    df_parts = pd.DataFrame(parts, columns=["Input", "CV macro-F1 (%)", "Test macro-F1 (%)"]).round(2)
    print("What each input of ROI-Loc contributes (location-only uses the EXPERT outline = an upper bound):")
    print(df_parts.assign(Input=df_parts.Input.str.replace("\n", " ")).to_string(index=False))
    df_parts.to_csv(OUT_DIR / "tables" / "table_roiloc_parts.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.4)); xs = np.arange(len(df_parts))
    ax.bar(xs - 0.2, df_parts["CV macro-F1 (%)"], 0.4, color="#30638e", label="cross-validation")
    ax.bar(xs + 0.2, df_parts["Test macro-F1 (%)"], 0.4, color="#2a9d8f", label="held-out test")
    for x_, v1, v2 in zip(xs, df_parts["CV macro-F1 (%)"], df_parts["Test macro-F1 (%)"]):
        for dx, v in [(-0.2, v1), (0.2, v2)]:
            if np.isfinite(v): ax.text(x_ + dx, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.5)
    ax.set(xticks=xs, xticklabels=df_parts.Input, ylabel="Macro-F1 (%)", ylim=(0, 104),
           title="ROI-Loc = what the tumour looks like + where it is (and nothing else)")
    ax.axhline(100 / 3, color="k", ls="--", lw=0.8); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_roiloc_parts.png", bbox_inches="tight"); plt.show()

# %%
# ─── 7.5d Transplant (solution 2) and skull control (experiment 3) ───
if RUN_EVALUATION and RELIANCE:
    pair = [n for n in ["CNX-Plain", "CNX-Transplant"] if n in REL]
    if len(pair) == 2:
        fig, ax = plt.subplots(figsize=(7.5, 3.3)); xs = np.arange(N_CLASSES)
        for k_, n in enumerate(pair):
            v = [df_relc.set_index("Model").loc[n, f"{c} still correct (%)"] for c in CLASSES]
            ax.bar(xs + (k_ - 0.5) * 0.38, v, 0.38, color=FAMILY_COLOR[family_of(n)], label=n)
            for x_, vv in zip(xs, v): ax.text(x_ + (k_ - 0.5) * 0.38, vv + 1, f"{vv:.0f}", ha="center", fontsize=7)
        ax.axhline(100 / 3, color="k", ls="--", lw=0.8)
        ax.set(xticks=xs, xticklabels=CLASSES, ylabel="Still correct with the tumour HIDDEN (%)", ylim=(0, 105),
               title="Tumour transplant: does the ordinary model stop answering without the tumour?")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout(); fig.savefig(FIG_DIR / "fig_transplant_effect.png", bbox_inches="tight"); plt.show()
if RUN_EVALUATION and "brats" in EVAL_SETS and "CNX-Strip" in READY:
    refs = [n for n in ["ROI-Crop", "ROI-Loc"] if n in READY]
    df_strip = pd.concat([brats_paired(r, ["CNX-Strip"]) for r in refs] +
                         ([brats_paired("CNX-Strip", ["CNX-Plain"])] if "CNX-Plain" in READY else []), ignore_index=True)
    print("SKULL CONTROL on BraTS (glioma-vs-meningioma AUC differences, patient bootstrap):")
    print(df_strip.to_string(index=False)); df_strip.to_csv(OUT_DIR / "tables" / "table_skull_control.csv", index=False)
    print("How to read: if 'CNX-Strip − CNX-Plain' is large and 'ROI-Crop − CNX-Strip' is ~0, ROI's BraTS gain came mostly from"
          "\nremoving the skull. If 'ROI-Crop − CNX-Strip' stays clearly positive, the gain is NOT explained by the skull.")

# %% [markdown]
# ## 8. Calibration and high-confidence predictions (internal test)

# %%
# ─── 8.1 ECE / Brier before and after temperature scaling ───
def ece_score(y, P, bins=15):
    conf, pred = P.max(1), hard(P); edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs((pred[m] == y[m]).mean() - conf[m].mean())
    return float(e)

def brier(y, P):
    return float(((P - np.eye(N_CLASSES)[y]) ** 2).sum(1).mean())

cal_rows = []
if RUN_EVALUATION and len(TEST_Y):
    for name in READY:
        for cal in [False, True]:
            P = ensemble_probs(name, "test", calibrated=cal); conf = P.max(1); ok = hard(P) == TEST_Y
            hi = conf >= CONF_THRESHOLD
            cal_rows.append({"Model": name, "Temperature scaled": cal, "ECE": round(ece_score(TEST_Y, P), 4),
                             "Brier": round(brier(TEST_Y, P), 4),
                             f"Coverage ≥{CONF_THRESHOLD} (%)": round(100 * hi.mean(), 1),
                             f"Precision ≥{CONF_THRESHOLD} (%)": round(100 * ok[hi].mean(), 1) if hi.any() else np.nan,
                             "Confident errors": int((hi & ~ok).sum())})
    df_cal = pd.DataFrame(cal_rows)
    print(df_cal.to_string(index=False))
    df_cal.to_csv(OUT_DIR / "tables" / "table_calibration.csv", index=False)

    show = list(dict.fromkeys(n for n in [PROPOSED, "CNX-Plain", "MTK"] if n in READY))
    fig, ax = plt.subplots(1, len(show), figsize=(3.2 * len(show), 3.1), squeeze=False)
    for a, name in zip(ax[0], show):
        for cal, col in [(False, "#bbbbbb"), (True, "#30638e")]:
            P = ensemble_probs(name, "test", calibrated=cal); conf, ok = P.max(1), hard(P) == TEST_Y
            edges = np.linspace(0, 1, 11); cx, cy = [], []
            for lo, hi_ in zip(edges[:-1], edges[1:]):
                m = (conf > lo) & (conf <= hi_)
                if m.sum() >= 5:
                    cx.append(conf[m].mean()); cy.append(ok[m].mean())
            a.plot(cx, cy, "o-", color=col, label=("TS " if cal else "raw ") + f"ECE {ece_score(TEST_Y, P):.3f}")
        a.plot([0, 1], [0, 1], ":", color="k"); a.set(title=name, xlabel="Confidence", ylabel="Accuracy")
        a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_calibration.png"); plt.show()

# %%
# ─── 8.2 Conformal prediction (cross-conformal): "confident answer" or "send to a radiologist" ───
# Each fold model is calibrated on its own validation fold (temperature-scaled). For a new image, every fold model gives
# a conformal p-value per class, and the p-values are pooled (cross-conformal, Vovk 2015). Class k is in the
# prediction set if its pooled p-value > alpha. One class in the set = confident answer; 2-3 classes = refer.
# Guarantee: on data like the development data the true class is in the set >= 1 - 2*alpha (in practice ~1 - alpha).
# New hospitals are a robustness test (no guarantee there).
ALPHAS = [0.05, 0.10]
def conformal_sets(name, set_name, alpha, keep=None):
    s = READY[name][0]; cal, test_scores = [], []
    for f in range(1, N_FOLDS + 1):
        _, rec_path, oof_path = run_paths(name, s, f)
        y = np.array(load_json(rec_path)["val_labels"]); T = TEMPS[(name, s, f)]
        Pv = softmax_np(np.load(oof_path) / T); cal.append(np.sort(1 - Pv[np.arange(len(y)), y]))
        Pt = softmax_np(np.load(pred_path(name, s, f, set_name)) / T)
        test_scores.append(1 - (Pt if keep is None else Pt[keep]))
    N = sum(len(c) for c in cal)
    counts = sum(len(c) - np.searchsorted(c, ts, side="left") for c, ts in zip(cal, test_scores))   # #{cal >= score}
    pval = (counts + 1) / (N + 1)
    return pval > alpha

conf_rows = []
if RUN_EVALUATION and len(TEST_Y):
    eval_sets = {"Internal test": ("test", None)}
    if "brats" in EVAL_SETS:
        eval_sets["BraTS"] = ("brats", None)
    if "bdneuro" in EVAL_SETS and BD_CLEAN:
        eval_sets["BDNeuro (unique)"] = ("bdneuro", np.isin(paths_of("bdneuro"), [p for p, _ in BD_CLEAN]))
    for name in READY:
        for alpha in ALPHAS:
            for set_label, (s, keep) in eval_sets.items():
                S = conformal_sets(name, s, alpha, keep); y = labels_of(s) if keep is None else labels_of(s)[keep]
                size = S.sum(1); single = size == 1
                P = ensemble_probs(name, s, calibrated=True); P = P if keep is None else P[keep]
                conf_rows.append({"Model": name, "alpha": alpha, "Set": set_label, "n": len(y),
                                  "Coverage (%)": round(100 * S[np.arange(len(y)), y].mean(), 1),
                                  "Target (%)": round(100 * (1 - alpha)),
                                  "Answered (1 label) (%)": round(100 * single.mean(), 1),
                                  "Accuracy of answers (%)": round(100 * (S[single].argmax(1) == y[single]).mean(), 1) if single.any() else np.nan,
                                  "Referred (%)": round(100 * (~single).mean(), 1)})
    df_conf = pd.DataFrame(conf_rows)
    print("Conformal referral, alpha = 0.05 (target: true class inside the answer set >= 95%):")
    print(df_conf[(df_conf.alpha == 0.05) & df_conf.Model.isin([PROPOSED, "CNX-Plain", "MTL-Seg", "RRR", "CAM-Guide", "MTK"])]
          .to_string(index=False))
    df_conf.to_csv(OUT_DIR / "tables" / "table_conformal.csv", index=False)

    show = list(dict.fromkeys(n for n in [PROPOSED, "CNX-Plain", "MTK"] if n in READY))
    d = df_conf[(df_conf.alpha == 0.05) & df_conf.Model.isin(show)].copy()
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for a, col, title in zip(ax, ["Coverage (%)", "Answered (1 label) (%)", "Accuracy of answers (%)"],
                             ["Coverage (target 95%)", "Answered without referral", "Accuracy of the answered cases"]):
        d.pivot(index="Set", columns="Model", values=col)[show].plot.bar(ax=a, rot=0, width=.8,
                                                                         color=[FAMILY_COLOR[family_of(n)] if n != "MTK" else "#888888" for n in show])
        a.set(title=title, ylabel="%", ylim=(0, 105), xlabel=""); a.legend(frameon=False, fontsize=6)
        if col == "Coverage (%)":
            a.axhline(95, color="k", ls="--", lw=1)
    fig.suptitle("Conformal referral (alpha = 0.05): answer when sure, refer the rest", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_conformal.png", bbox_inches="tight"); plt.show()

# %%
# ─── 8.3 Referral using the tumour map: refer when the model cannot find a tumour ───
# Risk-coverage curve: answer the most trustworthy X% of cases, error rate among them. Lower curve = safer.
# CTR-Net can use two signals: softmax confidence, and confidence x "tumour found" (peak of its tumour map).
def risk_coverage(score, correct):
    o = np.argsort(-score); err = 1 - correct[o].astype(float)
    cov = np.arange(1, len(o) + 1) / len(o); risk = np.cumsum(err) / np.arange(1, len(o) + 1)
    return cov, risk, float(np.mean(risk))                      # AURC = mean risk over all coverages

sel_rows = []
if RUN_EVALUATION:
    sets = [("Internal test", "test", None)]
    if "brats" in EVAL_SETS: sets.append(("BraTS", "brats", None))
    if "bdneuro" in EVAL_SETS and BD_CLEAN: sets.append(("BDNeuro (unique)", "bdneuro", None))
    show = list(dict.fromkeys(n for n in [PROPOSED, "ROI-m06-Transplant", "ROI-Crop-m06", "CNX-Transplant", "CNX-Plain", "CTR-Net", "MTL-Seg"] if n in READY))
    fig, ax = plt.subplots(1, len(sets), figsize=(3.8 * len(sets), 3.3), squeeze=False)
    for a, (lab, s, keep) in zip(ax[0], sets):
        y = labels_of(s)
        for n in show:
            P = ensemble_probs(n, s, calibrated=True)
            cov, risk, aurc = risk_coverage(P.max(1), hard(P) == y)
            a.plot(100 * cov, 100 * risk, lw=1.8 if n == PROPOSED else 1,
                   color={"ROI-Crop": "#f08a8f", "ROI-Crop-m06": "#f4a261", "ROI-m06-Transplant": "#264653", "CTR-Net": "#9b5de5",
                          "MTL-Seg": "#edae49"}.get(n, FAMILY_COLOR[family_of(n)]),
                   label=f"{n} (AURC {100 * aurc:.1f})")
            sel_rows.append({"Set": lab, "Model": n, "AURC (%)": round(100 * aurc, 2),
                             "Error when answering 80% (%)": round(100 * risk[int(0.8 * len(risk)) - 1], 1)})
        a.set(title=lab, xlabel="Cases answered (%)", ylabel="Error among answered (%)"); a.legend(frameon=False, fontsize=6)
    fig.suptitle("Selective prediction: lower curve = fewer mistakes for the same number of answered cases", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_risk_coverage.png", bbox_inches="tight"); plt.show()
    df_sel = pd.DataFrame(sel_rows); print(df_sel.to_string(index=False))
    df_sel.to_csv(OUT_DIR / "tables" / "table_selective_prediction.csv", index=False)
    print("AURC = average error over all referral levels (lower = better).")

# %%
# ─── 8.4 Tumour-free slices (BraTS): does the model still confidently name a tumour type? ───
# Real scans contain many slices WITHOUT tumour. A trustworthy model should be unsure there (or refer).
# A shortcut model gives a confident tumour type anyway. Same BraTS patients as the tumour slices above.
free_rows = []
if RUN_EVALUATION and "bfree" in EVAL_SETS:
    for n in READY:
        Pf = ensemble_probs(n, "bfree", calibrated=True); Pt = ensemble_probs(n, "brats", calibrated=True)
        Sf = conformal_sets(n, "bfree", 0.05)
        hf = hard(Pf); dist = np.bincount(hf[hf >= 0], minlength=N_CLASSES) / len(Pf)
        free_rows.append({"Model": n,
                          "Confident (>=0.85) on tumour-free (%)": round(100 * (Pf.max(1) >= 0.85).mean(), 1),
                          "Mean confidence on tumour-free": round(float(Pf.max(1).mean()), 3),
                          "Mean confidence on tumour slices": round(float(Pt.max(1).mean()), 3),
                          "AUROC tumour vs tumour-free (by confidence)": round(roc_auc_score(
                              np.r_[np.ones(len(Pt)), np.zeros(len(Pf))], np.r_[Pt.max(1), Pf.max(1)]), 3),
                          "Referred by conformal (%)": round(100 * (Sf.sum(1) != 1).mean(), 1),
                          **{f"called {c} (%)": round(100 * dist[k], 1) for k, c in enumerate(CLASSES)},
                          "abstained (no tumour found) (%)": round(100 * (hf < 0).mean(), 1)})
    df_free = pd.DataFrame(free_rows).sort_values("Confident (>=0.85) on tumour-free (%)")
    print(f"{len(EVAL_SETS['bfree'])} tumour-free BraTS slices. Lower 'confident' and higher 'referred' = more trustworthy.")
    print("AUROC column = can the model's own confidence tell real tumour slices from tumour-free ones? (0.5 = no, 1 = perfectly)")
    print(df_free.to_string(index=False)); df_free.to_csv(OUT_DIR / "tables" / "table_tumour_free.csv", index=False)
    order = list(df_free.Model)
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(order)), 3.6))
    xs = np.arange(len(order))
    ax.bar(xs - 0.2, df_free.set_index("Model").loc[order, "Confident (>=0.85) on tumour-free (%)"], 0.4,
           color=[FAMILY_COLOR[family_of(n)] for n in order], label="confident on TUMOUR-FREE slices (should be low)")
    ax.bar(xs + 0.2, 100 * df_free.set_index("Model").loc[order, "Mean confidence on tumour slices"], 0.4,
           color="#cccccc", label="mean confidence on tumour slices (reference)")
    ax.set_xticks(xs); ax.set_xticklabels(order, rotation=35, ha="right", fontsize=7)
    ax.set(ylabel="%", ylim=(0, 105), title="Do models confidently name a tumour type when there is NO tumour?")
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_tumour_free.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 8.5 Shortcut alarm (experiment 4): a per-image safety check that needs no training
# For every new image: find the tumour with the tumour map (MTL-Seg, same fold), **hide it**, and predict again.
# * If the confidence in the answer **drops** a lot → the model used the tumour → answer.
# * If the answer **does not change** (or no tumour was found) → the model answered from the surroundings → **refer**.
#
# This turns the tumour-hiding test into a tool that can run on every case. Compared with plain confidence on the
# internal test, BraTS, BDNeuro and the tumour-free slices.

# %%
# ─── 8.5 Shortcut alarm ───
ALARM_MODELS = [n for n in ["CNX-Plain", "CNX-Transplant", "CTR-Net", "MTL-Seg", "CAM-Guide", "ConvNeXt-Tiny"] if n in READY]
ALARM_SETS = [s for s in ["test", "brats", "bdneuro", "bfree"] if s in EVAL_SETS]
ALARM = {}
if RUN_EVALUATION and ALARM_MODELS and MASKS:
    banner("8.5 Shortcut alarm", f"{len(ALARM_MODELS)} models x {len(ALARM_SETS)} sets (cached after the first run)")
    (OUT_DIR / "alarm").mkdir(exist_ok=True)
    alarm_path = lambda n, s_, f, st: OUT_DIR / "alarm" / f"{n}_s{s_}_f{f}_{st}.npy"
    FOUND = {st: np.zeros(len(EVAL_SETS[st]), bool) for st in ALARM_SETS}
    for f in range(1, N_FOLDS + 1):
        need = [(n, st) for n in ALARM_MODELS for st in ALARM_SETS if not cached_ok(alarm_path(n, READY[n][0], f, st), len(EVAL_SETS[st]))]
        found_file = OUT_DIR / "alarm" / f"found_f{f}.npz"
        if not need and found_file.exists():
            z = np.load(found_file)
            for st in ALARM_SETS: FOUND[st] |= z[st]
            continue
        rng_a = np.random.default_rng(100 + f); HID, fnd = {}, {}
        for st in ALARM_SETS:
            maps = roi_seg_maps(ARRS[st], f); hs, fl = [], []
            for a, mp in zip(ARRS[st], maps):
                b = box_from_mask(mp >= 0.5)
                if b is None:
                    hs.append(a); fl.append(False); continue
                reg = cv2.dilate(b[4].astype(np.uint8), HIDE_KER).astype(bool); bm = brain_mask(a)
                sh = shifted_region(reg, bm, rng_a)
                if sh is None:
                    hs.append(cv2.inpaint(a, reg.astype(np.uint8) * 255, 7, cv2.INPAINT_TELEA))
                else:
                    dy, dx, _ = sh; ys, xs = np.where(reg); h_ = a.copy(); h_[ys, xs] = a[ys + dy, xs + dx]; hs.append(h_)
                fl.append(True)
            HID[st] = hs; fnd[st] = np.array(fl); FOUND[st] |= fnd[st]
        np.savez_compressed(found_file, **fnd)
        for n, st in need:
            m = load_checkpoint(n, READY[n][0], f); mean, std = norm_for(n, f); _CUR_SET["name"] = st
            np.save(alarm_path(n, READY[n][0], f, st), predict_any(n, f, m, HID[st], mean, std))
        _CUR_SET["name"] = None; del HID; free_mem()
        print(f"  fold {f} done", flush=True)

    def alarm_probs(n, st):
        s_ = READY[n][0]
        return np.mean([softmax_np(np.load(alarm_path(n, s_, f, st)) / TEMPS[(n, s_, f)]) for f in range(1, N_FOLDS + 1)], 0)

    rows, curves = [], {}
    for n in ALARM_MODELS:
        for st in ALARM_SETS:
            P = ensemble_probs(n, st, calibrated=True); Ph = alarm_probs(n, st); yhat = hard(P)
            conf = P.max(1); drop = conf - Ph[np.arange(len(P)), yhat]
            ALARM[(n, st)] = dict(conf=conf, drop=drop, combo=np.where(drop >= ALARM_DROP, conf, conf - 1), yhat=yhat)
        for st, lab in [("test", "Internal test"), ("brats", "BraTS"), ("bdneuro", "BDNeuro (unique)")]:
            if st not in ALARM_SETS:
                continue
            keep = np.isin(paths_of(st), [p for p, _ in BD_CLEAN]) if st == "bdneuro" else np.ones(len(EVAL_SETS[st]), bool)
            A = ALARM[(n, st)]; corr = (A["yhat"] == labels_of(st))[keep]
            r = {"Model": n, "Set": lab}
            for sc in ["conf", "drop", "combo"]:
                cov, risk, aurc = risk_coverage(A[sc][keep], corr); r[f"AURC {sc} (%)"] = round(100 * aurc, 2)
                if st == "brats": curves[(n, sc)] = (cov, risk)
            ans = A["drop"][keep] >= ALARM_DROP
            r["Answered by alarm rule (%)"] = round(100 * ans.mean(), 1)
            r["Accuracy of answered (%)"] = round(100 * corr[ans].mean(), 1) if ans.any() else np.nan
            r["Accuracy of all (%)"] = round(100 * corr.mean(), 1)
            rows.append(r)
        if "bfree" in ALARM_SETS and "brats" in ALARM_SETS:
            t_, f_ = ALARM[(n, "brats")], ALARM[(n, "bfree")]; yy = np.r_[np.ones(len(t_["conf"])), np.zeros(len(f_["conf"]))]
            rows.append({"Model": n, "Set": "BraTS tumour vs tumour-free",
                         "AUROC conf": round(roc_auc_score(yy, np.r_[t_["conf"], f_["conf"]]), 3),
                         "AUROC drop": round(roc_auc_score(yy, np.r_[t_["drop"], f_["drop"]]), 3),
                         "AUROC combo": round(roc_auc_score(yy, np.r_[t_["combo"], f_["combo"]]), 3),
                         "Tumour-free answered by alarm rule (%)": round(100 * (f_["drop"] >= ALARM_DROP).mean(), 1),
                         "Tumour-free confident >= 0.85 (%)": round(100 * (f_["conf"] >= 0.85).mean(), 1)})
    df_alarm = pd.DataFrame(rows)
    print("conf = plain confidence | drop = how much the confidence falls when the found tumour is hidden |"
          f"\ncombo = answer only if drop >= {ALARM_DROP} (then rank by confidence). AURC: lower = safer. AUROC: higher = better.")
    a1 = df_alarm[df_alarm.Set != "BraTS tumour vs tumour-free"].dropna(axis=1, how="all")
    a2 = df_alarm[df_alarm.Set == "BraTS tumour vs tumour-free"].dropna(axis=1, how="all")
    print(a1.to_string(index=False)); print("\nCan the score tell real tumour slices from tumour-free slices?")
    print(a2.to_string(index=False))
    df_alarm.to_csv(OUT_DIR / "tables" / "table_shortcut_alarm.csv", index=False)

    ref = "CNX-Plain" if "CNX-Plain" in ALARM_MODELS else ALARM_MODELS[0]
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    for sc, col, lab in [("conf", "#888888", "confidence only"), ("drop", "#edae49", "confidence drop when tumour hidden"),
                         ("combo", "#d1495b", f"alarm rule (drop >= {ALARM_DROP}) + confidence")]:
        if (ref, sc) in curves:
            cov, risk = curves[(ref, sc)]; ax[0].plot(100 * cov, 100 * risk, color=col, label=lab)
    ax[0].set(xlabel="BraTS cases answered (%)", ylabel="Error among answered (%)", title=f"{ref} on BraTS: risk vs coverage")
    ax[0].legend(frameon=False, fontsize=6.5)
    if "bfree" in ALARM_SETS:
        t_, f_ = ALARM[(ref, "brats")], ALARM[(ref, "bfree")]
        bins = np.linspace(-0.4, 1, 29)
        ax[1].hist(t_["drop"], bins, alpha=0.6, color="#30638e", density=True, label="slices WITH tumour")
        ax[1].hist(f_["drop"], bins, alpha=0.6, color="#d1495b", density=True, label="tumour-FREE slices")
        ax[1].axvline(ALARM_DROP, color="k", ls="--", lw=1)
        ax[1].set(xlabel="Confidence drop when the found tumour is hidden", ylabel="density", title=f"{ref}: the alarm signal")
        ax[1].legend(frameon=False, fontsize=6.5)
        d_ = a2.set_index("Model")
        xs = np.arange(len(d_)); w = 0.38
        ax[2].bar(xs - w / 2, d_["AUROC conf"], w, color="#888888", label="confidence")
        ax[2].bar(xs + w / 2, d_["AUROC combo"], w, color="#d1495b", label="alarm rule")
        ax[2].axhline(0.5, color="k", ls="--", lw=0.8)
        ax[2].set(xticks=xs, ylim=(0.3, 1), ylabel="AUROC", title="Tumour vs tumour-free slices")
        ax[2].set_xticklabels(d_.index, rotation=30, ha="right", fontsize=7); ax[2].legend(frameon=False, fontsize=6.5)
    fig.suptitle("Shortcut alarm: hide the found tumour at test time - if the answer does not change, refer the case", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG_DIR / "fig_shortcut_alarm.png", bbox_inches="tight"); plt.show()
else:
    df_alarm = pd.DataFrame()
